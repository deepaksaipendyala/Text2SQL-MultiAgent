from __future__ import annotations
import httpx
import json, os, sqlite3, textwrap, argparse, re, logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable

from dotenv import load_dotenv
load_dotenv()

# ───────────────────────────────────────── logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
log = logging.getLogger("main")
log.info("Booting HR NL→SQL pipeline at level %s", LOG_LEVEL)

# ───────────────────────────────────────── LLM helpers
from langchain_openai import ChatOpenAI

_CACHE: Dict[str, Any] = {}

def get_oa_chat(model: str = "gpt-4o-mini", temperature: float = 0.1):
    log.debug("Creating OpenAI chat model %s, temp=%s", model, temperature)
    return ChatOpenAI(model=model, temperature=temperature)

# ───────────────────────────────────────── demo HR DB
def bootstrap_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE candidates (
          id INTEGER PRIMARY KEY,
          name TEXT,
          email TEXT,
          location TEXT
        );
        CREATE TABLE applications (
          id INTEGER PRIMARY KEY,
          candidate_id INTEGER,
          job_title TEXT,
          status TEXT
        );
        CREATE TABLE experience (
          id INTEGER PRIMARY KEY,
          candidate_id INTEGER,
          company TEXT,
          role TEXT,
          years INTEGER
        );
        CREATE TABLE skills (
          id INTEGER PRIMARY KEY,
          candidate_id INTEGER,
          skill TEXT
        );
        INSERT INTO candidates VALUES
          (1,'alice','alice@mail.com','ny'),
          (2,'bob','bob@mail.com','sf'),
          (3,'cara','cara@mail.com','ny'),
          (4,'david','david@mail.com','ny');
        INSERT INTO applications VALUES
          (1,1,'data analyst','pending'),
          (2,2,'ml Engineer','accepted'),
          (3,3,'data scientist','rejected'),
          (4,4,'senior data scientist','pending');
        INSERT INTO experience VALUES
          (1,1,'companyA','analyst',2),
          (2,2,'companyB','engineer',3),
          (3,3,'companyC','data scientist',4),
          (4,4,'companyD','senior data scientist',6);
        INSERT INTO skills VALUES
          (1,1,'excel'),
          (2,2,'pytorch'),
          (3,3,'sql'),
          (4,4,'python'),
          (5,4,'sql');
        """
    )
    return conn

SCHEMA_INFO = "\n".join(
    r[0] for r in bootstrap_db().execute("SELECT sql FROM sqlite_master WHERE type='table';")
)

# ───────────────────────────────────────── LangGraph state
@dataclass
class State:
    question: str
    rewrites: List[str] = field(default_factory=list)
    duckdb_sql: List[str] = field(default_factory=list)
    gpt_sql: List[str] = field(default_factory=list)
    final_sql: str | None = None
    result: Any = None

# ───────────────────────────────────────── Generators
PROMPT_DUCK = textwrap.dedent("""
### Instruction: Translate to valid DuckDB SQL.
### Schema:
{schema}
### Question:
{q}
Rules you must follow:
- Use **only** the columns and tables listed in the schema.
- Wrap all string literals in single quotes.
- Match exact lower or upper case of column names as per DB Schema.
- Do **not** invent column or table names.
- Remember: **all text values in the DB are lowercase.** Use exact lowercase literals or `LOWER(column)` during comparison.
- If unsure, fall back to `SELECT * FROM <table>`.
### SQL:
""")

PROMPT_GPT = textwrap.dedent(f"""
You are a helpful AI assistant that translates natural language questions into **strictly valid SQLite SQL queries**.

Rules you must follow:
- Use **only** the columns and tables listed in the schema.
- Wrap all string literals in single quotes.
- Match exact lower or upper case of column names as per DB Schema.
- Do **not** invent column or table names.
- Remember: **all text values in the DB are lowercase.** Use exact lowercase literals or `LOWER(column)` during comparison.
- If unsure, fall back to `SELECT * FROM <table>`.
                             
SCHEMA:
{SCHEMA_INFO}

QUESTION: "{{q}}"

Your response must contain **only** valid SQLite SQL. Do not include explanation or markdown formatting.
""")


OLLAMA_MODEL = "duckdb-nsql:7b-q4_K_M"

def DUCK_GEN(q: str) -> str:
    prompt = PROMPT_DUCK.format(schema=SCHEMA_INFO, q=q)
    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=60.0,
    )
    response.raise_for_status()
    out = response.json()
    return out.get("response", "").strip()

GPT_GEN = lambda q: get_oa_chat().invoke(
    PROMPT_GPT.format(schema=SCHEMA_INFO, q=q)
).content.strip()

# ───────────────────────────────────────── Nodes
_REWRITE_PROMPT = (
    "You are a paraphraser. Return a JSON array with exactly 3 diverse rewrites of: '{q}'. No markdown."
)

def rewrite_node(state: State):
    logger = logging.getLogger("rewrite_node")
    logger.info("Received question: %s", state.question)
    raw = get_oa_chat().invoke(_REWRITE_PROMPT.format(q=state.question)).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
    rewrites = json.loads(raw)
    logger.debug("Rewrites → %s", rewrites)
    return {"rewrites": rewrites}

SEP = "<|s|>"

def duckdb_node(state: State):
    logger = logging.getLogger("duckdb_node")
    joined = SEP.join(state.rewrites)
    raw = DUCK_GEN(joined)
    sqls = [s.strip() for s in raw.split(SEP)]
    logger.debug("DuckDB SQLs → %s", sqls)
    return {"duckdb_sql": sqls[: len(state.rewrites)]}

_ENHANCE_PROMPT = textwrap.dedent("""\
Given the DB schema below and the candidate SQL attempts, write a **better** SQLite query that answers the same natural-language question.

SCHEMA:
{schema}

CANDIDATE SQL QUERIES:
{cands}

QUESTION: "{q}"

Return SQL only.
""")

def gpt_node(state: State):
    logger = logging.getLogger("gpt_node")
    direct_sql = GPT_GEN(state.question)

    enhanced_sql = get_oa_chat().invoke(
        _ENHANCE_PROMPT.format(
            schema=SCHEMA_INFO,         
            cands="\n---\n".join(state.duckdb_sql),
            q=state.question,
        )
    ).content.strip()

    logger.debug("GPT direct SQL  → %s", direct_sql)
    logger.debug("GPT enhanced SQL → %s", enhanced_sql)
    return {"gpt_sql": [direct_sql, enhanced_sql]}

_VALIDATOR_PROMPT = textwrap.dedent("""\
You are a SQL validator. Pick the single best SQLite query.

Guidelines
- Use only table / column names in the schema.
- **All text comparisons must be case-correct as per DB Schema** 
- Match literals exactly ('NY', not 'New York') as per DB Schema.
- Prefer the most specific valid query; if all are invalid, choose the simplest valid SELECT.

Return JSON: {{ "sql": "<chosen query>" }}

SCHEMA:
{schema}

QUESTION:
{q}

CANDIDATE QUERIES:
{c}

Respond with:
{{ "sql": "<your chosen query>" }}
""")

def validator_node(state: State):
    logger = logging.getLogger("validator_node")
    cands = state.duckdb_sql + state.gpt_sql
    resp = get_oa_chat(temperature=0).invoke(
        _VALIDATOR_PROMPT.format(schema=SCHEMA_INFO, q=state.question, c="\n---\n".join(cands))
    ).content.strip()
    logger.debug("Validator raw → %s", resp)

    if resp.startswith("```"):
        resp = "\n".join(line for line in resp.splitlines() if not line.startswith("```"))

    try:
        parsed = json.loads(resp)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if "sql" in k.lower():
                    logger.info("Chosen SQL → %s", v.strip())
                    return {"final_sql": v.strip()}
    except json.JSONDecodeError:
        pass

    m = re.search(r'["\']?sql["\']?\s*:\s*["\'](.+?)["\']', resp, re.DOTALL | re.IGNORECASE)
    if m:
        logger.info("Chosen SQL (regex) → %s", m.group(1).strip())
        return {"final_sql": m.group(1).strip()}

    logger.warning("Falling back to first candidate")
    return {"final_sql": cands[0].strip()}

_REPAIR_PROMPT = textwrap.dedent("""\
The SQL below failed when executed.

ERROR:
{err}

Remember: **all text values in the DB are lowercase.**  
Use exact lowercase literals or `LOWER(column)` during comparison.

SCHEMA:
{schema}

QUESTION:
{q}

ORIGINAL SQL:
{bad_sql}

Please return a corrected SQLite query that resolves the error.
Respond with SQL only (no markdown, no explanation).
""")

def exec_node(state: State):
    """
    Execute the validator-chosen SQL.
    • If it raises a sqlite3.Error  OR returns zero rows,
      ask the LLM to repair the query and try once more.
    • Return dict with final_sql (may be repaired) and result (rows or error msg).
    """
    logger = logging.getLogger("exec_node")
    conn = bootstrap_db()

    def run(sql: str) -> tuple[bool, Any]:
        """Return (ok, rows_or_err). ok=True only if query succeeds AND rows > 0."""
        try:
            rows = conn.execute(sql).fetchall()
            return (len(rows) > 0), rows
        except sqlite3.Error as e:
            return False, str(e)

    # ── first attempt
    logger.info("Executing SQL → %s", state.final_sql)
    ok, result = run(state.final_sql)
    if ok:
        logger.info("Rows returned → %s", result)
        conn.close()
        return {"final_sql": state.final_sql, "result": result}

    # ── query failed or returned empty → ask LLM to repair
    err_msg = result if isinstance(result, str) else "Query returned zero rows"
    logger.warning("Execution issue → %s", err_msg)

    fix_sql = get_oa_chat().invoke(
        _REPAIR_PROMPT.format(
            err=err_msg,
            schema=SCHEMA_INFO,
            q=state.question,
            bad_sql=state.final_sql,
        )
    ).content.strip()
    logger.info("Retrying with repaired SQL → %s", fix_sql)

    # ── second attempt
    ok2, result2 = run(fix_sql)
    conn.close()

    if ok2:
        logger.info("Rows returned (after fix) → %s", result2)
        return {"final_sql": fix_sql, "result": result2}

    final_err = result2 if isinstance(result2, str) else "Still no rows after fix"
    logger.error("Second execution issue → %s", final_err)
    return {"final_sql": fix_sql, "result": f"Exec error (after fix): {final_err}"}


# ───────────────────────────────────────── Build graph
from langgraph.graph import StateGraph, END

g = StateGraph(State)
for name, fn in [
    ("rewrite", rewrite_node),
    ("duck", duckdb_node),
    ("gpt", gpt_node),
    ("val", validator_node),
    ("exec", exec_node),
]:
    g.add_node(name, fn)

g.set_entry_point("rewrite")
for a, b in [
    ("rewrite", "duck"),
    ("duck", "gpt"),
    ("gpt", "val"),
    ("val", "exec"),
]:
    g.add_edge(a, b)

g.add_edge("exec", END)
APP = g.compile()

# ───────────────────────────────────────── CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL→SQL multi-agent demo (with logging)")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    log.info("Invoking graph for question: %s", args.question)
    out = APP.invoke({"question": args.question})
    from rich import print as rprint
    rprint("[bold]SQL →[/]", out["final_sql"])
