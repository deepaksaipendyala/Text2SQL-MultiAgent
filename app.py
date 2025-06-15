import streamlit as st
from dotenv import load_dotenv
import sqlite3
import os
from text2sql import APP

load_dotenv()

st.set_page_config(page_title="Text-to-SQL Demo", layout="wide")
st.title("Text-to-SQL HR Assistant")

question = st.text_input(
    "Ask a question about candidates:",
    placeholder="e.g. List candidates with more than 3 years of experience in NY"
)

if st.button("Generate SQL and Run Query") and question:
    with st.spinner("Processing..."):
        state = {"question": question}
        result_state = APP.invoke(state)

        # Display generated SQL
        st.subheader("Final SQL")
        if "final_sql" in result_state and result_state["final_sql"]:
            st.code(result_state["final_sql"], language="sql")
        else:
            st.warning("No SQL was generated.")

        # Display results
        st.subheader("Result Table")
        if isinstance(result_state.get("result"), list) and result_state["result"]:
            headers = ["id", "name", "email", "role", "experience_years", "skills", "location", "availability"]
            st.dataframe([dict(zip(headers, row)) for row in result_state["result"]])
        else:
            st.error("No results found or query failed.")
