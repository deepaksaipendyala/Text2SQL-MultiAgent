from huggingface_hub import hf_hub_download

model_name = "QuantFactory/DuckDB-NSQL-7B-v0.1-GGUF"
model_file = "DuckDB-NSQL-7B-v0.1-q8_0.gguf"

model_path = hf_hub_download(
    repo_id=model_name,
    filename=model_file,
    local_dir="./models"  # use your local path
)

# print("Downloaded to:", model_path)

# from llama_cpp import Llama
# llm = Llama(model_path="./models/DuckDB-NSQL-7B-v0.1-q8_0.gguf",
#             n_gpu_layers=0,      # or few
#             n_ctx=1024,
#             n_threads=8)         # match physical cores
# print(llm("SELECT 1;", max_tokens=1))

# import gc, torch    # torch optional, if you used it elsewhere
# gc.collect()        # force Python to free objects