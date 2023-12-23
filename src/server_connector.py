import logging
import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from chromadb.config import Settings
import yaml
import torch
from transformers import AutoTokenizer
from termcolor import cprint
import gc
import tempfile
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QMessageBox

ENABLE_PRINT = True
ENABLE_CUDA_PRINT = False

def my_cprint(*args, **kwargs):
    if ENABLE_PRINT:
        filename = "server_connector.py"
        modified_message = f"{filename}: {args[0]}"
        cprint(modified_message, *args[1:], **kwargs)

def print_cuda_memory():
    if ENABLE_CUDA_PRINT:
        max_allocated_memory = torch.cuda.max_memory_allocated()
        memory_allocated = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()

        my_cprint(f"Max CUDA memory allocated: {max_allocated_memory / (1024**2):.2f} MB", "green")
        my_cprint(f"Total CUDA memory allocated: {memory_allocated / (1024**2):.2f} MB", "white")
        my_cprint(f"Total CUDA memory reserved: {reserved_memory / (1024**2):.2f} MB", "white")

print_cuda_memory()

ROOT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
PERSIST_DIRECTORY = ROOT_DIRECTORY / "Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=str(PERSIST_DIRECTORY), anonymized_telemetry=False
)

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

def save_metadata_to_file(metadata_list, output_file_path):
    with output_file_path.open('w', encoding='utf-8') as output_file:
        for metadata in metadata_list:
            output_file.write(str(metadata) + '\n')

def format_metadata_as_citations(metadata_list):
    citations = [Path(metadata['file_path']).name for metadata in metadata_list]
    return "\n".join(citations)

def write_contexts_to_file_and_open(contexts):
    with contexts_output_file_path.open('w', encoding='utf-8') as file:
        for index, context in enumerate(contexts, start=1):
            file.write(f"------------ Context {index} ---------------\n\n\n")
            file.write(context + "\n\n\n")
    
    if os.name == 'nt':
        os.startfile(contexts_output_file_path)
    elif os.name == 'posix':
        subprocess.run(['open', contexts_output_file_path])

def connect_to_local_chatgpt(prompt):
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        server_config = config.get('server', {})
        openai_api_base = server_config.get('connection_str')
        openai_api_key = server_config.get('api_key')
        prefix = server_config.get('prefix')
        suffix = server_config.get('suffix')
        prompt_format_disabled = server_config.get('prompt_format_disabled', False)
        model_temperature = server_config.get('model_temperature')
        model_max_tokens = server_config.get('model_max_tokens')

    openai.api_base = openai_api_base
    openai.api_key = openai_api_key

    if prompt_format_disabled:
        formatted_prompt = prompt
    else:
        formatted_prompt = f"{prefix}{prompt}{suffix}"

    response = openai.ChatCompletion.create(
        model="local model",
        temperature=model_temperature,
        max_tokens=model_max_tokens,
        messages=[{"role": "user", "content": formatted_prompt}], stream=True
    )
    for chunk in response:
        if 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            chunk_message = chunk['choices'][0]['delta']['content']
            yield chunk_message

def ask_local_chatgpt(query, persist_directory=str(PERSIST_DIRECTORY), client_settings=CHROMA_SETTINGS):
    my_cprint("Attempting to connect to server.", "white")
    print_cuda_memory()

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        test_embeddings = config.get('test_embeddings', False)

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        try:
            EMBEDDING_MODEL_NAME = config['EMBEDDING_MODEL_NAME']
        except KeyError:
            msg_box = QMessageBox()
            msg_box.setText("Must download and choose an embedding model to use first!")
            msg_box.exec()
            raise
        compute_device = config['Compute_Device']['database_query']
        config_data = config.get('embedding-models', {})
        score_threshold = float(config['database']['similarity'])
        k = int(config['database']['contexts'])

    model_kwargs = {"device": compute_device}

    my_cprint("Embedding model loaded.", "green")
    if "instructor" in EMBEDDING_MODEL_NAME:
        embed_instruction = config_data['instructor'].get('embed_instruction')
        query_instruction = config_data['instructor'].get('query_instruction')

        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            embed_instruction=embed_instruction,
            query_instruction=query_instruction
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        query_instruction = config_data['bge'].get('query_instruction')

        embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            query_instruction=query_instruction
        )

    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs
        )

    tokenizer_path = "./Tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    my_cprint("Database initialized.", "white")

    retriever = db.as_retriever(search_kwargs={'score_threshold': score_threshold, 'k': k})

    my_cprint("Querying database.", "white")
    try:
        relevant_contexts = retriever.get_relevant_documents(query)
        logging.info("Retrieved %d relevant contexts", len(relevant_contexts))
    except Exception as e:
        logging.error("Error querying database: %s", str(e))
        raise

    if not relevant_contexts:
        logging.warning("No relevant contexts found for the query")

    contexts = [document.page_content for document in relevant_contexts]
    metadata_list = [document.metadata for document in relevant_contexts]

    save_metadata_to_file(metadata_list, metadata_output_file_path)

    if test_embeddings:
        write_contexts_to_file_and_open(contexts)
        return {"answer": "Contexts written to temporary file and opened", "sources": relevant_contexts}

    prepend_string = "Only base your answer to the following question on the provided context/contexts accompanying this question.  If you cannot answer based on the included context/contexts alone, please state so."
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

    my_cprint(f"Number of relevant contexts: {len(relevant_contexts)}", "white")

    total_tokens = sum(len(tokenizer.encode(context)) for context in contexts)
    my_cprint(f"Total number of tokens in contexts: {total_tokens}", "white")

    response_json = connect_to_local_chatgpt(augmented_query)

    full_response = []

    for chunk_message in response_json:
        if full_response and isinstance(full_response[-1], str):
            full_response[-1] += chunk_message
        else:
            full_response.append(chunk_message)

        yield chunk_message

    chat_history_file_path = ROOT_DIRECTORY / 'chat_history.txt'
    with chat_history_file_path.open('w', encoding='utf-8') as file:
        for message in full_response:
            file.write(message)

    yield "\n\n"
    
    # LLM's response complete
    # format and append citations
    citations = format_metadata_as_citations(metadata_list)
    
    unique_citations = []
    for citation in citations.split("\n"):
        if citation not in unique_citations:
            unique_citations.append(citation)
    
    yield "\n".join(unique_citations)

    del embeddings.client
    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    my_cprint("Embedding model removed from memory.", "red")

    return {"answer": response_json, "sources": relevant_contexts}

if __name__ == "__main__":
    user_input = "Your query here"
    ask_local_chatgpt(user_input)
