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
        my_cprint(f"Total CUDA memory allocated: {memory_allocated / (1024**2):.2f} MB", "yellow")
        my_cprint(f"Total CUDA memory reserved: {reserved_memory / (1024**2):.2f} MB", "yellow")

print_cuda_memory()

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

def write_contexts_to_file_and_open(contexts):
    with open('contexts.txt', 'w', encoding='utf-8') as file:
        for index, context in enumerate(contexts, start=1):
            file.write(f"------------ Context {index} ---------------\n\n\n")
            file.write(context + "\n\n\n")
    temp_file_path = 'contexts.txt'

    if os.name == 'nt':
        os.startfile(temp_file_path)
    elif os.name == 'posix':
        subprocess.run(['open', temp_file_path])

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

def ask_local_chatgpt(query, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS):
    my_cprint("Attempting to connect to server.", "yellow")
    print_cuda_memory()

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        test_embeddings = config.get('test_embeddings', False)

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        EMBEDDING_MODEL_NAME = config['EMBEDDING_MODEL_NAME']
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

    # Set score_threshold and k
    my_cprint("Querying database.", "yellow")
    retriever = db.as_retriever(search_kwargs={'score_threshold': score_threshold, 'k': k})

    relevant_contexts = retriever.get_relevant_documents(query)
    contexts = [document.page_content for document in relevant_contexts]

    # Is test_embeddings is True?
    if test_embeddings:
        write_contexts_to_file_and_open(contexts)
        return {"answer": "Contexts written to temporary file and opened", "sources": relevant_contexts}

    prepend_string = "Only base your answer to the following question on the provided context."
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

    my_cprint(f"Number of relevant contexts: {len(relevant_contexts)}", "yellow")

    total_tokens = sum(len(tokenizer.encode(context)) for context in contexts)
    my_cprint(f"Total number of tokens in contexts: {total_tokens}", "yellow")

    response_json = connect_to_local_chatgpt(augmented_query)

    for chunk_message in response_json:
        yield chunk_message
    
    del embeddings.client
    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    my_cprint("Embedding model removed from memory.", "red")
    
    return {"answer": response_json, "sources": relevant_contexts}

def interact_with_chat(user_input):
    my_cprint("interact_with_chat function", "yellow")
    global last_response
    response = ask_local_chatgpt(user_input)
    answer = response['answer']
    last_response = answer
    
    return answer
