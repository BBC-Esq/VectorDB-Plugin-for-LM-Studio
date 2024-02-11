import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from chromadb.config import Settings
import yaml
import torch
from transformers import AutoTokenizer
import gc
import tempfile
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QMessageBox
import sys
from utilities import my_cprint

ROOT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
PERSIST_DIRECTORY = ROOT_DIRECTORY / "Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=str(PERSIST_DIRECTORY), anonymized_telemetry=False
)

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

stop_streaming = False

def save_metadata_to_file(metadata_list, output_file_path):
    with output_file_path.open('w', encoding='utf-8') as output_file:
        for metadata in metadata_list:
            output_file.write(str(metadata) + '\n')

def format_metadata_as_citations(metadata_list):
    citations = [Path(metadata['file_path']).name for metadata in metadata_list]
    return "\n".join(citations)

def write_contexts_to_file_and_open(contexts, metadata_list):
    contexts_output_file_path = Path('contexts.txt')

    with contexts_output_file_path.open('w', encoding='utf-8') as file:
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = Path(metadata['file_path']).name
            file.write(f"---------- Context {index} | From File: {file_name} ----------\n\n")
            file.write(context + "\n\n")
    
    if os.name == 'nt':
        os.startfile(contexts_output_file_path)
    elif sys.platform == 'darwin':
        os.system(f'open {contexts_output_file_path}')
    elif sys.platform.startswith('linux'):
        os.system(f'xdg-open {contexts_output_file_path}')
    else:
        raise NotImplementedError("Unsupported operating system")

def connect_to_local_chatgpt(prompt):
    global stop_streaming

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
        if stop_streaming:
            yield None
            break
        if 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            chunk_message = chunk['choices'][0]['delta']['content']
            yield chunk_message

def initialize_vector_model(EMBEDDING_MODEL_NAME, config_data, compute_device):
    if "instructor" in EMBEDDING_MODEL_NAME:
        embed_instruction = config_data['instructor'].get('embed_instruction')
        query_instruction = config_data['instructor'].get('query_instruction')

        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": compute_device},
            embed_instruction=embed_instruction,
            query_instruction=query_instruction
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        query_instruction = config_data['bge'].get('query_instruction')

        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": compute_device},
            query_instruction=query_instruction
        )

    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": compute_device}
        )
        
def initialize_database(embeddings, persist_directory, client_settings, score_threshold, k, search_filter):
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    retriever = db.as_retriever(search_kwargs={
        'score_threshold': score_threshold, 
        'k': k,
        'filter': search_filter
    })

    return db, retriever

def retrieve_and_filter_contexts(query, config, embeddings, db):
    search_term = config['database'].get('search_term', '').lower()
    document_types = config['database'].get('document_types', '')
    score_threshold = float(config['database']['similarity'])
    k = int(config['database']['contexts'])

    if document_types:
        search_filter = {'document_type': document_types}
    else:
        search_filter = {}

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'score_threshold': score_threshold, 
            'k': k,
            'filter': search_filter
        }
    )

    my_cprint("Querying database.", "white")
    relevant_contexts = retriever.get_relevant_documents(query)

    filtered_contexts = [doc for doc in relevant_contexts if search_term in doc.page_content.lower()]

    if not filtered_contexts:
        my_cprint("No relevant contexts/chunks found.  Make sure the following settings are not too restrictive: (1) Similarity, (2) Contexts, and (3) Search Filter", "yellow")

    return filtered_contexts

def ask_local_chatgpt(query, chunks_only, persist_directory=str(PERSIST_DIRECTORY), client_settings=CHROMA_SETTINGS):
    global stop_streaming
    stop_streaming = False
    my_cprint("Attempting to connect to server.", "white")

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    EMBEDDING_MODEL_NAME = config['EMBEDDING_MODEL_NAME']
    compute_device = config['Compute_Device']['database_query']
    config_data = config.get('embedding-models', {})

    embeddings = initialize_vector_model(EMBEDDING_MODEL_NAME, config_data, compute_device)
    my_cprint("Embedding model loaded.", "green")

    tokenizer_path = "./Tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Initialize Database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=client_settings,
    )
    my_cprint("Database initialized.", "white")

    # Retrieve and filter contexts
    filtered_contexts = retrieve_and_filter_contexts(query, config, embeddings, db)

    contexts = [document.page_content for document in filtered_contexts]
    metadata_list = [document.metadata for document in filtered_contexts]

    # Save metadata to file
    save_metadata_to_file(metadata_list, metadata_output_file_path)

    if chunks_only:
        write_contexts_to_file_and_open(contexts, metadata_list)
        return {"answer": "Contexts written to temporary file and opened", "sources": filtered_contexts}

    # Prepare the augmented query
    prepend_string = "Only base your answer to the following question on the provided context/contexts accompanying this question. If you cannot answer based on the included context/contexts alone, please state so."
    augmented_query = prepend_string + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

    my_cprint(f"Number of relevant contexts: {len(filtered_contexts)}", "white")

    total_tokens = sum(len(tokenizer.encode(context)) for context in contexts)
    my_cprint(f"Total number of tokens in contexts: {total_tokens}", "white")

    # Connect to local ChatGPT and process response
    response_json = connect_to_local_chatgpt(augmented_query)

    full_response = []
    for chunk_message in response_json:
        if chunk_message is None:
            break
        if full_response and isinstance(full_response[-1], str):
            full_response[-1] += chunk_message
        else:
            full_response.append(chunk_message)

        yield chunk_message

    # Write chat history
    chat_history_file_path = ROOT_DIRECTORY / 'chat_history.txt'
    with chat_history_file_path.open('w', encoding='utf-8') as file:
        for message in full_response:
            file.write(message)

    yield "\n\n"

    # Generate and yield citations
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

    return {"answer": response_json, "sources": filtered_contexts}

def stop_interaction():
    global stop_streaming
    stop_streaming = True

'''
# Another example using my specific document metadata extracted:

retriever = db.as_retriever(search_kwargs={
    'score_threshold': score_threshold, 
    'k': k,
    'filter': {
        'file_path': '/path/to/specific/directory/',
        'file_type': '.pdf',
        'file_name': 'example_document',
        'file_size': {'$gt': 1024},  # Example: file size greater than 1024 bytes
        'creation_date': {'$gt': '2022-01-01'},  # Example: created after January 1, 2022
        'modification_date': {'$gt': '2022-01-01'},  # Example: modified after January 1, 2022
        'image': 'False'
    }
})

# Use a filter to only retrieve documents from a specific paper
docsearch.as_retriever(
    search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
'''