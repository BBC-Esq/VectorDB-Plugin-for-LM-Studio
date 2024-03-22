import os
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from chromadb.config import Settings
import yaml
import torch
from transformers import AutoTokenizer
import gc
from pathlib import Path
from PySide6.QtWidgets import QMessageBox
import sys
from utilities import my_cprint
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


ROOT_DIRECTORY = Path(__file__).resolve().parent
PERSIST_DIRECTORY = ROOT_DIRECTORY / "Vector_DB"

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

def write_contexts_to_file_and_open(contexts, metadata_list):
    contexts_output_file_path = Path('contexts.txt')
    with contexts_output_file_path.open('w', encoding='utf-8') as file:
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = metadata.get('file_name')
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
    
    # obtains settings to interact with server
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        server_config = config.get('server', {})
        base_url = server_config.get('connection_str')
        prefix = server_config.get('prefix', '')
        suffix = server_config.get('suffix', '')
        prompt_format_disabled = server_config.get('prompt_format_disabled')
        model_temperature = server_config.get('model_temperature')
        model_max_tokens = server_config.get('model_max_tokens')

    client = OpenAI(base_url=base_url, api_key='not-needed')

    if prompt_format_disabled:
        formatted_prompt = prompt
    else:
        # user my program's select prefix and suffix if "disabled" checkbox is not checked
        formatted_prompt = f"{prefix}{prompt}{suffix}"

    # "chat completions" api from openai
    stream = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=model_temperature,
        max_tokens=model_max_tokens,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def initialize_vector_model(EMBEDDING_MODEL_NAME, config_data, compute_device):
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 1}
    
    if "instructor" in EMBEDDING_MODEL_NAME:
        embed_instruction = config_data['instructor'].get('embed_instruction')
        query_instruction = config_data['instructor'].get('query_instruction')
        return HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": compute_device}, embed_instruction=embed_instruction, query_instruction=query_instruction, encode_kwargs=encode_kwargs)
        
    elif "bge" in EMBEDDING_MODEL_NAME:
        query_instruction = config_data['bge'].get('query_instruction')
        return HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": compute_device}, query_instruction=query_instruction, encode_kwargs=encode_kwargs)
        
    else:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": compute_device}, encode_kwargs=encode_kwargs)

def initialize_database(embeddings, persist_directory, client_settings, score_threshold, k, search_filter):
    # initialize chromadb db
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=client_settings)
    
    # initialize retriever object from langchain
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': score_threshold, 'k': k, 'filter': search_filter})
    return db, retriever

def retrieve_and_filter_contexts(query, config, embeddings, db):
    
    # define variables based on the information in the config object passed to this function
    search_term = config['database'].get('search_term', '').lower()
    document_types = config['database'].get('document_types', '')
    score_threshold = float(config['database']['similarity'])
    k = int(config['database']['contexts'])
    search_filter = {'document_type': document_types} if document_types else {}
    
    # user chromadb via langchain to instantiate the "as_retriever" search object
    # initializing retriever object
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': score_threshold, 'k': k, 'filter': search_filter})
    
    # actually search the database using "query" as a parameter
    relevant_contexts = retriever.get_relevant_documents(query)
    # apply filter to contexts received, thus defining the "filtered_contexts" variable to only include contexts that satisfy the filter
    filtered_contexts = [doc for doc in relevant_contexts if search_term in doc.page_content.lower()]
    # print message if no contexts left after filtering
    if not filtered_contexts:
        my_cprint("No relevant contexts/chunks found.  Make sure the following settings are not too restrictive: (1) Similarity, (2) Contexts, and (3) Search Filter", "yellow")
    #
    return filtered_contexts

def load_configuration():
    with open('config.yaml', 'r') as config_file:
        print("Configuration loaded.")
        return yaml.safe_load(config_file)

def initialize_system_components(query, config):
    database_to_search = config['database']['database_to_search']
    root_directory = Path(__file__).resolve().parent
    persist_directory = root_directory / "Vector_DB" / database_to_search
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = initialize_vector_model(config['EMBEDDING_MODEL_NAME'], config.get('embedding-models', {}), config['Compute_Device']['database_query'])
    my_cprint("Vector model loaded.", "green")
    
    db = Chroma(persist_directory=str(persist_directory), embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    
    return db, embeddings, root_directory

def prepare_query_and_contexts(db, query, config, embeddings):    
    # call the retrieve_and_filter_contexts function
    filtered_contexts = retrieve_and_filter_contexts(query, config, embeddings, db)
    
    # define the contexts variable as a list of each separate context obtained (after the filter of course)
    contexts = [document.page_content for document in filtered_contexts]
    # define metadata_list as a list of metadata for each respective context
    metadata_list = [document.metadata for document in filtered_contexts]
    
    print(f"Found {len(filtered_contexts)} contexts.")
    return contexts, metadata_list, filtered_contexts

def perform_chatgpt_interaction(augmented_query):
    # define "response_json" and invoke "connect_to_local_chatgpt" function
    # which provides a response streamed in chunks
    response_json = connect_to_local_chatgpt(augmented_query)
    return response_json
    # goes back to ask_local_chatgpt function

def handle_response_and_cleanup(full_response, metadata_list, embeddings, root_directory):
    citations = format_metadata_as_citations(metadata_list)
    unique_citations = set(citations.split("\n"))
    
    del embeddings.client
    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    print("Embedding model removed from memory.")
    
    return "\n".join(unique_citations)

def ask_local_chatgpt(query, chunks_only): # entry point
    config = load_configuration()
    
    db, embeddings, root_directory = initialize_system_components(query, config)
    
    # defines contexts (after being subjected to a possible filter), metadata_list, and filtered_contexts
    contexts, metadata_list, filtered_contexts = prepare_query_and_contexts(db, query, config, embeddings)
    
    # calls the function to create metadata.txt
    save_metadata_to_file(metadata_list, metadata_output_file_path)
    
    # if "chunks only" is checked, creates a .txt file with the filtered chunks and automatically opens it instead of connecting to server
    if chunks_only:
        write_contexts_to_file_and_open(contexts, metadata_list)
        yield "Contexts written to temporary file and opened."

        return
    
    # only runs if "chunks only" is not checked
    prepend_string = "Only base your answer on the provided context/contexts. If you cannot, please state so."
    # formats the structure of what is sent to the server
    augmented_query = f"{prepend_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
    
    # GENERATES A RESPONSE from the server unless/until "stop_streaming" is "true"
    full_response = ""  # accumulate response
    response_generator = perform_chatgpt_interaction(augmented_query)
    for response_chunk in response_generator:
        # sends each "response_chunk" to the GUI
        yield response_chunk
        full_response += response_chunk  # Accumulate response chunks
    
    # save accuulated response to .txt
    with open('chat_history.txt', 'w', encoding='utf-8') as f:
        f.write(full_response)
    
    yield "\n\n"
    
    # SEND CITATIONS TO GUI one by one with a newline for each citation
    citations = handle_response_and_cleanup([], metadata_list, embeddings, root_directory)
    for citation in citations.split("\n"):
        yield citation