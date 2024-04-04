import os
from openai import OpenAI
from langchain_community.vectorstores import TileDB
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
import yaml
import torch
from transformers import AutoTokenizer
import gc
from pathlib import Path
from PySide6.QtWidgets import QMessageBox
import sys
from utilities import my_cprint
from InstructorEmbedding import INSTRUCTOR
from typing import Dict, Optional, Union
from huggingface_hub import snapshot_download

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"
            
def save_metadata_to_file(metadata_list):
    with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
        for metadata in metadata_list:
            output_file.write(str(metadata) + '\n')

def format_metadata_as_citations(metadata_list):
    citations = [Path(metadata['file_path']).name for metadata in metadata_list]
    return "\n".join(citations)

def yield_formatted_contexts(contexts, metadata_list):
    for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
        file_name = metadata.get('file_name', 'Unknown')
        formatted_context = (
            f"---------- Context {index} | From File: {file_name} ----------\n\n"
            f"{context}\n\n"
        )
        yield formatted_context

def connect_to_local_chatgpt(prompt):
    
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
        formatted_prompt = f"{prefix}{prompt}{suffix}"

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

def initialize_vector_model(config):    
    database_to_search = config['database']['database_to_search']
    model_path = config['created_databases'][database_to_search]['model']
    
    compute_device = config['Compute_Device']['database_query']
    encode_kwargs = {'normalize_embeddings': False, 'batch_size': 1}
    
    if "instructor" in model_path:
        
        return HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-base",
            cache_folder=model_path,
            model_kwargs={"device": compute_device},
            encode_kwargs=encode_kwargs,
        )
        
    elif "bge" in model_path:
        query_instruction = config['embedding-models']['bge']['query_instruction']
        
        return HuggingFaceBgeEmbeddings(
            model_name=model_path,
            model_kwargs={"device": compute_device},
            encode_kwargs=encode_kwargs
        )
        
    elif "nomic" in model_path:        
        model = HuggingFaceBgeEmbeddings(
            model_name=model_path,
            model_kwargs={"device": compute_device, "trust_remote_code": True},
            encode_kwargs=encode_kwargs,
            query_instruction = "search_query: Answer this question.",
        )
        
    else:
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": compute_device},
            encode_kwargs=encode_kwargs
        )

def initialize_database(config, embeddings):
    database_to_search = config['database']['database_to_search']
    persist_directory = ROOT_DIRECTORY / "Vector_DB" / database_to_search
    
    db = TileDB.load(index_uri=str(persist_directory), embedding=embeddings, allow_dangerous_deserialization=True)
    
    return db

def initialize_retriever(config, db):
    document_types = config['database'].get('document_types', '')
    search_filter = {'document_type': document_types} if document_types else {}
    score_threshold = float(config['database']['similarity'])
    k = int(config['database']['contexts'])
    search_type = "similarity"
    
    retriever = db.as_retriever(
    search_type=search_type,
    search_kwargs={
        'score_threshold': score_threshold,
        'k': k,
        'filter': search_filter
    }
)
    
    return retriever

def retrieve_and_filter_contexts(config, retriever, query):
    relevant_contexts = retriever.get_relevant_documents(query=query)
    
    search_term = config['database'].get('search_term', '').lower()
    filtered_contexts = [doc for doc in relevant_contexts if search_term in doc.page_content.lower()]
    
    if filtered_contexts:
        print(f"Returning {len(filtered_contexts)} filtered contexts.")
    else:
        my_cprint("No relevant contexts/chunks found.  Make sure the following settings are not too restrictive: (1) Similarity, (2) Contexts, and (3) Search Filter", "yellow")

    contexts = [document.page_content for document in filtered_contexts]
    metadata_list = [document.metadata for document in filtered_contexts]
    
    return contexts, metadata_list, filtered_contexts

def load_configuration():
    with open('config.yaml', 'r') as config_file:
        return yaml.safe_load(config_file)

def perform_chatgpt_interaction(augmented_query):
    response_json = connect_to_local_chatgpt(augmented_query)
    return response_json

def handle_response_and_cleanup(full_response, metadata_list, embeddings):
    citations = format_metadata_as_citations(metadata_list)
    unique_citations = set(citations.split("\n"))
    
    del embeddings.client
    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    print("Embedding model removed from memory.")
    
    return "\n".join(unique_citations)

def ask_local_chatgpt(query, chunks_only): # ENTRY POINT
    # load config.yaml
    config = load_configuration()
    
    # get embedding function
    embeddings = initialize_vector_model(config)
    
    # initialize vector db
    db = initialize_database(config, embeddings)
    
    # initialize retriever
    retriever = initialize_retriever(config, db)
    
    # use retriever
    contexts, metadata_list, filtered_contexts = retrieve_and_filter_contexts(config, retriever, query)
    
    # save metadata from filtered contexts
    save_metadata_to_file(metadata_list)
    
    # display chunks only
    if chunks_only:
        for formatted_context in yield_formatted_contexts(contexts, metadata_list):
            yield formatted_context
        return
    
    # augment query
    prepend_string = "Only base your answer on the provided context/contexts. If you cannot, please state so."
    augmented_query = f"{prepend_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
    
    # connect to local LLM
    full_response = ""
    response_generator = perform_chatgpt_interaction(augmented_query)
    for response_chunk in response_generator:
        yield response_chunk
        full_response += response_chunk
    
    # save chat_history.txt for TTS
    with open('chat_history.txt', 'w', encoding='utf-8') as f:
        f.write(full_response)
    
    yield "\n\n"
    
    # format and display citations
    citations = handle_response_and_cleanup(full_response, metadata_list, embeddings)
    for citation in citations.split("\n\n"):
        yield citation
