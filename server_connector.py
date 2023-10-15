import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from chromadb.config import Settings
import yaml
import torch

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

def connect_to_local_chatgpt(prompt):
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        server_config = config.get('server', {})
        openai_api_base = server_config.get('connection_str')
        openai_api_key = server_config.get('api_key')
        prefix = server_config.get('prefix')
        suffix = server_config.get('suffix')
        model_temperature = server_config.get('model_temperature')
        model_max_tokens = server_config.get('model_max_tokens')
    
    openai.api_base = openai_api_base
    openai.api_key = openai_api_key

    formatted_prompt = f"{prefix}{prompt}{suffix}"
    response = openai.ChatCompletion.create(
        model="local model",
        temperature=model_temperature,
        max_tokens=model_max_tokens,
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    return response.choices[0].message["content"]

def ask_local_chatgpt(query, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS):
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        EMBEDDING_MODEL_NAME = config['EMBEDDING_MODEL_NAME']
        COMPUTE_DEVICE = config.get("COMPUTE_DEVICE", "cpu")

    if "instructor" in EMBEDDING_MODEL_NAME:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
            embed_instruction="Represent the document for retrievel:",
            query_instruction="Represent the question for retrieving supporting documents:"
        )
    elif "bge" in EMBEDDING_MODEL_NAME and "large-en-v1.5" not in EMBEDDING_MODEL_NAME:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
            query_instruction="Represent this sentence for searching relevant passages:"
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': COMPUTE_DEVICE},
        )

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=client_settings,
    )
    retriever = db.as_retriever()
    relevant_contexts = retriever.get_relevant_documents(query)
    contexts = [document.page_content for document in relevant_contexts]
    prepend_string = "Only base your answer to the following question on the provided context."
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
    response_json = connect_to_local_chatgpt(augmented_query)
        
    return {"answer": response_json, "sources": relevant_contexts}

def interact_with_chat(user_input):
    global last_response
    response = ask_local_chatgpt(user_input)
    answer = response['answer']
    last_response = answer
    return answer