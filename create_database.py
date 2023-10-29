import logging
import os
import shutil
import yaml
import gc
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from document_processor import load_documents, split_documents
import torch
from utilities import validate_symbolic_links

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

def validate_config(config_data):
    required_keys = ["EMBEDDING_MODEL_NAME", "COMPUTE_DEVICE"]
    missing_keys = []

    for key in required_keys:
        if key not in config_data:
            missing_keys.append(key)

    if missing_keys:
        raise KeyError(f"Missing required keys in config file: {', '.join(missing_keys)}")

def main():
    with open(os.path.join(ROOT_DIRECTORY, "config.yaml"), 'r') as stream:
        config_data = yaml.safe_load(stream)
    
    validate_config(config_data)
    
    EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
    COMPUTE_DEVICE = config_data.get("COMPUTE_DEVICE")

    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")

    documents = load_documents(SOURCE_DIRECTORY)
    texts = split_documents(documents)

    validate_symbolic_links(SOURCE_DIRECTORY)

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = get_embeddings(EMBEDDING_MODEL_NAME, COMPUTE_DEVICE, config_data)

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY)

    db = Chroma.from_documents(
        texts, embeddings, 
        persist_directory=PERSIST_DIRECTORY, 
        client_settings=CHROMA_SETTINGS,
    )

    db.persist()

    print("Vector database has been created.")

    embeddings = None
    del embeddings

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

def get_embeddings(EMBEDDING_MODEL_NAME, COMPUTE_DEVICE, config_data, normalize_embeddings=False):
    if "instructor" in EMBEDDING_MODEL_NAME:
        embed_instruction = config_data['embedding-models']['instructor'].get('embed_instruction')
        query_instruction = config_data['embedding-models']['instructor'].get('query_instruction')

        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
            embed_instruction=embed_instruction,
            query_instruction=query_instruction
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        query_instruction = config_data['embedding-models']['bge'].get('query_instruction')

        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
            query_instruction=query_instruction,
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )
    
    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO
    )
    main()
