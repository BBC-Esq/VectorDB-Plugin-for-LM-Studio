import logging
import os
import shutil
import torch
import yaml
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings  # Updated import
from document_loader import load_documents
from document_chunker import split_documents
import gc

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

# Updated the CHROMA_SETTINGS with the settings from the older version
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

def main():
    with open("config.yaml", 'r') as stream:
        config_data = yaml.safe_load(stream)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME", "")
        COMPUTE_DEVICE = config_data.get("COMPUTE_DEVICE", "cpu")

    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    texts = split_documents(documents)
    
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")
    
    if "instructor" in EMBEDDING_MODEL_NAME:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
            query_instruction="Represent the document for retrieval."
        )
    elif "bge" in EMBEDDING_MODEL_NAME and "large-en-v1.5" not in EMBEDDING_MODEL_NAME:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": COMPUTE_DEVICE},
        )
    
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        os.makedirs(PERSIST_DIRECTORY)
    
    # Updated Chroma instantiation with older settings
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()  # Added persistence call

    embeddings = None
    db = None
    gc.collect()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()

