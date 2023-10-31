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
from termcolor import cprint

from termcolor import cprint

ENABLE_PRINT = True

def my_cprint(*args, **kwargs):
    if ENABLE_PRINT:
        filename = "create_database.py"
        modified_message = f"{filename}: {args[0]}"
        cprint(modified_message, *args[1:], **kwargs)

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

my_cprint("Initializing Chroma settings", "cyan")
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

def validate_config(config_data):
    my_cprint("Validating config", "cyan")
    required_keys = ["EMBEDDING_MODEL_NAME", "COMPUTE_DEVICE"]
    missing_keys = []

    for key in required_keys:
        if key not in config_data:
            missing_keys.append(key)

    if missing_keys:
        raise KeyError(f"Missing required keys in config file: {', '.join(missing_keys)}")

def main():
    my_cprint("Main function started", "cyan")

    with open(os.path.join(ROOT_DIRECTORY, "config.yaml"), 'r') as stream:
        config_data = yaml.safe_load(stream)

    validate_config(config_data)

    EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
    COMPUTE_DEVICE = config_data.get("COMPUTE_DEVICE")

    my_cprint(f"Loading documents from {SOURCE_DIRECTORY}", "cyan")
    documents = load_documents(SOURCE_DIRECTORY)
    texts = split_documents(documents)

    validate_symbolic_links(SOURCE_DIRECTORY)
    my_cprint(f"Split into {len(texts)} chunks of text", "cyan")

    my_cprint("Generating embeddings", "cyan")
    embeddings = get_embeddings(EMBEDDING_MODEL_NAME, COMPUTE_DEVICE, config_data)

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY)

    my_cprint("Creating Chroma database", "cyan")
    db = Chroma.from_documents(
        texts, embeddings, 
        persist_directory=PERSIST_DIRECTORY, 
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()

    my_cprint("Vector database has been created.", "cyan")

    embeddings = None
    del embeddings

    my_cprint("Cleaning up", "cyan")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    
    my_cprint("Done cleaning up", "cyan")

def get_embeddings(EMBEDDING_MODEL_NAME, COMPUTE_DEVICE, config_data, normalize_embeddings=False):
    my_cprint("Getting embeddings", "cyan")
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
