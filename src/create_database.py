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
from pathlib import Path
import os

ENABLE_PRINT = True

def my_cprint(*args, **kwargs):
    if ENABLE_PRINT:
        modified_message = f"create_database.py: {args[0]}"
        cprint(modified_message, *args[1:], **kwargs)

ROOT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
PERSIST_DIRECTORY = ROOT_DIRECTORY / "Vector_DB"
INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=str(PERSIST_DIRECTORY),
    anonymized_telemetry=False
)

def main():

    with open(ROOT_DIRECTORY / "config.yaml", 'r') as stream:
        config_data = yaml.safe_load(stream)

    EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")

    my_cprint(f"Loading documents.", "white")
    documents = load_documents(SOURCE_DIRECTORY) # First invocation of document_processor.py script
    my_cprint(f"Successfully loaded documents.", "white")
    
    texts = split_documents(documents) # Second invocation of document_processor.py script
    my_cprint(f"Successfully split documents.", "white")
    
    embeddings = get_embeddings(EMBEDDING_MODEL_NAME, config_data)
    my_cprint("Embedding model loaded.", "green")

    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    my_cprint("Creating database.", "white")
    
    db = Chroma.from_documents(
        texts, embeddings, 
        persist_directory=str(PERSIST_DIRECTORY), 
        client_settings=CHROMA_SETTINGS,
    )
    
    my_cprint("Persisting database.", "white")
    db.persist()
    my_cprint("Database persisted.", "white")
    
    del embeddings.client
    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    my_cprint("Embedding model removed from memory.", "red")

def get_embeddings(EMBEDDING_MODEL_NAME, config_data, normalize_embeddings=False):
    my_cprint("Creating embeddings.", "white")
    
    compute_device = config_data['Compute_Device']['database_creation']
    
    if "instructor" in EMBEDDING_MODEL_NAME:
        embed_instruction = config_data['embedding-models']['instructor'].get('embed_instruction')
        query_instruction = config_data['embedding-models']['instructor'].get('query_instruction')

        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": compute_device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
            embed_instruction=embed_instruction,
            query_instruction=query_instruction
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        query_instruction = config_data['embedding-models']['bge'].get('query_instruction')

        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": compute_device},
            query_instruction=query_instruction,
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )
    
    else:
        
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": compute_device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )

if __name__ == "__main__":
    main()
