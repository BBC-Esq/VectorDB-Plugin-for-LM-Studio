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
# from memory_profiler import profile

ENABLE_PRINT = True
ENABLE_CUDA_PRINT = False

torch.cuda.reset_peak_memory_stats()

def my_cprint(*args, **kwargs):
    if ENABLE_PRINT:
        modified_message = f"create_database.py: {args[0]}"
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
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

# @profile
def main():
    print_cuda_memory()

    with open(os.path.join(ROOT_DIRECTORY, "config.yaml"), 'r') as stream:
        config_data = yaml.safe_load(stream)

    EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")

    my_cprint(f"Loading documents.", "cyan")
    documents = load_documents(SOURCE_DIRECTORY)
    my_cprint(f"Successfully loaded documents.", "cyan")
    
    texts = split_documents(documents)
    print_cuda_memory()
    
    embeddings = get_embeddings(EMBEDDING_MODEL_NAME, config_data)
    my_cprint("Embedding model loaded.", "green")
    print_cuda_memory()

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY)

    my_cprint("Creating database.", "cyan")
    
    db = Chroma.from_documents(
        texts, embeddings, 
        persist_directory=PERSIST_DIRECTORY, 
        client_settings=CHROMA_SETTINGS,
    )
    print_cuda_memory()
    
    my_cprint("Persisting database.", "cyan")
    db.persist()
    my_cprint("Database persisted.", "cyan")
    print_cuda_memory()
    
    del embeddings.client
    print_cuda_memory()
    
    del embeddings
    print_cuda_memory()
    
    torch.cuda.empty_cache()
    print_cuda_memory()
    
    gc.collect()
    my_cprint("Embedding model removed from memory.", "red")
    print_cuda_memory()
    

# @profile
def get_embeddings(EMBEDDING_MODEL_NAME, config_data, normalize_embeddings=False):
    my_cprint("Creating embeddings.", "cyan")
    print_cuda_memory()
    
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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO
    )
    main()
