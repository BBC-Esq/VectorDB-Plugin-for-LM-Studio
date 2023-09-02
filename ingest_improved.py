import logging
import os
import sys
import shutil
import tkinter as tk
from tkinter import messagebox
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFMinerLoader
from chromadb.config import Settings

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
INGEST_THREADS = os.cpu_count() or 8


def show_error_dialog(message):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showerror("Error", message)
    root.destroy()
    
# Read the embedding model name from the command line argument
if len(sys.argv) > 1:
    EMBEDDING_MODEL_NAME = sys.argv[1]
else:
    error_msg = "You must select a valid folder that holds the embedding model you want to use."
    show_error_dialog(error_msg)
    raise ValueError(error_msg)

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

DOCUMENT_MAP = {
    ".pdf": PDFMinerLoader,
}

def load_single_document(file_path: str) -> Document:
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
        return (data_list, filepaths)

def load_documents(source_dir: str) -> list[Document]:
    all_files = os.listdir(source_dir)
    paths = [os.path.join(source_dir, file_path) for file_path in all_files if os.path.splitext(file_path)[1] in DOCUMENT_MAP.keys()]

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = [executor.submit(load_document_batch, paths[i : (i + chunksize)]) for i in range(0, len(paths), chunksize)]
        for future in as_completed(futures):
            contents, _ = future.result()
            docs.extend(contents)

    return docs

def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    return documents, []  # We're only processing PDFs now, no more split based on extensions

def main():
# Determine the appropriate compute device
    if torch.cuda.is_available() and torch.version.cuda:
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    elif torch.cuda.is_available() and torch.version.hip:
        device_type = "cuda"
    else:
        device_type = "cpu"

    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, _ = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(text_documents)
    
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")
    
    if "instructor" in EMBEDDING_MODEL_NAME:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            query_instruction="Represent the legal treatise for retrieval."
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )
    
    # Delete contents of the PERSIST_DIRECTORY before creating the vector database
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        os.makedirs(PERSIST_DIRECTORY)
    
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()