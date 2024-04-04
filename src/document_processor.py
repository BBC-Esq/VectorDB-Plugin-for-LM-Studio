import os
import yaml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from pathlib import Path
from langchain_community.docstore.document import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    EverNoteLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)

from constants import DOCUMENT_LOADERS
from loader_images import loader_cogvlm, loader_llava#, loader_salesforce
from extract_metadata import extract_document_metadata
from utilities import my_cprint
import logging

logging.getLogger('unstructured').setLevel(logging.ERROR)

ROOT_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
INGEST_THREADS = max(4, os.cpu_count() - 4)

for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]

def load_single_document(file_path: Path) -> Document:
    print(f"Loading document: {file_path.name}")
    file_extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS.get(file_extension)

    if not loader_class:
        raise ValueError(f"Document type for extension {file_extension} is undefined")

    loader_options = {
        "encoding": "utf-8",
        "autodetect_encoding": True
    }

    if file_extension in [".epub", ".rtf", ".odt", ".md", ".html"]:
        loader_options.update({"mode": "single", "strategy": "fast"})
    elif file_extension in [".xlsx", ".xlsd"]:
        loader_options.update({"mode": "single"})
    elif file_extension in [".docx", ".csv", ".txt"]:
        pass
    else:
        loader_options = {}

    loader = loader_class(str(file_path), **loader_options)

    document = loader.load()[0]

    # Extract and update metadata
    metadata = extract_document_metadata(file_path)
    document.metadata.update(metadata)
    
    return document

def load_document_batch(filepaths):
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
    return (data_list, filepaths)

def load_documents(source_dir: Path) -> list:
    all_files = list(source_dir.iterdir())
    doc_paths = [f for f in all_files if f.suffix.lower() in (key.lower() for key in DOCUMENT_LOADERS.keys())]
    
    docs = []

    if doc_paths:
        n_workers = min(INGEST_THREADS, max(len(doc_paths), 1))
        print(f"Using {n_workers} CPU threads to load {len(doc_paths)} non-image/non-audio files...")
        chunksize = round(len(doc_paths) / n_workers)
        
        if chunksize > 0:
            with ProcessPoolExecutor(n_workers) as executor:
                futures = [executor.submit(load_document_batch, doc_paths[i : i + chunksize]) for i in range(0, len(doc_paths), chunksize)]
                for future in as_completed(futures):
                    contents, _ = future.result()
                    docs.extend(contents)
        else:
            my_cprint("Chunk size calculation error, but proceeding with other file types.", "red")

    return docs

def split_documents(documents):
    with open("config.yaml", "r", encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
        chunk_size = config["database"]["chunk_size"]
        chunk_overlap = config["database"]["chunk_overlap"]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    
    print(f"Number of chunks created: {len(texts)}")
    
    ''' Uncomment to check chunk lengths.
    chunk_sizes = [len(text.page_content) for text in texts]
    min_size = min(chunk_sizes)
    average_size = sum(chunk_sizes) / len(texts)
    max_size = max(chunk_sizes)
    
    size_ranges = range(1, max_size + 1, 100)
    for size_range in size_ranges:
        lower_bound = size_range
        upper_bound = size_range + 99
        count = sum(lower_bound <= size <= upper_bound for size in chunk_sizes)
        my_cprint(f"Chunks between {lower_bound} and {upper_bound} characters: {count}", "white")
    '''

    return texts