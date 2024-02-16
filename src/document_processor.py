import os
import yaml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    EverNoteLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)

import pickle
from constants import DOCUMENT_LOADERS
from loader_images import loader_cogvlm, loader_llava, loader_salesforce
from extract_metadata import extract_document_metadata
from utilities import my_cprint

ROOT_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
INGEST_THREADS = os.cpu_count() or 8

for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]

def choose_image_loader(config):
    chosen_model = config["vision"]["chosen_model"]

    if chosen_model == 'llava' or chosen_model == 'bakllava':
        image_loader = loader_llava()
        return image_loader.llava_process_images()
    elif chosen_model == 'cogvlm':
        image_loader = loader_cogvlm()
        return image_loader.cogvlm_process_images()
    elif chosen_model == 'salesforce':
        image_loader = loader_salesforce()
        return image_loader.salesforce_process_images()
    else:
        return []

def load_single_document(file_path: Path) -> Document:
    file_extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS.get(file_extension)

    if loader_class:
        if file_extension == ".txt":
            loader = loader_class(str(file_path), encoding='utf-8', autodetect_encoding=True)
        elif file_extension == ".epub":
            loader = UnstructuredEPubLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif file_extension == ".rtf":
            loader = UnstructuredRTFLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".odt":
            loader = UnstructuredODTLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".xlsx" or file_extension == ".xlsd":
            loader = UnstructuredExcelLoader(str(file_path), mode="single")
        elif file_extension == ".html":
            loader = UnstructuredHTMLLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(str(file_path), mode="single")
        else:
            loader = loader_class(str(file_path))
    else:
        raise ValueError(f"Document type for extension {file_extension} is undefined")

    document = loader.load()[0]

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
    pkl_paths = [f for f in all_files if f.suffix.lower() == '.pkl']
    
    docs = []

    if doc_paths:
        n_workers = min(INGEST_THREADS, max(len(doc_paths), 1))
        my_cprint(f"Number of workers assigned: {n_workers}", "white")
        chunksize = round(len(doc_paths) / n_workers)
        
        if chunksize > 0:
            with ProcessPoolExecutor(n_workers) as executor:
                futures = [executor.submit(load_document_batch, doc_paths[i : i + chunksize]) for i in range(0, len(doc_paths), chunksize)]
                for future in as_completed(futures):
                    contents, _ = future.result()
                    docs.extend(contents)
            my_cprint(f"Number of document files loaded: {len(docs)}", "yellow")
        else:
            my_cprint("Chunk size calculation error, but proceeding with other file types.", "red")

    for pkl_path in pkl_paths:
        my_cprint(f"Loading audio transcriptions, if any.", "green")
        try:
            with open(pkl_path, 'rb') as pkl_file:
                doc = pickle.load(pkl_file)
                docs.append(doc)
            my_cprint(f"Loaded audio transcription document object from {pkl_path}.", "green")
        except Exception as e:
            my_cprint(f"Error loading {pkl_path}: {e}", "red")

    additional_docs = []
    
    my_cprint("Loading images, if any.", "yellow")
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

        with ProcessPoolExecutor(1) as executor:
            future = executor.submit(choose_image_loader, config)
            processed_docs = future.result()
            additional_docs = processed_docs if processed_docs is not None else []

    docs.extend(additional_docs)

    return docs

def split_documents(documents):
    my_cprint("Splitting documents.", "white")
    with open("config.yaml", "r", encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
        chunk_size = config["database"]["chunk_size"]
        chunk_overlap = config["database"]["chunk_overlap"]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    
    my_cprint(f"Number of Chunks: {len(texts)}", "white")
    
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
    
    return texts