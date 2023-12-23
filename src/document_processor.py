import os
import yaml
from termcolor import cprint
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
    EverNoteLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
    UnstructuredMarkdownLoader
)

from constants import DOCUMENT_LOADERS

ENABLE_PRINT = True

def my_cprint(*args, **kwargs):
    if ENABLE_PRINT:
        filename = "document_processor.py"
        modified_message = f"{filename}: {args[0]}"
        cprint(modified_message, *args[1:], **kwargs)

ROOT_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
INGEST_THREADS = os.cpu_count() or 8

for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]

def load_single_document(file_path: Path) -> Document:
    file_extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS.get(file_extension)

    if loader_class:
        if file_extension == ".txt":
            loader = loader_class(str(file_path), encoding='utf-8')
        elif file_extension == ".json":
            jq_schema = ".[]"  # Assuming you still want to keep JSON loading with the previous schema
            loader = loader_class(str(file_path), jq_schema=jq_schema)
        elif file_extension == ".epub":
            loader = UnstructuredEPubLoader(str(file_path), mode="single")
        else:
            loader = loader_class(str(file_path))
    else:
        raise ValueError(f"Document type for extension {file_extension} is undefined")

    document = loader.load()[0]
    '''
    # Write the content to a .txt file
    with open("output_load_single_document.txt", "w", encoding="utf-8") as output_file:
        output_file.write(document.page_content)
    '''
    return document

def load_document_batch(filepaths):
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
    return (data_list, filepaths)

def load_documents(source_dir: Path) -> list[Document]:
    all_files = list(source_dir.iterdir())
    paths = [f for f in all_files if f.suffix in DOCUMENT_LOADERS.keys()]
    
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    my_cprint(f"Number of workers assigned: {n_workers}", "white")
    chunksize = round(len(paths) / n_workers)
    
    if chunksize == 0:
        raise ValueError(f"chunksize must be a non-zero integer, but got {chunksize}. len(paths): {len(paths)}, n_workers: {n_workers}")
    
    docs = []
    
    with ProcessPoolExecutor(n_workers) as executor:
        futures = [executor.submit(load_document_batch, paths[i : (i + chunksize)]) for i in range(0, len(paths), chunksize)]
        for future in as_completed(futures):
            contents, _ = future.result()
            docs.extend(contents)
            my_cprint(f"Number of files loaded: {len(docs)}", "white")
    
    return docs # end of first invocation by create_database.py

def split_documents(documents):
    my_cprint(f"Splitting documents.", "white")
    with open("config.yaml", "r") as config_file:
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
    
    size_ranges = range(1, max_size+1, 100)
    for size_range in size_ranges:
        lower_bound = size_range
        upper_bound = size_range + 99
        count = sum(lower_bound <= size <= upper_bound for size in chunk_sizes)
        my_cprint(f"Chunks between {lower_bound} and {upper_bound} characters: {count}", "white")
    
    return texts
