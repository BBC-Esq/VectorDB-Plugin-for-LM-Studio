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

from constants import DOCUMENT_LOADERS
from loader_vision_llava import llava_process_images
from loader_vision_cogvlm import cogvlm_process_images
from loader_salesforce import salesforce_process_images

ENABLE_PRINT = True
ROOT_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
INGEST_THREADS = os.cpu_count() or 8

def my_cprint(*args, **kwargs):
    if ENABLE_PRINT:
        filename = "document_processor.py"
        modified_message = f"{filename}: {args[0]}"
        cprint(modified_message, *args[1:], **kwargs)

for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]

from langchain.document_loaders import (
    UnstructuredEPubLoader, UnstructuredRTFLoader, 
    UnstructuredODTLoader, UnstructuredMarkdownLoader, 
    UnstructuredExcelLoader, UnstructuredCSVLoader
)

def process_images_wrapper(config):
    chosen_model = config["vision"]["chosen_model"]

    if chosen_model == 'llava' or chosen_model == 'bakllava':
        return llava_process_images()
    elif chosen_model == 'cogvlm':
        return cogvlm_process_images()
    elif chosen_model == 'salesforce':
        return salesforce_process_images()
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
            loader = Docx2txtLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".rtf":
            loader = UnstructuredRTFLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".odt":
            loader = UnstructuredODTLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".xlsx" or file_extension == ".xlsd":
            loader = UnstructuredExcelLoader(str(file_path), mode="single")
        elif file_extension == ".html" or file_extension == ".htm":
            loader = UnstructuredHTMLLoader(str(file_path), mode="single", strategy="fast")
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(str(file_path), mode="single")
        else:
            loader = loader_class(str(file_path))
    else:
        raise ValueError(f"Document type for extension {file_extension} is undefined")

    document = loader.load()[0]

    # with open("output_load_single_document.txt", "w", encoding="utf-8") as output_file:
        # output_file.write(document.page_content)

    # text extracted before metadata added
    return document


def load_document_batch(filepaths):
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
    return (data_list, filepaths) # "data_list" = list of all document objects created by load single document

def load_documents(source_dir: Path) -> list[Document]:
    all_files = list(source_dir.iterdir())
    paths = [f for f in all_files if f.suffix in DOCUMENT_LOADERS.keys()]
    
    docs = []

    if paths:
        n_workers = min(INGEST_THREADS, max(len(paths), 1))
        my_cprint(f"Number of workers assigned: {n_workers}", "white")
        chunksize = round(len(paths) / n_workers)
        
        if chunksize == 0:
            raise ValueError(f"chunksize must be a non-zero integer, but got {chunksize}. len(paths): {len(paths)}, n_workers: {n_workers}")

        with ProcessPoolExecutor(n_workers) as executor:
            futures = [executor.submit(load_document_batch, paths[i : (i + chunksize)]) for i in range(0, len(paths), chunksize)]
            for future in as_completed(futures):
                contents, _ = future.result()
                docs.extend(contents)
                my_cprint(f"Number of NON-IMAGE files loaded: {len(docs)}", "yellow")

    additional_docs = []
    
    my_cprint(f"Loading images, if any.", "yellow")

    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

        # Use ProcessPoolExecutor to run the selected image processing function in a separate process
        with ProcessPoolExecutor(1) as executor:
            future = executor.submit(process_images_wrapper, config)
            processed_docs = future.result()
            additional_docs = processed_docs if processed_docs is not None else []

    docs.extend(additional_docs)

    return docs

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

'''
# document object structure: Document(page_content="[ALL TEXT EXTRACTED]", metadata={'source': '[FULL FILE PATH WITH DOUBLE BACKSLASHES'})
# list structure: [Document(page_content="...", metadata={'source': '...'}), Document(page_content="...", metadata={'source': '...'})]
'''