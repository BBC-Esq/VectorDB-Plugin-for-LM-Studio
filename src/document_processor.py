# compatible with langchain 0.3+

import os
import logging
import warnings
import yaml
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    BSHTMLLoader
)

from constants import DOCUMENT_LOADERS
from extract_metadata import extract_document_metadata, add_pymupdf_page_metadata

logging.basicConfig(
    level=logging.ERROR,  # Only log errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processor.log', mode='w')  # 'w' mode overwrites the file
    ]
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
INGEST_THREADS = max(4, os.cpu_count() - 4)

for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]

def load_single_document(file_path: Path) -> Document:
    file_extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS.get(file_extension)
    if not loader_class:
        print(f"\033[91mFailed---> {file_path.name} (extension: {file_extension})\033[0m")
        logging.error(f"Unsupported file type: {file_path.name} (extension: {file_extension})")
        return None
    
    loader_options = {}
    
    if file_extension in [".epub", ".rtf", ".odt", ".md"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension in [".eml", ".msg"]:
        loader_options.update({
            "mode": "single",
            "process_attachments": False,
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension == ".html":
        loader_options.update({
            "open_encoding": "utf-8",
            "bs_kwargs": {
                "features": "lxml",  # Specify the parser to use (lxml is generally fast and lenient)
                # "parse_only": SoupStrainer("body"),  # Optionally parse only the body tag
                # "from_encoding": "iso-8859-1",  # Specify a different input encoding if needed
            },
            "get_text_separator": "\n",  # Use newline as separator when extracting text
            # Additional parameters and comments:
            # "file_path": "path/to/file.html",  # Usually set automatically by the loader
            # "open_encoding": None,  # Set to None to let BeautifulSoup detect encoding
            # "get_text_separator": " ",  # Use space instead of newline if preferred
        })
    elif file_extension in [".xlsx", ".xls", ".xlsm"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension in [".csv", ".txt"]:
        loader_options.update({
            "encoding": "utf-8",
            "autodetect_encoding": True
        })
    
    try:
        if file_extension in [".epub", ".rtf", ".odt", ".md", ".eml", ".msg", ".xlsx", ".xls", ".xlsm"]:
            unstructured_kwargs = loader_options.pop("unstructured_kwargs", {})
            loader = loader_class(str(file_path), mode=loader_options.get("mode", "single"), **unstructured_kwargs)
        else:
            loader = loader_class(str(file_path), **loader_options)
            
        documents = loader.load()
        
        if not documents:
            print(f"\033[91mFailed---> {file_path.name} (No content extracted)\033[0m")
            logging.error(f"No content could be extracted from file: {file_path.name}")
            return None
            
        document = documents[0]
        metadata = extract_document_metadata(file_path)
        document.metadata.update(metadata)
        
        print(f"Loaded---> {file_path.name}")
        return document
        
    except (OSError, UnicodeDecodeError) as e:
        print(f"\033[91mFailed---> {file_path.name} (Access/encoding error)\033[0m")
        logging.error(f"File access/encoding error - File: {file_path.name} - Error: {str(e)}")
        return None
    except Exception as e:
        print(f"\033[91mFailed---> {file_path.name} (Unexpected error)\033[0m")
        logging.error(f"Unexpected error processing file: {file_path.name} - Error: {str(e)}")
        return None

def load_document_batch(filepaths, threads_per_process):
    with ThreadPoolExecutor(threads_per_process) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
    return (data_list, filepaths)

def load_documents(source_dir: Path) -> list:
    all_files = list(source_dir.iterdir())
    doc_paths = [f for f in all_files if f.suffix.lower() in (key.lower() for key in DOCUMENT_LOADERS.keys())]
    
    docs = []

    if doc_paths:
        n_workers = min(INGEST_THREADS, max(len(doc_paths), 1))
        
        total_cores = os.cpu_count()
        threads_per_process = 1
        
        with ProcessPoolExecutor(n_workers) as executor:
            chunksize = math.ceil(len(doc_paths) / n_workers)
            futures = []
            for i in range(0, len(doc_paths), chunksize):
                chunk_paths = doc_paths[i:i + chunksize]
                futures.append(executor.submit(load_document_batch, chunk_paths, threads_per_process))
            
            for future in as_completed(futures):
                contents, _ = future.result()
                docs.extend(contents)
    
    return docs

def split_documents(documents=None, text_documents_pdf=None):
    try:
        print("\nSplitting documents into chunks.")
        with open("config.yaml", "r", encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
            chunk_size = config["database"]["chunk_size"]
            chunk_overlap = config["database"]["chunk_overlap"]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        texts = []

        # debugging
        # print(f"Documents list before splitting: {[doc.metadata.get('file_type') for doc in documents]}")
        # print(f"PDF Documents list before splitting: {[doc.metadata.get('file_type') for doc in text_documents_pdf]}")

        # split non-PDF document objects
        if documents:
            for i, doc in enumerate(documents):
                if not isinstance(doc.page_content, str):
                    logging.warning(f"Document {i} content is not a string. Converting to string.")
                    documents[i].page_content = str(doc.page_content)

            texts = text_splitter.split_documents(documents)

        # split PDF document objects
        
        """
        PDF files are split using the custom pymupdfparser, which adds custom page markers in the following format:
        
        [[page1]]This is the text content of the first page.
        It might contain multiple lines, paragraphs, or sections.

        [[page2]]This is the text content of the second page.
        Again, it could be as long as necessary, depending on the content.

        [[page3]]Finally, this is the text content of the third page.
        """
        
        if text_documents_pdf:
            processed_pdf_docs = []
            for doc in text_documents_pdf:
                chunked_docs = add_pymupdf_page_metadata(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                processed_pdf_docs.extend(chunked_docs)
            texts.extend(processed_pdf_docs)

        return texts

    except Exception as e:
            logging.exception("Error during document splitting")
            logging.error(f"Error type: {type(e)}")
            raise