import os

# Global settings and variables
ENABLE_PRINT = False
GLOBAL_VAR = True

# Directories
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"

# Ingest settings
INGEST_THREADS = os.cpu_count() or 8

# Document mappings
DOCUMENT_MAP = {
    ".pdf": "PDFMinerLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
    ".json": "JSONLoader",
    ".enex": "EverNoteLoader",
    ".eml": "UnstructuredEmailLoader",
    ".msg": "UnstructuredEmailLoader",
    ".csv": "UnstructuredCSVLoader",
    ".xls": "UnstructuredExcelLoader",
    ".xlsx": "UnstructuredExcelLoader",
    ".rtf": "UnstructuredRTFLoader",
}
