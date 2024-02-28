import shutil
import yaml
import gc
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from document_processor import load_documents, split_documents
from loader_audio import load_audio_documents
from loader_images import specify_image_loader
import torch
from utilities import validate_symbolic_links, my_cprint
from pathlib import Path
import os
import logging
from PySide6.QtCore import QDir

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(pathname)s:%(lineno)s - %(funcName)s'
)
logging.getLogger('chromadb.db.duckdb').setLevel(logging.WARNING)

class CreateVectorDB:
    def __init__(self, database_name):
        self.ROOT_DIRECTORY = Path(__file__).resolve().parent
        self.SOURCE_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB"
        self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name
        self.INGEST_THREADS = os.cpu_count() or 8
        self.CHROMA_SETTINGS = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.PERSIST_DIRECTORY),
            anonymized_telemetry=False
        )

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)

    def create_embeddings(self, embedding_model_name, config_data):
        my_cprint("Creating embeddings.", "white")
        compute_device = config_data['Compute_Device']['database_creation']

        if "instructor" in embedding_model_name:
            embed_instruction = config_data['embedding-models']['instructor'].get('embed_instruction')
            query_instruction = config_data['embedding-models']['instructor'].get('query_instruction')

            return HuggingFaceInstructEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": compute_device},
                embed_instruction=embed_instruction,
                query_instruction=query_instruction,
                encode_kwargs={'normalize_embeddings': True}
            )

        elif "bge" in embedding_model_name:
            query_instruction = config_data['embedding-models']['bge'].get('query_instruction')

            return HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": compute_device},
                query_instruction=query_instruction,
                encode_kwargs={'normalize_embeddings': True}
            )

        else:
            return HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": compute_device},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        
        # load non-image/non-audio documents
        documents = load_documents(self.SOURCE_DIRECTORY)
        my_cprint(f"Loaded {len(documents)} document file(s).", "green")
        
        # load audio documents
        audio_documents = load_audio_documents(self.SOURCE_DIRECTORY)
        documents.extend(audio_documents)
        my_cprint(f"Loaded {len(audio_documents)} audio file(s).", "green")
        
        # load image documents
        image_documents = specify_image_loader()
        documents.extend(image_documents)
        my_cprint(f"Loaded {len(image_documents)} image file(s).", "green")

        texts = split_documents(documents) # split document objects

        embeddings = self.create_embeddings(EMBEDDING_MODEL_NAME, config_data)
        my_cprint("Embedding model loaded.", "green")

        my_cprint("Creating database.", "white")

        db = Chroma.from_documents(
            texts, embeddings,
            persist_directory=str(self.PERSIST_DIRECTORY),
            client_settings=self.CHROMA_SETTINGS,
        )

        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        
        db.persist()
        my_cprint("Database created.", "white")

        del embeddings.client
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Embedding model removed from memory.", "red")
