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
from pathlib import Path
import os
from utilities import backup_database, my_cprint
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(pathname)s:%(lineno)s - %(funcName)s'
)
logging.getLogger('chromadb.db.duckdb').setLevel(logging.WARNING)

class CreateVectorDB:
    def __init__(self):
        self.ROOT_DIRECTORY = Path(__file__).resolve().parent
        self.SOURCE_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB"
        self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB"
        self.INGEST_THREADS = os.cpu_count() or 8

        self.CHROMA_SETTINGS = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.PERSIST_DIRECTORY),
            anonymized_telemetry=False
        )

    def run(self):
        with open(self.ROOT_DIRECTORY / "config.yaml", 'r') as stream:
            config_data = yaml.safe_load(stream)

        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")

        my_cprint(f"Loading documents.", "white")
        documents = load_documents(self.SOURCE_DIRECTORY)  # invoke document_processor.py; returns a list of document objects
        if documents is None or len(documents) == 0:
            my_cprint("No documents to load.", "red")
            return
        my_cprint(f"Successfully loaded documents.", "white")

        texts = split_documents(documents) # invoke document_processor.py again; returns a list of split document objects

        embeddings = self.get_embeddings(EMBEDDING_MODEL_NAME, config_data)
        my_cprint("Embedding model loaded.", "green")

        if self.PERSIST_DIRECTORY.exists():
            shutil.rmtree(self.PERSIST_DIRECTORY)
        self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        my_cprint("Creating database.", "white")

        db = Chroma.from_documents(
            texts, embeddings,
            persist_directory=str(self.PERSIST_DIRECTORY),
            client_settings=self.CHROMA_SETTINGS,
        )

        my_cprint("Persisting database.", "white")
        db.persist()
        my_cprint("Database persisted.", "white")

        backup_database()

        del embeddings.client
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Embedding model removed from memory.", "red")

    def get_embeddings(self, EMBEDDING_MODEL_NAME, config_data):
        my_cprint("Creating embeddings.", "white")

        compute_device = config_data['Compute_Device']['database_creation']

        if "instructor" in EMBEDDING_MODEL_NAME:
            embed_instruction = config_data['embedding-models']['instructor'].get('embed_instruction')
            query_instruction = config_data['embedding-models']['instructor'].get('query_instruction')

            return HuggingFaceInstructEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": compute_device},
                embed_instruction=embed_instruction,
                query_instruction=query_instruction # cache_folder=, encode_kwargs=
            )

        elif "bge" in EMBEDDING_MODEL_NAME:
            query_instruction = config_data['embedding-models']['bge'].get('query_instruction')

            return HuggingFaceBgeEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": compute_device},
                query_instruction=query_instruction # encode_kwargs=, cache_folder=
            )

        else:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": compute_device} # encode_kwargs=, cache_folder=, multi_process=
            )

if __name__ == "__main__":
    create_vector_db = CreateVectorDB()
    create_vector_db.run()
