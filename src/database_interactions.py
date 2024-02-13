import shutil
import yaml
import gc
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from document_processor import load_documents, split_documents
import torch
from utilities import validate_symbolic_links, backup_database, my_cprint
from pathlib import Path
import os
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
                query_instruction=query_instruction # cache_folder=, encode_kwargs=
            )

        elif "bge" in embedding_model_name:
            query_instruction = config_data['embedding-models']['bge'].get('query_instruction')

            return HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": compute_device},
                query_instruction=query_instruction # encode_kwargs=, cache_folder=
            )

        else:
            return HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": compute_device} # encode_kwargs=, cache_folder=, multi_process=
            )
    
    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")

        my_cprint("Loading documents.", "white")
        documents = load_documents(self.SOURCE_DIRECTORY) # returns a list of full-text document objects
        if documents is None or len(documents) == 0:
            my_cprint("No documents to load.", "red")
            return
        my_cprint("Successfully loaded documents.", "white")

        texts = split_documents(documents) # returns a list of chunked document objects

        embeddings = self.create_embeddings(EMBEDDING_MODEL_NAME, config_data)
        my_cprint("Embedding model loaded.", "green")

        if self.PERSIST_DIRECTORY.exists():
            shutil.rmtree(self.PERSIST_DIRECTORY)
        self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        my_cprint("Creating database.", "white")

        db = Chroma.from_documents(
            texts, embeddings,
            persist_directory=str(self.PERSIST_DIRECTORY),
            client_settings=self.CHROMA_SETTINGS,
            # collection_name="Test123",
            # collection_metadata (Optional[Dict]),
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

# To delete entries based on the "hash" metadata attribute, you can use this as_retriever method to create a retriever that filters documents based on their metadata. Once you retrieve the documents with the specific hash, you can then extract their IDs and use the delete method to remove them from the vectorstore.

# class CreateVectorDB:
    # # ... [other methods] ...

    # def delete_entries_by_hash(self, target_hash):
        # my_cprint(f"Deleting entries with hash: {target_hash}", "red")

        # # Initialize the retriever with a filter for the specific hash
        # retriever = self.db.as_retriever(search_kwargs={'filter': {'hash': target_hash}})

        # # Retrieve documents with the specific hash
        # documents = retriever.search("")

        # # Extract IDs from the documents
        # ids_to_delete = [doc.id for doc in documents]

        # # Delete entries with the extracted IDs
        # if ids_to_delete:
            # self.db.delete(ids=ids_to_delete)
            # my_cprint(f"Deleted {len(ids_to_delete)} entries from the database.", "green")
        # else:
            # my_cprint("No entries found with the specified hash.", "yellow")

