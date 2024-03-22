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
import time
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(pathname)s:%(lineno)s - %(funcName)s'
)
logging.getLogger('chromadb.db.duckdb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

class CreateVectorDB:
    def __init__(self, database_name):
        self.ROOT_DIRECTORY = Path(__file__).resolve().parent
        self.SOURCE_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB"
        self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name
        self.CHROMA_SETTINGS = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.PERSIST_DIRECTORY),
            anonymized_telemetry=False
        )

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    
    def initialize_vector_model(self, embedding_model_name, config_data):
        compute_device = config_data['Compute_Device']['database_creation']
        model_kwargs = {"device": compute_device}
        encode_kwargs = {'normalize_embeddings': True, 'show_progress_bar': True, 'batch_size': 8}

        if compute_device.lower() == 'cpu':
            encode_kwargs['batch_size'] = 2
        else:
            batch_size_mapping = {
                'instructor-xl': 1,
                'instructor-large': 3,
                ('jina-embedding-l', 'bge-large', 'gte-large', 'roberta-large'): 4,
                'jina-embedding-s': 9,
                ('bge-small', 'gte-small'): 10,
                ('MiniLM'): 20,
            }

            for key, value in batch_size_mapping.items():
                if embedding_model_name in key if isinstance(key, tuple) else key in embedding_model_name:
                    encode_kwargs['batch_size'] = value
                    break

        if "instructor" in embedding_model_name:
            embed_instruction = config_data['embedding-models']['instructor'].get('embed_instruction')
            query_instruction = config_data['embedding-models']['instructor'].get('query_instruction')
            model = HuggingFaceInstructEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                embed_instruction=embed_instruction,
                query_instruction=query_instruction,
                encode_kwargs=encode_kwargs
            )
        elif "bge" in embedding_model_name:
            query_instruction = config_data['embedding-models']['bge'].get('query_instruction')
            model = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                query_instruction=query_instruction,
                encode_kwargs=encode_kwargs
            )
        else:
            model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

        return model, encode_kwargs

    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        
        # load non-image/non-audio files
        documents = load_documents(self.SOURCE_DIRECTORY)        
        
        # load transcripts
        audio_documents = load_audio_documents(self.SOURCE_DIRECTORY)
        documents.extend(audio_documents)
        if len(audio_documents) > 0:
            print(f"Loaded {len(audio_documents)} audio transcription(s)...")
        
        # load images
        image_documents = specify_image_loader()
        documents.extend(image_documents)

        # split documents
        texts = split_documents(documents)

        # create vectors
        embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)
        my_cprint(f"Vector model loaded into memory with batch size {encode_kwargs['batch_size']}...", "green")

        # create database
        my_cprint("Computing vectors and creating database...\n\n NOTE:\n\nThe progress bar only relates to computing vectors.  After this, the vectors are put into the vector database and there is no visual feedback while this occurs.  This could take a significant amount of time.  Unless you see an error message, rest assured it is still working.\n", "yellow")
        
        start_time = time.time()
        
        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            
        db = Chroma.from_documents(
            texts, embeddings,
            persist_directory=str(self.PERSIST_DIRECTORY),
            client_settings=self.CHROMA_SETTINGS,
        )
        
        db.persist()
        print("Database created.")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Creation of vectors and inserting into the database took {elapsed_time:.2f} seconds.")
        my_cprint("Database saved.", "cyan")

        del embeddings.client
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Embedding model removed from memory.", "red")
