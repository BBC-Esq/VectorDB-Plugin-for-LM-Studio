import shutil
import yaml
import gc
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import TileDB
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

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    
    def initialize_vector_model(self, embedding_model_name, config_data):
        compute_device = config_data['Compute_Device']['database_creation']
        model_kwargs = {"device": compute_device}
        encode_kwargs = {'normalize_embeddings': False, 'batch_size': 8}

        if compute_device.lower() == 'cpu':
            encode_kwargs['batch_size'] = 2
        else:
            batch_size_mapping = {
                'instructor-xl': 2,
                'instructor-large': 3,
                ('jina-embedding-l', 'bge-large', 'gte-large', 'roberta-large'): 4,
                'jina-embedding-s': 9,
                ('bge-small', 'gte-small'): 10,
                ('MiniLM',): 20,
            }

            for key, value in batch_size_mapping.items():
                if embedding_model_name in key if isinstance(key, tuple) else key in embedding_model_name:
                    encode_kwargs['batch_size'] = value
                    break

        if "instructor" in embedding_model_name:
            embed_instruction = config_data['embedding-models']['instructor'].get('embed_instruction')
            query_instruction = config_data['embedding-models']['instructor'].get('query_instruction')
            encode_kwargs['show_progress_bar'] = True
            
            model = HuggingFaceInstructEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                embed_instruction=embed_instruction,
                query_instruction=query_instruction,
                encode_kwargs=encode_kwargs
            )
        elif "bge" in embedding_model_name:
            query_instruction = config_data['embedding-models']['bge'].get('query_instruction')
            encode_kwargs['show_progress_bar'] = True
            
            model = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                query_instruction=query_instruction,
                encode_kwargs=encode_kwargs
            )
        else:
            model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                show_progress=True,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

        return model, encode_kwargs

    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        
        # load non-image/non-audio files into list
        documents = load_documents(self.SOURCE_DIRECTORY)        
        
        # load transcripts and add to list
        audio_documents = load_audio_documents(self.SOURCE_DIRECTORY)
        documents.extend(audio_documents)
        if len(audio_documents) > 0:
            print(f"Loaded {len(audio_documents)} audio transcription(s)...")
        
        # load images and add to list
        image_documents = specify_image_loader()
        documents.extend(image_documents)

        # split all documents in list
        texts = split_documents(documents)

        # create vectors
        embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)
        my_cprint(f"Vector model loaded into memory with batch size {encode_kwargs['batch_size']}...", "green")

        # create database
        my_cprint("Computing vectors and creating database...\n\n NOTE:\n\nThe progress bar only relates to computing vectors.  After this, the vectors are put into the vector database and there is no visual feedback while this occurs.  This could take a significant amount of time.  Unless you see an error message, rest assured it is still working.\n", "yellow")
        
        start_time = time.time()
        
        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            
        db = TileDB.from_documents(
            documents=texts,
            embedding=embeddings,
            index_uri=str(self.PERSIST_DIRECTORY),
            allow_dangerous_deserialization=True,
            metric="euclidean",
            index_type="FLAT",
        )
        
        #db.persist()
        print("Database created.")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Creation of vectors and inserting into the database took {elapsed_time:.2f} seconds.")
        my_cprint("Database saved.", "cyan")

        #del db
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