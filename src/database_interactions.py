import shutil
import yaml
import gc
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import TileDB
from document_processor import load_documents, split_documents
from loader_images import specify_image_loader
import torch
from utilities import validate_symbolic_links, my_cprint
from pathlib import Path
import os
import logging
from PySide6.QtCore import QDir
import time
import pickle
from InstructorEmbedding import INSTRUCTOR
from typing import Dict, Optional, Union
from huggingface_hub import snapshot_download

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
        self.SAVE_JSON_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB" / database_name

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    
    def initialize_vector_model(self, embedding_model_name, config_data):
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        compute_device = config_data['Compute_Device']['database_creation']
        model_kwargs = {"device": compute_device}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 8}

        if compute_device.lower() == 'cpu':
            encode_kwargs['batch_size'] = 2
        else:
            batch_size_mapping = {
                'sentence-t5-xxl': 1,
                ('instructor-xl', 'sentence-t5-xl'): 2,
                'instructor-large': 3,
                ('jina-embedding-l', 'bge-large', 'gte-large', 'roberta-large'): 4,
                'jina-embedding-s': 9,
                ('bge-small', 'gte-small'): 10,
                ('MiniLM',): 20,
            }

            for key, value in batch_size_mapping.items():
                if isinstance(key, tuple):
                    if any(model_name_part in EMBEDDING_MODEL_NAME for model_name_part in key):
                        encode_kwargs['batch_size'] = value
                        break
                else:
                    if key in EMBEDDING_MODEL_NAME:
                        encode_kwargs['batch_size'] = value
                        break
                        
            my_cprint(f"Vector model initialized with a batch size of {encode_kwargs['batch_size']}", "blue")

        if "instructor" in embedding_model_name:
            encode_kwargs['show_progress_bar'] = True

            if "xl" in embedding_model_name:
                model_version = "xl"
            elif "base" in embedding_model_name:
                model_version = "base"
            else:
                model_version = "large"

            model_name = f"hkunlp/instructor-{model_version}"
            
            model = HuggingFaceInstructEmbeddings(
                model_name=model_name,
                cache_folder=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
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
            
        elif "nomic" in embedding_model_name:
            model_kwargs['trust_remote_code'] = True
            encode_kwargs['show_progress_bar'] = True
            
            model = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                embed_instruction = "search_document:"
            )
        else:
            model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                show_progress=True,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

        return model, encode_kwargs

    def create_database(self, texts, embeddings):
        my_cprint("Creating vectors and database...\n\n NOTE:\n\nNOTE: The progress bar only relates to computing vectors, not inserting them into the database.  Rest assured, after it reaches 100% it is still working unless you get an error message.\n", "yellow")

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

        print("Database created.")

        end_time = time.time()
        elapsed_time = end_time - start_time

        my_cprint("Database saved.", "cyan")
        print(f"Creation of vectors and inserting into the database took {elapsed_time:.2f} seconds.")
        
    def save_documents_to_json(self, json_docs_to_save):
        self.SAVE_JSON_DIRECTORY.mkdir(parents=True, exist_ok=True)

        for document in json_docs_to_save:
            document_hash = document.metadata.get('hash', None)
            if document_hash:
                json_filename = f"{document_hash}.json"
                json_file_path = self.SAVE_JSON_DIRECTORY / json_filename
                
                actual_file_path = document.metadata.get('file_path')
                if os.path.islink(actual_file_path):
                    resolved_path = os.path.realpath(actual_file_path)
                    document.metadata['file_path'] = resolved_path

                document_json = document.json(indent=4)
                
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json_file.write(document_json)
            else:
                print("Warning: Document missing 'hash' in metadata. Skipping JSON creation.")
    
    def load_audio_documents(self, source_dir: Path = None) -> list:
        if source_dir is None:
            source_dir = self.SOURCE_DIRECTORY
        json_paths = [f for f in source_dir.iterdir() if f.suffix.lower() == '.json']
        docs = []

        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    json_str = json_file.read()
                    doc = Document.parse_raw(json_str)
                    docs.append(doc)
            except Exception as e:
                my_cprint(f"Error loading {json_path}: {e}", "red")

        return docs
    
    def clear_docs_for_db_folder(self):
        for item in self.SOURCE_DIRECTORY.iterdir():
            if item.is_file() or item.is_symlink():
                try:
                    item.unlink()
                except Exception as e:
                    print(f"Failed to delete {item}: {e}")
    
    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        
        # load non-image/non-audio documents
        documents = load_documents(self.SOURCE_DIRECTORY)
        
        # load image documents
        image_documents = specify_image_loader()
        documents.extend(image_documents)
        
        json_docs_to_save = documents
        
        # load audio documents
        audio_documents = self.load_audio_documents()  # Now calling the method internally
        documents.extend(audio_documents)
        if len(audio_documents) > 0:
            print(f"Loaded {len(audio_documents)} audio transcription(s)...")

        # split each document in the list of documents
        texts = split_documents(documents)

        # initialize vector model
        embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

        # create database
        self.create_database(texts, embeddings)
        
        self.save_documents_to_json(json_docs_to_save)
        
        del embeddings.client
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Embedding model removed from memory.", "red")
        
        # clear ingest folder
        self.clear_docs_for_db_folder()
        print("Cleared all files and symlinks in Docs_for_DB folder.")

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