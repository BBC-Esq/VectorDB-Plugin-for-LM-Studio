import gc
import logging
import warnings
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import yaml
from InstructorEmbedding import INSTRUCTOR
from PySide6.QtCore import QDir
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import TileDB

from document_processor import load_documents, split_documents
from module_process_images import choose_image_loader
from utilities import my_cprint

datasets_logger = logging.getLogger('datasets')
datasets_logger.setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger().setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(filename)s:%(lineno)d - %(message)s'
)
            
class CreateVectorDB:
    def __init__(self, database_name):
        self.ROOT_DIRECTORY = Path(__file__).resolve().parent
        self.SOURCE_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB"
        self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name
        self.SAVE_JSON_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name / "json"

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    
    @torch.inference_mode()
    def initialize_vector_model(self, embedding_model_name, config_data):
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        compute_device = config_data['Compute_Device']['database_creation']
        model_kwargs = {"device": compute_device, "trust_remote_code": True}
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
                ('MiniLM',): 30,
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

        if "instructor" in embedding_model_name:
            encode_kwargs['show_progress_bar'] = True
            print(f"Using embedding model path: {embedding_model_name}")

            model = HuggingFaceInstructEmbeddings(
                model_name=embedding_model_name,
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
        
        else:
            # model_kwargs["trust_remote_code"] = True
            model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                show_progress=True,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            # model.client.to(dtype=torch.float16) # currently this works while adding the parameter within "model_kwargs" does not for reasons I have yet to figure out.  Moreover, it only worked with all-mpnet and sentence-xxl and not instructor-based models.  TO DO!!!

        my_cprint(f"{EMBEDDING_MODEL_NAME.rsplit('/', 1)[-1]} loaded.", "green")
        
        return model, encode_kwargs

    @torch.inference_mode()
    def create_database(self, texts, embeddings):
        logging.info("Entering create_database method")
        logging.info(f"Number of documents to be inserted into the database: {len(texts)}")
        logging.info(f"Type of texts variable: {type(texts)}")

        my_cprint("Creating vectors and database...\n\nNOTE: The progress bar relates to computing vectors.  Afterwards, it takes a little time to insert them into the database and save to disk.\n", "yellow")

        start_time = time.time()

        logging.info(f"Checking if directory {self.PERSIST_DIRECTORY} exists")
        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {self.PERSIST_DIRECTORY}")
        else:
            logging.info(f"Directory already exists: {self.PERSIST_DIRECTORY}")

        try:
            logging.info("Calling TileDB.from_documents()")
            db = TileDB.from_documents(
                documents=texts,
                embedding=embeddings,
                index_uri=str(self.PERSIST_DIRECTORY),
                allow_dangerous_deserialization=True,
                metric="euclidean",
                index_type="FLAT",
            )
        except Exception as e:
            logging.error(f"Error creating TileDB database: {str(e)}")
            raise

        print("Database created.")

        end_time = time.time()
        elapsed_time = end_time - start_time

        print("Database saved to disk.")
        logging.info(f"Creation of vectors and inserting into the database took {elapsed_time:.2f} seconds.")
        
    def save_documents_to_json(self, json_docs_to_save):
        if not self.SAVE_JSON_DIRECTORY.exists():
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

    def save_document_structures(self, documents):
        with open(self.ROOT_DIRECTORY / "document_structures.txt", 'w', encoding='utf-8') as file:
            for doc in documents:
                file.write(str(doc))
                file.write('\n\n')


    def save_documents_to_pickle(self, documents):
        pickle_directory = self.ROOT_DIRECTORY / "pickle"
        
        if not pickle_directory.exists():
            pickle_directory.mkdir(parents=True, exist_ok=True)
        
        for file in pickle_directory.glob("*.pickle"):
            file.unlink()

        time.sleep(3)

        for i, doc in enumerate(documents):
            pickle_file_path = pickle_directory / f"document_{i}.pickle"
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(doc, pickle_file)

    
    @torch.inference_mode()
    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        
        # load non-image/non-audio documents
        print("Processing documents, if any...")
        documents = load_documents(self.SOURCE_DIRECTORY)
        logging.info(f"Type of documents after load_documents: {type(documents)}")
        if isinstance(documents, list) and documents:
            logging.info(f"Type of first element in documents: {type(documents[0])}")
        
        # load image documents
        # time.sleep(3)
        print("Processing images, if any...")
        image_documents = choose_image_loader()
        logging.info(f"Type of image_documents: {type(image_documents)}")
        if isinstance(image_documents, list) and image_documents:
            logging.info(f"Type of first element in image_documents: {type(image_documents[0])}")
        if len(image_documents) > 0:
            documents.extend(image_documents)
        
        json_docs_to_save = documents
        
        # load audio documents
        print("Processing audio transcripts, if any...")
        audio_documents = self.load_audio_documents()
        logging.info(f"Type of audio_documents: {type(audio_documents)}")
        if isinstance(audio_documents, list) and audio_documents:
            logging.info(f"Type of first element in audio_documents: {type(audio_documents[0])}")
        documents.extend(audio_documents)
        if len(audio_documents) > 0:
            print(f"Loaded {len(audio_documents)} audio transcription(s)...")

        # Before splitting
        logging.info(f"Type of documents before split_documents: {type(documents)}")
        if isinstance(documents, list) and documents:
            logging.info(f"Type of first element in documents: {type(documents[0])}")

        # split each document in the list of documents
        print("Splitting documents...")
        texts = split_documents(documents)

        # After splitting
        logging.info(f"Type of texts after split_documents: {type(texts)}")
        if isinstance(texts, list) and texts:
            logging.info(f"Type of first element in texts: {type(texts[0])}")

        self.save_documents_to_pickle(texts)
        self.save_document_structures(texts)

        # initialize vector model
        embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

        # Before creating database
        logging.info(f"Type of texts before create_database: {type(texts)}")
        if isinstance(texts, list) and texts:
            logging.info(f"Type of first element in texts: {type(texts[0])}")

        # create database
        print("Starting vector database creation process...")
        self.create_database(texts, embeddings)
        
        self.save_documents_to_json(json_docs_to_save)
        
        del embeddings.client
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Vector model removed from memory.", "red")
        
        self.clear_docs_for_db_folder()


class QueryVectorDB:
    def __init__(self, selected_database):
        self.config = self.load_configuration()
        self.selected_database = selected_database
        self.embeddings = self.initialize_vector_model()
        self.db = self.initialize_database()
        self.retriever = self.initialize_retriever()

    def load_configuration(self):
        config_file_path = Path(__file__).resolve().parent / "config.yaml"
        with open(config_file_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def initialize_vector_model(self):    
        model_path = self.config['created_databases'][self.selected_database]['model']
        
        compute_device = self.config['Compute_Device']['database_query']
        encode_kwargs = {'normalize_embeddings': False, 'batch_size': 1}
        
        if "instructor" in model_path:
            return HuggingFaceInstructEmbeddings(
                model_name=model_path,
                model_kwargs={"device": compute_device},
                encode_kwargs=encode_kwargs,
            )
        elif "bge" in model_path:
            query_instruction = self.config['embedding-models']['bge']['query_instruction']
            return HuggingFaceBgeEmbeddings(model_name=model_path, model_kwargs={"device": compute_device}, query_instruction=query_instruction, encode_kwargs=encode_kwargs)
        else:
            return HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"device": compute_device, "trust_remote_code": True}, encode_kwargs=encode_kwargs)

    def initialize_database(self):
        persist_directory = Path(__file__).resolve().parent / "Vector_DB" / self.selected_database
        
        return TileDB.load(index_uri=str(persist_directory), embedding=self.embeddings, allow_dangerous_deserialization=True)

    def initialize_retriever(self):
        document_types = self.config['database'].get('document_types', '')
        search_filter = {'document_type': document_types} if document_types else {}
        score_threshold = float(self.config['database']['similarity'])
        k = int(self.config['database']['contexts'])
        search_type = "similarity"
        
        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs={
                'score_threshold': score_threshold,
                'k': k,
                'filter': search_filter
            }
        )

    def search(self, query):
        relevant_contexts = self.retriever.invoke(input=query)
        
        search_term = self.config['database'].get('search_term', '').lower()
        filtered_contexts = [doc for doc in relevant_contexts if search_term in doc.page_content.lower()]
        
        contexts = [document.page_content for document in filtered_contexts]
        metadata_list = [document.metadata for document in filtered_contexts]
        
        return contexts, metadata_list