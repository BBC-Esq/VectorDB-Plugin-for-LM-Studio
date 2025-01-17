# database_interactions.py

import gc
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional
import threading
import re
import sqlite3
import torch
import yaml
import concurrent.futures
import queue
from collections import defaultdict, deque
import shutil
import random

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import TileDB

from document_processor import load_documents, split_documents
from module_process_images import choose_image_loader
from utilities import my_cprint, get_model_native_precision, get_appropriate_dtype, supports_flash_attention
from constants import VECTOR_MODELS

class BaseEmbeddingModel:
    def __init__(self, model_name, model_kwargs, encode_kwargs, is_query=False):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.is_query = is_query

    def prepare_kwargs(self):
        return self.model_kwargs

    def prepare_encode_kwargs(self):
        if self.is_query:
            # Override batch size for queries
            self.encode_kwargs['batch_size'] = 1
        return self.encode_kwargs

    def create(self):
        prepared_kwargs = self.prepare_kwargs()
        prepared_encode_kwargs = self.prepare_encode_kwargs()
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            show_progress=not self.is_query,  # Only show progress for database creation
            model_kwargs=prepared_kwargs,
            encode_kwargs=prepared_encode_kwargs
        )


class InstructorEmbedding(BaseEmbeddingModel):
    def create(self):
        instruction = (
            "Represent the question for retrieving supporting documents:"
            if self.is_query else
            "Represent the document for retrieval:"
        )
        
        model = HuggingFaceInstructEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.prepare_encode_kwargs(),
            show_progress=not self.is_query,
            embed_instruction=instruction
        )
        
        if "torch_dtype" in self.model_kwargs.get("model_kwargs", {}):
            torch_dtype = self.model_kwargs["model_kwargs"]["torch_dtype"]
            if torch_dtype is not None:
                model.client[0].auto_model = model.client[0].auto_model.to(torch_dtype)
        
        return model


class SnowflakeEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        if "large" in self.model_name.lower():
            return self.model_kwargs
        
        snow_kwargs = deepcopy(self.model_kwargs)
        compute_device = self.model_kwargs.get("device", "").lower()
        is_cuda = compute_device == "cuda"
        use_xformers = is_cuda and supports_flash_attention()
        
        snow_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager" if use_xformers else "sdpa"
        }
        
        return snow_kwargs


class StellaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        stella_kwargs = deepcopy(self.model_kwargs)
        stella_kwargs["model_kwargs"].update({
            "trust_remote_code": True
        })
        if "torch_dtype" in self.model_kwargs.get("model_kwargs", {}):
            stella_kwargs["model_kwargs"]["torch_dtype"] = self.model_kwargs["model_kwargs"]["torch_dtype"]
        
        return stella_kwargs

    def prepare_encode_kwargs(self):
        encode_kwargs = super().prepare_encode_kwargs()
        if self.is_query:
            encode_kwargs["prompt_name"] = "s2p_query"
        return encode_kwargs


class AlibabaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        ali_kwargs = deepcopy(self.model_kwargs)
        ali_kwargs["model_kwargs"].update({
            "tokenizer_kwargs": {
                "max_length": 8192,
                "padding": True,
                "truncation": True
            }
        })

        compute_device = self.model_kwargs.get("device", "").lower()
        is_cuda = compute_device == "cuda"
        use_xformers = is_cuda and supports_flash_attention()
        ali_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager" if use_xformers else "sdpa"
        }

        return ali_kwargs


def create_vector_db_in_process(database_name):
    create_vector_db = CreateVectorDB(database_name=database_name)
    create_vector_db.run()

def process_chunks_only_query(database_name, query, result_queue):
    try:
        query_db = QueryVectorDB(database_name)
        contexts, metadata_list = query_db.search(query)
        
        formatted_contexts = []
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = metadata.get('file_name', 'Unknown')
            cleaned_context = re.sub(r'\n[ \t]+\n', '\n\n', context)
            cleaned_context = re.sub(r'\n\s*\n\s*\n*', '\n\n', cleaned_context.strip())
            formatted_context = (
                f"{'-'*80}\n"
                f"CONTEXT {index} | {file_name}\n"
                f"{'-'*80}\n"
                f"{cleaned_context}\n"
            )
            formatted_contexts.append(formatted_context)
            
        result_queue.put("\n".join(formatted_contexts))
    except Exception as e:
        result_queue.put(f"Error querying database: {str(e)}")
    finally:
        if 'query_db' in locals():
            query_db.cleanup()


class CreateVectorDB:
    def __init__(self, database_name):
        self.ROOT_DIRECTORY = Path(__file__).resolve().parent
        self.SOURCE_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB"
        self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)

    @torch.inference_mode()
    def initialize_vector_model(self, embedding_model_name, config_data):
        compute_device = config_data['Compute_Device']['database_creation']
        use_half = config_data.get("database", {}).get("half", False)
        model_native_precision = get_model_native_precision(embedding_model_name, VECTOR_MODELS)
        torch_dtype = get_appropriate_dtype(compute_device, use_half, model_native_precision)

        model_kwargs = {
            "device": compute_device, 
            "trust_remote_code": True,
            "model_kwargs": {
                "torch_dtype": torch_dtype if torch_dtype is not None else None
            }
        }

        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 8}

        if compute_device.lower() == 'cpu':
            encode_kwargs['batch_size'] = 2
        else:
            batch_size_mapping = {
                't5-xxl': 2,
                't5-xl': 2,
                'instructor-xl': 2,
                'stella': 2,
                'gte-large': 4,
                't5-large': 4,
                'bge-large': 4,
                'instructor-large': 4,
                'e5-large': 4,
                'arctic-embed-l': 4,
                't5-base': 6,
                'e5-small': 16,
                'bge-small': 16,
                'Granite-30m-English': 16,
            }

            for key, value in batch_size_mapping.items():
                if isinstance(key, tuple):
                    if any(model_name_part in embedding_model_name for model_name_part in key):
                        encode_kwargs['batch_size'] = value
                        break
                else:
                    if key in embedding_model_name:
                        encode_kwargs['batch_size'] = value
                        break

        if "instructor" in embedding_model_name.lower():
            model = InstructorEmbedding(embedding_model_name, model_kwargs, encode_kwargs).create()

        elif "snowflake" in embedding_model_name.lower():
            model = SnowflakeEmbedding(embedding_model_name, model_kwargs, encode_kwargs).create()

        elif "Alibaba" in embedding_model_name.lower():
            model = AlibabaEmbedding(embedding_model_name, model_kwargs, encode_kwargs).create()

        elif "stella" in embedding_model_name.lower():
            model = StellaEmbedding(embedding_model_name, model_kwargs, encode_kwargs).create()

        else:
            model = BaseEmbeddingModel(embedding_model_name, model_kwargs, encode_kwargs).create()

        model_name = os.path.basename(embedding_model_name)
        precision = "float32" if torch_dtype is None else str(torch_dtype).split('.')[-1]
        my_cprint(f"{model_name} ({precision}) loaded using a batch size of {encode_kwargs['batch_size']}.", "green")

        return model, encode_kwargs

    @torch.inference_mode()
    def create_database(self, texts, embeddings):
        my_cprint("\nComputing vectors...", "yellow")
        start_time = time.time()
        
        hash_id_mappings = []
        MAX_UINT64 = 18446744073709551615

        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {self.PERSIST_DIRECTORY}")
        else:
            logging.warning(f"Directory already exists: {self.PERSIST_DIRECTORY}")

        try:
            # Initialize collections for all chunks
            all_texts = []
            all_metadatas = []
            all_ids = []
            chunk_counters = defaultdict(int)

            # Process all texts and generate IDs
            for doc in texts:
                file_hash = doc.metadata.get('hash')
                chunk_counters[file_hash] += 1
                tiledb_id = str(random.randint(0, MAX_UINT64 - 1))
                
                all_texts.append(doc.page_content)
                all_metadatas.append(doc.metadata)
                all_ids.append(tiledb_id)
                hash_id_mappings.append((tiledb_id, file_hash))

            with open(self.ROOT_DIRECTORY / "config.yaml", 'r', encoding='utf-8') as config_file:
                config_data = yaml.safe_load(config_file)

            TileDB.create(
                index_uri=str(self.PERSIST_DIRECTORY),
                index_type="FLAT",
                dimensions=config_data.get("EMBEDDING_MODEL_DIMENSIONS"),
                vector_type=np.float32,
                metadatas=True
            )

            db = TileDB.load(
                index_uri=str(self.PERSIST_DIRECTORY),
                embedding=embeddings,
                metric="cosine",
                allow_dangerous_deserialization=True
            )

            try:
                db.add_texts(
                    texts=all_texts,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                my_cprint(f"Processed {len(all_texts)} chunks", "yellow")
            except Exception as e:
                logging.error(f"Error processing texts: {str(e)}")
                raise

            end_time = time.time()
            elapsed_time = end_time - start_time
            my_cprint(f"Database created. Elapsed time: {elapsed_time:.2f} seconds.", "green")
            
            return hash_id_mappings

        except Exception as e:
            logging.error(f"Error creating database: {str(e)}")
            if self.PERSIST_DIRECTORY.exists():
                try:
                    shutil.rmtree(self.PERSIST_DIRECTORY)
                    logging.info(f"Cleaned up failed database creation at: {self.PERSIST_DIRECTORY}")
                except Exception as cleanup_error:
                    logging.error(f"Failed to clean up database directory: {cleanup_error}")
            raise

    def create_metadata_db(self, documents, hash_id_mappings):
        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        sqlite_db_path = self.PERSIST_DIRECTORY / "metadata.db"
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                hash TEXT,
                file_path TEXT,
                page_content TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hash_chunk_ids (
                tiledb_id TEXT PRIMARY KEY,
                hash TEXT
            )
        ''')

        try:
            for document in documents:
                metadata = document.metadata
                cursor.execute('''
                    INSERT INTO document_metadata (file_name, hash, file_path, page_content)
                    VALUES (?, ?, ?, ?)
                ''', (
                    metadata.get("file_name", ""),
                    metadata.get("hash", ""),
                    metadata.get("file_path", ""),
                    document.page_content
                ))

            for tiledb_id, file_hash in hash_id_mappings:
                cursor.execute('''
                    INSERT INTO hash_chunk_ids (tiledb_id, hash)
                    VALUES (?, ?)
                ''', (tiledb_id, file_hash))

            conn.commit()
        finally:
            conn.close()

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

    @torch.inference_mode()
    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        
        # list to hold "document objects"        
        documents = []
        
        # load text document objects
        text_documents = load_documents(self.SOURCE_DIRECTORY)
        if isinstance(text_documents, list) and text_documents:
            documents.extend(text_documents)

        # separate lists for pdf and non-pdf document objects
        text_documents_pdf = [doc for doc in documents if doc.metadata.get("file_type") == ".pdf"]
        documents = [doc for doc in documents if doc.metadata.get("file_type") != ".pdf"]
        
        # load image descriptions
        print("Loading any images...")
        image_documents = choose_image_loader()
        if isinstance(image_documents, list) and image_documents:
            if len(image_documents) > 0:
                documents.extend(image_documents)
        
        # load audio transcriptions
        print("Loading any audio transcripts...")
        audio_documents = self.load_audio_documents()
        if isinstance(audio_documents, list) and audio_documents:
            documents.extend(audio_documents)

        # create a list to save pre-split text for sqliteDB
        json_docs_to_save = []
        json_docs_to_save.extend(documents)
        json_docs_to_save.extend(text_documents_pdf)

        # blank list to hold all split document objects
        texts = []

        # split document objects and add to list
        if (isinstance(documents, list) and documents) or (isinstance(text_documents_pdf, list) and text_documents_pdf):
            texts = split_documents(documents, text_documents_pdf)
            print(f"Documents split into {len(texts)} chunks.")

        del documents, text_documents_pdf
        gc.collect()

        # create db
        if isinstance(texts, list) and texts:
            embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            # Get hash->ID mappings along with creating the vector database
            hash_id_mappings = self.create_database(texts, embeddings)

            del texts
            gc.collect()

            # Pass mappings to metadata db creation
            self.create_metadata_db(json_docs_to_save, hash_id_mappings)
            del json_docs_to_save
            gc.collect()
            self.clear_docs_for_db_folder()


class QueryVectorDB:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, selected_database):
        self.config = self.load_configuration()
        self.selected_database = selected_database
        self.embeddings = None
        self.db = None
        self.model_name = None
        self._debug_id = id(self)
        logging.debug(f"Created new QueryVectorDB instance {self._debug_id} for database {selected_database}")

    @classmethod
    def get_instance(cls, selected_database):
        with cls._instance_lock:
            if cls._instance is not None:
                if cls._instance.selected_database != selected_database:
                    print(f"Database changed from {cls._instance.selected_database} to {selected_database}")
                    cls._instance.cleanup()
                    cls._instance = None
                else:
                    logging.debug(f"Reusing existing instance {cls._instance._debug_id} for database {selected_database}")

            if cls._instance is None:
                cls._instance = cls(selected_database)

            return cls._instance

    def load_configuration(self):
        config_path = Path(__file__).resolve().parent / 'config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    @torch.inference_mode()
    def initialize_vector_model(self):     
        model_path = self.config['created_databases'][self.selected_database]['model']
        self.model_name = os.path.basename(model_path)
        compute_device = self.config['Compute_Device']['database_query']

        model_kwargs = {
            "device": compute_device, 
            "trust_remote_code": True,
            "model_kwargs": {}
        }
        encode_kwargs = {'normalize_embeddings': True}

        # Create the appropriate embedding model
        if "instructor" in model_path.lower():
            embeddings = InstructorEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()
        elif "snowflake" in model_path.lower():
            embeddings = SnowflakeEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()
        elif "Alibaba" in model_path.lower():
            embeddings = AlibabaEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()
        elif "stella" in model_path.lower():
            embeddings = StellaEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()
        else:
            if "bge" in model_path.lower():
                encode_kwargs["prompt"] = "Represent this sentence for searching relevant passages: "
            embeddings = BaseEmbeddingModel(model_path, model_kwargs, encode_kwargs, is_query=True).create()

        return embeddings

    def initialize_database(self):
        persist_directory = Path(__file__).resolve().parent / "Vector_DB" / self.selected_database
        
        return TileDB.load(index_uri=str(persist_directory), embedding=self.embeddings, allow_dangerous_deserialization=True)

    def is_special_prefix_model(self):
        model_path = self.config['created_databases'][self.selected_database]['model']
        return "intfloat" in model_path.lower() or "snowflake" in model_path.lower()

    @torch.inference_mode()
    def search(self, query, k: Optional[int] = None, score_threshold: Optional[float] = None):
        if not self.embeddings:
            logging.info(f"Initializing embedding model for database {self.selected_database}")
            self.embeddings = self.initialize_vector_model()

        if not self.db:
            logging.info(f"Initializing database connection for {self.selected_database}")
            self.db = self.initialize_database()

        self.config = self.load_configuration()
        document_types = self.config['database'].get('document_types', '')
        search_filter = {'document_type': document_types} if document_types else {}
        
        k = k if k is not None else int(self.config['database']['contexts'])
        score_threshold = score_threshold if score_threshold is not None else float(self.config['database']['similarity'])

        if self.is_special_prefix_model():
            query = f"query: {query}"

        relevant_contexts = self.db.similarity_search_with_score(
            query,
            k=k,
            filter=search_filter,
            score_threshold=score_threshold
        )

        search_term = self.config['database'].get('search_term', '').lower()
        filtered_contexts = [(doc, score) for doc, score in relevant_contexts if search_term in doc.page_content.lower()]

        contexts = [document.page_content for document, _ in filtered_contexts]
        metadata_list = [document.metadata for document, _ in filtered_contexts]
        scores = [score for _, score in filtered_contexts]

        for metadata, score in zip(metadata_list, scores):
            metadata['similarity_score'] = score

        return contexts, metadata_list

    def cleanup(self):
        logging.info(f"Cleaning up QueryVectorDB instance {self._debug_id} for database {self.selected_database}")
        
        if self.embeddings:
            logging.debug(f"Unloading embedding model for database {self.selected_database}")
            del self.embeddings
            self.embeddings = None

        if self.db:
            logging.debug(f"Closing database connection for {self.selected_database}")
            del self.db
            self.db = None

        if torch.cuda.is_available():
            logging.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()

        gc.collect()
        logging.debug(f"Cleanup completed for instance {self._debug_id}")

        # my_cprint(f"{self.model_name} removed from memory.", "red")


# class DeleteFromVectorDB:
    # def __init__(self, selected_database):
       # """
       # Initialize with path to the selected vector database.
       # """
       # self.ROOT_DIRECTORY = Path(__file__).resolve().parent
       # self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / selected_database

    # def get_tiledb_ids_for_hash(self, file_hash):
       # """
       # Retrieve all TileDB IDs associated with a given hash from SQLite database.
       # """
       # sqlite_db_path = self.PERSIST_DIRECTORY / "metadata.db"
       # conn = sqlite3.connect(sqlite_db_path)
       # cursor = conn.cursor()

       # try:
           # cursor.execute('''
               # SELECT tiledb_id FROM hash_chunk_ids 
               # WHERE hash = ?
           # ''', (file_hash,))

           # tiledb_ids = [row[0] for row in cursor.fetchall()]
           # return tiledb_ids
       # finally:
           # conn.close()

    # def delete_from_sqlite(self, file_hash, conn):
       # """
       # Delete all entries associated with the hash from both SQLite tables.
       # """
       # cursor = conn.cursor()

       # # Delete from document_metadata table
       # cursor.execute('''
           # DELETE FROM document_metadata 
           # WHERE hash = ?
       # ''', (file_hash,))

       # # Delete from hash_chunk_ids table
       # cursor.execute('''
           # DELETE FROM hash_chunk_ids 
           # WHERE hash = ?
       # ''', (file_hash,))

    # def delete_from_tiledb(self, tiledb_ids):
        # """
        # Delete vectors from TileDB using their IDs.
        # """
        # try:
            # db = TileDB.load(
                # index_uri=str(self.PERSIST_DIRECTORY),
                # embedding=None,  # We don't need embeddings for deletion
                # allow_dangerous_deserialization=True
            # )

            # success = db.delete(ids=tiledb_ids)
            # return success
        # except Exception as e:
            # logging.error(f"TileDB deletion error: {str(e)}")
            # raise

    # def delete_vectors(self, file_hashes: list[str]):
        # """
        # Main method to handle deletion from both TileDB and SQLite databases with transaction support.
        # Ensures atomic operations - either all deletions succeed or none do.

        # Args:
            # file_hashes (list[str]): List of file hashes whose vectors should be deleted

        # Returns:
            # bool: True if deletion was successful, False otherwise

        # Raises:
            # Exception: If deletion process fails
        # """
        # sqlite_db_path = self.PERSIST_DIRECTORY / "metadata.db"
        # conn = sqlite3.connect(sqlite_db_path)

        # try:
            # conn.execute("BEGIN TRANSACTION")

            # # Get all TileDB IDs for all hashes
            # all_tiledb_ids = []
            # for file_hash in file_hashes:
                # tiledb_ids = self.get_tiledb_ids_for_hash(file_hash)
                # if not tiledb_ids:
                    # logging.warning(f"No vectors found for hash {file_hash}")
                    # continue
                # all_tiledb_ids.extend(tiledb_ids)

            # if not all_tiledb_ids:
                # logging.warning("No vectors found for any of the provided hashes")
                # conn.rollback()
                # return False

            # # Delete from TileDB first
            # tiledb_success = self.delete_from_tiledb(all_tiledb_ids)

            # if not tiledb_success:
                # logging.error("TileDB deletion failed")
                # conn.rollback()
                # return False

            # # If TileDB deletion succeeded, delete from SQLite for all hashes
            # for file_hash in file_hashes:
                # self.delete_from_sqlite(file_hash, conn)

            # conn.commit()

            # logging.info(f"Successfully deleted {len(all_tiledb_ids)} vectors across {len(file_hashes)} hashes")
            # return True

        # except Exception as e:
            # logging.error(f"Error during deletion: {str(e)}")
            # conn.rollback()
            # raise
        # finally:
            # conn.close()

    # def verify_deletion(self, file_hashes: list[str]):
        # """
        # Verify that all traces of the hashes have been removed from both databases.
        # """
        # all_clean = True
        # remaining_tiledb_count = 0
        # remaining_metadata_count = 0

        # for file_hash in file_hashes:
           # tiledb_ids = self.get_tiledb_ids_for_hash(file_hash)
           # if tiledb_ids:
               # remaining_tiledb_count += len(tiledb_ids)
               # all_clean = False
               
        # if remaining_tiledb_count > 0:
           # logging.warning(f"Found {remaining_tiledb_count} remaining TileDB IDs across all hashes")

        # sqlite_db_path = self.PERSIST_DIRECTORY / "metadata.db"
        # conn = sqlite3.connect(sqlite_db_path)
        # cursor = conn.cursor()

        # try:
           # placeholders = ','.join('?' * len(file_hashes))
           # cursor.execute(f'''
               # SELECT COUNT(*) FROM document_metadata 
               # WHERE hash IN ({placeholders})
           # ''', file_hashes)

           # remaining_metadata_count = cursor.fetchone()[0]
           # if remaining_metadata_count > 0:
               # logging.warning(f"Found {remaining_metadata_count} remaining metadata entries across all hashes")
               # all_clean = False

           # return all_clean
        # finally:
           # conn.close()

    # def cleanup(self):
        # if torch.cuda.is_available():
           # torch.cuda.empty_cache()
        # gc.collect()