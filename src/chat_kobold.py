import gc
import json
import logging
from pathlib import Path
import torch
import yaml
import requests
import sseclient
from PySide6.QtCore import QThread, Signal, QObject

from database_interactions import QueryVectorDB
from utilities import format_citations, normalize_chat_text
from constants import rag_string

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

class KoboldSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citation_signal = Signal(str)

class KoboldChat:
    def __init__(self):
        self.signals = KoboldSignals()
        self.config = self.load_configuration()
        self.query_vector_db = None
        self.api_url = "http://localhost:5001/api/extra/generate/stream"

    def load_configuration(self):
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def connect_to_kobold(self, augmented_query):
        payload = {
            "prompt": augmented_query,
            "max_context_length": 8192,
            "max_length": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        try:
            response = requests.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()
            client = sseclient.SSEClient(response)

            for event in client.events():
                if event.event == "message":
                    try:
                        data = json.loads(event.data)
                        if 'token' in data:
                            yield data['token']
                    except json.JSONDecodeError:
                        logging.error(f"Failed to parse JSON: {event.data}")
                        raise ValueError(f"Failed to parse response: {event.data}")
        except Exception as e:
            logging.error(f"Error in Kobold API request: {str(e)}")
            raise

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)
        
        if self.query_vector_db:
            self.query_vector_db.cleanup()
            # print("Embedding model removed from memory.")

        if torch.cuda.empty_cache():
            torch.cuda.empty_cache()
        gc.collect()

        return citations

    def save_metadata_to_file(self, metadata_list):
        with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
            for metadata in metadata_list:
                output_file.write(f"{metadata}\n")

    def ask_kobold(self, query, selected_database):
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            self.query_vector_db = QueryVectorDB(selected_database)

        contexts, metadata_list = self.query_vector_db.search(query)

        self.save_metadata_to_file(metadata_list)

        if not contexts:
            self.signals.error_signal.emit("No relevant contexts found.")
            self.signals.finished_signal.emit()
            return

        augmented_query = f"{rag_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + f"\n\n-----\n\n{query}"

        full_response = ""
        response_generator = self.connect_to_kobold(augmented_query)
        for response_chunk in response_generator:
            self.signals.response_signal.emit(response_chunk)
            full_response += response_chunk

        with open('chat_history.txt', 'w', encoding='utf-8') as f:
            normalized_response = normalize_chat_text(full_response)
            f.write(normalized_response)

        self.signals.response_signal.emit("\n")

        citations = self.handle_response_and_cleanup(full_response, metadata_list)
        self.signals.citation_signal.emit(citations)
        self.signals.finished_signal.emit()

class KoboldThread(QThread):
    def __init__(self, query, selected_database):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.kobold_chat = KoboldChat()

    def run(self):
        try:
            self.kobold_chat.ask_kobold(self.query, self.selected_database)
        except Exception as e:
            logging.error(f"Error in KoboldThread: {str(e)}")
            self.kobold_chat.signals.error_signal.emit(str(e))