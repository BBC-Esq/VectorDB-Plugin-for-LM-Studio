import gc
import logging
from pathlib import Path
import os
import torch
import yaml
from openai import OpenAI
from PySide6.QtCore import QThread, Signal, QObject

from database_interactions import QueryVectorDB
from utilities import format_citations, normalize_chat_text
from constants import rag_string, system_message

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

class ChatGPTSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citation_signal = Signal(str)

class ChatGPTChat:
    def __init__(self):
        self.signals = ChatGPTSignals()
        self.config = self.load_configuration()
        self.query_vector_db = None

    def load_configuration(self):
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def connect_to_chatgpt(self, augmented_query):
        openai_config = self.config.get('openai', {})
        model = openai_config.get('model', 'gpt-4o-mini')
        temperature = openai_config.get('temperature', 0.2)
        reasoning_effort = openai_config.get('reasoning_effort', 'medium')
        api_key = openai_config.get('api_key')
        
        if not api_key:
            raise ValueError("OpenAI API key not found in config.yaml.\n\n  Please set it within the 'File' menu.")
        
        client = OpenAI(api_key=api_key)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": augmented_query}
        ]

        # base parameters
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }

        # only for thinking models
        # see here before implementing: https://platform.openai.com/docs/guides/reasoning
        if model in ["o1", "o3-mini"]:
            completion_params["reasoning_effort"] = reasoning_effort

        stream = client.chat.completions.create(**completion_params)

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)

        if self.query_vector_db:
            if hasattr(self.query_vector_db.embeddings, 'client'):
                del self.query_vector_db.embeddings.client
            del self.query_vector_db.embeddings

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Embedding model removed from memory.")

        return citations

    def save_metadata_to_file(self, metadata_list):
        with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
            for metadata in metadata_list:
                output_file.write(f"{metadata}\n")

    def ask_chatgpt(self, query, selected_database):
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
        response_generator = self.connect_to_chatgpt(augmented_query)
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

class ChatGPTThread(QThread):
    def __init__(self, query, selected_database):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.chatgpt_chat = ChatGPTChat()

    def run(self):
        try:
            self.chatgpt_chat.ask_chatgpt(self.query, self.selected_database)
        except Exception as e:
            logging.error(f"Error in ChatGPTThread: {str(e)}")
            self.chatgpt_chat.signals.error_signal.emit(str(e))
