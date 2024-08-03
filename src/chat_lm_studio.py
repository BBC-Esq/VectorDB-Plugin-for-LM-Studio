import gc
import os
from pathlib import Path

import torch
import yaml
from openai import OpenAI
from PySide6.QtCore import QThread, Signal, QObject

from database_interactions import QueryVectorDB
from utilities import my_cprint

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

class LMStudioSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citation_signal = Signal(str)

class LMStudioChat:
    def __init__(self):
        self.signals = LMStudioSignals()
        self.config = self.load_configuration()
        self.query_vector_db = None

    def load_configuration(self):
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def connect_to_local_chatgpt(self, prompt):
        server_config = self.config.get('server', {})
        base_url = server_config.get('connection_str')
        prefix = server_config.get('prefix', '')
        suffix = server_config.get('suffix', '')
        prompt_format_disabled = server_config.get('prompt_format_disabled')
        model_temperature = server_config.get('model_temperature')
        model_max_tokens = server_config.get('model_max_tokens')

        client = OpenAI(base_url=base_url, api_key='not-needed')

        formatted_prompt = prompt if prompt_format_disabled else f"{prefix}{prompt}{suffix}"

        stream = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=model_temperature,
            max_tokens=model_max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = self.format_metadata_as_citations(metadata_list)
        unique_citations = set(citations.split("\n"))
        
        if self.query_vector_db and self.query_vector_db.embeddings:
            del self.query_vector_db.embeddings.client
            del self.query_vector_db.embeddings
        torch.cuda.empty_cache()
        gc.collect()
        print("Embedding model removed from memory.")
        
        return "\n".join(unique_citations)

    def save_metadata_to_file(self, metadata_list):
        with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
            for metadata in metadata_list:
                output_file.write(f"{metadata}\n")

    def format_metadata_as_citations(self, metadata_list):
        citations = [Path(metadata['file_path']).name for metadata in metadata_list]
        return "\n".join(citations)

    def yield_formatted_contexts(self, contexts, metadata_list):
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = metadata.get('file_name', 'Unknown')
            formatted_context = f"---------- Context {index} | From File: {file_name} ----------\n\n{context}\n\n"
            yield formatted_context

    def ask_local_chatgpt(self, query, chunks_only, selected_database):
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            self.query_vector_db = QueryVectorDB(selected_database)
        
        contexts, metadata_list = self.query_vector_db.search(query)
        
        self.save_metadata_to_file(metadata_list)
        
        if not contexts:
            self.signals.error_signal.emit("No relevant contexts found.")
            self.signals.finished_signal.emit()
            return
        
        if chunks_only:
            for formatted_context in self.yield_formatted_contexts(contexts, metadata_list):
                self.signals.response_signal.emit(formatted_context)
            self.signals.finished_signal.emit()
            return
        
        prepend_string = "Only base your answer on the provided context/contexts. If you cannot, please state so."
        augmented_query = f"{prepend_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + f"\n\n-----\n\n{query}"
        
        full_response = ""
        response_generator = self.connect_to_local_chatgpt(augmented_query)
        for response_chunk in response_generator:
            self.signals.response_signal.emit(response_chunk)
            full_response += response_chunk
        
        with open('chat_history.txt', 'w', encoding='utf-8') as f:
            f.write(full_response)
        
        self.signals.response_signal.emit("\n\n")
        
        citations = self.handle_response_and_cleanup(full_response, metadata_list)
        self.signals.citation_signal.emit(citations)
        self.signals.finished_signal.emit()

class LMStudioChatThread(QThread):
    def __init__(self, query, chunks_only, selected_database):
        super().__init__()
        self.query = query
        self.chunks_only = chunks_only
        self.selected_database = selected_database
        self.lm_studio_chat = LMStudioChat()

    def run(self):
        try:
            self.lm_studio_chat.ask_local_chatgpt(self.query, self.chunks_only, self.selected_database)
        except Exception as e:
            self.lm_studio_chat.signals.error_signal.emit(str(e))

def is_lm_studio_available():
    # This function should check if LM Studio is available and running
    # For now, we'll just return True as a placeholder
    return True