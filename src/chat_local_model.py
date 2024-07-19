import time
import logging
import gc

import torch
from pathlib import Path
from multiprocessing import Process, Pipe
from multiprocessing.connection import PipeConnection  # New import
from PySide6.QtCore import QObject, Signal

import module_chat
from database_interactions import QueryVectorDB
from utilities import my_cprint

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='local_model_chat.log')

class LocalModelSignals(QObject):
    response_signal = Signal(str)
    citations_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    model_loaded_signal = Signal()
    model_unloaded_signal = Signal()  # New signal

class LocalModelChat:
    def __init__(self):
        self.model_process = None
        self.model_pipe = None
        self.current_model = None
        self.signals = LocalModelSignals()

    def start_model_process(self, model_name):
        logging.info(f"Starting model process for {model_name}")
        if self.current_model != model_name:
            if self.is_model_loaded():
                self.terminate_current_process()
            
            parent_conn, child_conn = Pipe()
            self.model_pipe = parent_conn
            self.model_process = Process(target=self._local_model_process, args=(child_conn, model_name))
            self.model_process.start()
            self.current_model = model_name
            self._start_listening_thread()
            self.signals.model_loaded_signal.emit()
            logging.info(f"Model process started for {model_name}")
        else:
            logging.info(f"Model {model_name} is already loaded")

    def terminate_current_process(self):
        logging.info("Terminating current process")
        if self.model_process is not None:
            try:
                if self.model_pipe:
                    try:
                        self.model_pipe.send(("exit", None))
                    except (BrokenPipeError, OSError):
                        logging.warning("Pipe already closed")
                    finally:
                        self.model_pipe.close()
                        self.model_pipe = None
                
                # Store the process in a local variable
                process = self.model_process
                # Immediately set self.model_process to None
                self.model_process = None
                
                # Now use the local variable for further operations
                if process.is_alive():
                #if process is not None and process.is_alive():
                    process.join(timeout=10)
                    if process.is_alive():
                        logging.warning("Process did not terminate, forcing termination")
                        process.terminate()
                        process.join(timeout=5)
            except Exception as e:
                logging.exception(f"Error during process termination: {e}")
        else:
            logging.info("No process to terminate")

        # Ensure all resources are cleared
        self.model_pipe = None
        self.model_process = None
        self.current_model = None
        gc.collect()
        time.sleep(0.5)
        logging.info("Current process terminated and resources cleared")
        self.signals.model_unloaded_signal.emit()

    def start_chat(self, user_question, chunks_only, selected_model, selected_database):
        if not self.model_pipe:
            self.signals.error_signal.emit("Model not loaded. Please start a model first.")
            return

        self.model_pipe.send(("question", (user_question, chunks_only, selected_model, selected_database)))

    def is_model_loaded(self):
        return self.model_process is not None and self.model_process.is_alive()

    def eject_model(self):
        logging.info("Ejecting model")
        self.terminate_current_process()
        logging.info("Model ejected")

    def _start_listening_thread(self):
        import threading
        threading.Thread(target=self._listen_for_response, daemon=True).start()

    def _listen_for_response(self):
        logging.info("Starting response listener")
        while True:
            if not self.model_pipe or not isinstance(self.model_pipe, PipeConnection):
                logging.info("Model pipe is no longer valid. Exiting listener.")
                break
            
            try:
                if self.model_pipe.poll(timeout=1):
                    message_type, message = self.model_pipe.recv()
                    if message_type == "response":
                        self.signals.response_signal.emit(message)
                    elif message_type == "citations":
                        self.signals.citations_signal.emit(message)
                    elif message_type == "error":
                        self.signals.error_signal.emit(message)
                    elif message_type == "finished":
                        self.signals.finished_signal.emit()
                        if message == "exit":
                            logging.info("Received exit signal, terminating listener")
                            break
                else:
                    time.sleep(0.1)
            except (BrokenPipeError, EOFError, OSError) as e:
                logging.info(f"Pipe communication error: {e}. Model likely ejected. Exiting listener.")
                break
            except Exception as e:
                logging.exception(f"Unexpected error in _listen_for_response: {e}")
                break

        logging.info("Response listener terminated")
        self.cleanup_listener_resources()

    def cleanup_listener_resources(self):
        logging.info("Cleaning up listener resources")
        self.model_pipe = None
        self.model_process = None
        self.current_model = None
        self.signals.model_unloaded_signal.emit()  # Emit the new signal
        logging.info("Listener resource cleanup completed")

    @staticmethod
    def _local_model_process(conn, model_name):
        model_instance = module_chat.choose_model(model_name)
        query_vector_db = None
        current_database = None
        try:
            while True:
                try:
                    message_type, message = conn.recv()
                    if message_type == "question":
                        user_question, chunks_only, _, selected_database = message
                        if query_vector_db is None or current_database != selected_database:
                            query_vector_db = QueryVectorDB(selected_database)
                            current_database = selected_database
                        contexts, metadata_list = query_vector_db.search(user_question)
                        if not contexts:
                            conn.send(("error", "No relevant contexts found."))
                            conn.send(("finished", None))
                            continue
                        if chunks_only:
                            formatted_contexts = LocalModelChat._format_contexts_and_metadata(contexts, metadata_list)
                            conn.send(("response", formatted_contexts))
                            conn.send(("finished", None))
                            continue
                        prepend_string = "Here are the contexts to base your answer on, but I need to reiterate only base your response on these contexts and do not use outside knowledge that you may have been trained with."
                        augmented_query = f"{prepend_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + user_question
                        response = module_chat.generate_response(model_instance, augmented_query)
                        conn.send(("response", response))
                        with open('chat_history.txt', 'w', encoding='utf-8') as f:
                            f.write(response)
                        citations = LocalModelChat._format_metadata_as_citations(metadata_list)
                        conn.send(("citations", citations))
                        conn.send(("finished", None))
                    elif message_type == "exit":
                        break
                except EOFError:
                    logging.error("Connection closed by main process.")
                    break
                except Exception as e:
                    logging.exception(f"Error in local_model_process: {e}")
                    conn.send(("error", str(e)))
                    conn.send(("finished", None))
        finally:
            module_chat.cleanup_resources(model_instance, model_instance.tokenizer)
            conn.close()
            logging.info("Local chat model removed from memory.")

    @staticmethod
    def _format_contexts_and_metadata(contexts, metadata_list):
        formatted_contexts = []
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = metadata.get('file_name', 'Unknown')
            formatted_context = (
                f"---------- Context {index} | From File: {file_name} ----------\n"
                f"{context}\n"
            )
            formatted_contexts.append(formatted_context)
        return "\n".join(formatted_contexts)

    @staticmethod
    def _format_metadata_as_citations(metadata_list):
        citations = [Path(metadata['file_path']).name for metadata in metadata_list]
        unique_citations = set(citations)
        return "\n".join(unique_citations)

def is_cuda_available():
    return torch.cuda.is_available()