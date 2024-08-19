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
from utilities import format_citations


class LocalModelSignals(QObject):
    response_signal = Signal(str)  # 7. signal for sending response
    citations_signal = Signal(str)  # 8. signal for sending citations
    error_signal = Signal(str)  # 9. signal for sending error messages
    finished_signal = Signal()  # 10. signal for indicating process completion
    model_loaded_signal = Signal()  # 3. signal for indicating model loaded
    model_unloaded_signal = Signal()  # 11. signal for indicating model unloaded
    token_count_signal = Signal(str)

class LocalModelChat:
    def __init__(self):
        self.model_process = None
        self.model_pipe = None  # creates a blank pipe
        self.current_model = None
        self.signals = LocalModelSignals()

    def start_model_process(self, model_name):
        logging.info(f"Starting model process for {model_name}")
        if self.current_model != model_name:
            if self.is_model_loaded():
                self.terminate_current_process()
            
            # turns the blank pipe into a bidirectional pipe with parent process and child process ends
            parent_conn, child_conn = Pipe()
            # establishes the parent process end of the pipe
            self.model_pipe = parent_conn
            # assigns the child process end of the pipe to the _local_model_process method
            self.model_process = Process(target=self._local_model_process, args=(child_conn, model_name))
            self.model_process.start()
            self.current_model = model_name
            self._start_listening_thread()
            # 3. Signal-model is loaded
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
                
                process = self.model_process
                self.model_process = None
                
                if process.is_alive():
                    process.join(timeout=10)
                    if process.is_alive():
                        logging.warning("Process did not terminate, forcing termination")
                        process.terminate()
                        process.join(timeout=5)
            except Exception as e:
                logging.exception(f"Error during process termination: {e}")
        else:
            logging.info("No process to terminate")

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

        # Sends the information selected by a user in gui_tabs_database_query.py to the new child process
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
        """
        This method continuously listens for messages coming through the pipe. When a message is received,
        it's processed based on its type, and then the appropriate signal is emitted. These signals are what
        connect the data received from the pipe to the GUI elements in the DatabaseQueryTab class.
        """
        logging.info("Starting response listener")
        while True:
            if not self.model_pipe or not isinstance(self.model_pipe, PipeConnection):
                logging.info("Model pipe is no longer valid. Exiting listener.")
                break
            
            try:
                # Checks for incoming messages via pipe every 1 second from _local_model_process method
                if self.model_pipe.poll(timeout=1):
                    # 6. process data from pip depending on "message_type"
                    message_type, message = self.model_pipe.recv()
                    if message_type in ["response", "partial_response"]:
                        # 7. Signal-triggers update_response_local_model in database_query_tab
                        self.signals.response_signal.emit(message)
                    elif message_type == "citations":
                        # 8. Signal-triggers display_citations_in_widget in database_query_tab
                        self.signals.citations_signal.emit(message)
                    elif message_type == "error":
                        # 9. Signal-triggers on_submission_finished in database_query_tab
                        self.signals.error_signal.emit(message)
                    elif message_type == "finished":
                        # 10. Signal-triggers on_submission_finished in database_query_tab
                        self.signals.finished_signal.emit()
                        if message == "exit":
                            logging.info("Received exit signal, terminating listener")
                            break
                    # Signal
                    elif message_type == "token_counts":
                        self.signals.token_count_signal.emit(message)
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
        # 11. Signal-triggers on_model_unloaded in database_query_tab
        self.signals.model_unloaded_signal.emit()
        logging.info("Listener resource cleanup completed")

    @staticmethod
    def _local_model_process(conn, model_name): # runs generation within the child process
        model_instance = module_chat.choose_model(model_name)
        query_vector_db = None
        current_database = None
        try:
            while True:
                try:
                    message_type, message = conn.recv()
                    if message_type == "question": # generate streamed response
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
                        # DEBUG
                        # print(augmented_query)

                        # counts tokens using the chosen model's tokenizer
                        prepend_token_count = len(model_instance.tokenizer.encode(prepend_string))
                        context_token_count = len(model_instance.tokenizer.encode("\n\n---\n\n".join(contexts)))
                        user_question_token_count = len(model_instance.tokenizer.encode(user_question))
                        
                        full_response = ""
                        # pipe - "_listen_for_response" in the parent process
                        for partial_response in module_chat.generate_response(model_instance, augmented_query):
                            full_response += partial_response
                            conn.send(("partial_response", partial_response))

                        response_token_count = len(model_instance.tokenizer.encode(full_response))
                        # calculate available tokens
                        remaining_tokens = model_instance.max_length - (prepend_token_count + user_question_token_count + context_token_count + response_token_count)
                        
                        # construct string
                        total_tokens = prepend_token_count + context_token_count + user_question_token_count + response_token_count
                        token_count_string = (
                            f"<span style='color:#2ECC40;'>available tokens ({model_instance.max_length})</span>"
                            f"<span style='color:#FF4136;'> - rag instruction ({prepend_token_count}) - query ({user_question_token_count})"
                            f" - contexts ({context_token_count}) - response ({response_token_count})</span>"
                            f"<span style='color:white;'> = {remaining_tokens} remaining tokens.</span>"
                        )

                        # pipe - "_listen_for_response" in parent process
                        conn.send(("token_counts", token_count_string))

                        with open('chat_history.txt', 'w', encoding='utf-8') as f:
                            f.write(full_response)
                        citations = format_citations(metadata_list)
                        # pipe - "_listen_for_response" in parent process
                        conn.send(("citations", citations))
                        # pipe - "_listen_for_response" in parent process
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

def is_cuda_available():
    return torch.cuda.is_available()