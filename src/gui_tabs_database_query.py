import os
import signal
import threading
from multiprocessing import Process, Pipe
from pathlib import Path

import torch
import yaml
from PySide6.QtCore import QThread, Signal, QObject, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout, QMessageBox,
                               QApplication, QComboBox, QLabel)

import module_chat
import server_connector
from constants import CHUNKS_ONLY_TOOLTIP, SPEAK_RESPONSE_TOOLTIP, CHAT_MODELS
from database_interactions import QueryVectorDB
from module_tts import run_tts
from module_chat import generate_response, cleanup_resources
from module_voice_recorder import VoiceRecorder
from utilities import check_preconditions_for_submit_question, my_cprint

current_dir = Path(__file__).resolve().parent
input_text_file = str(current_dir / 'chat_history.txt')

class RefreshingComboBox(QComboBox):
    def __init__(self, parent=None):
        super(RefreshingComboBox, self).__init__(parent)

    def showPopup(self):
        self.clear()
        self.addItems(self.parent().load_created_databases())
        super(RefreshingComboBox, self).showPopup()

class SubmitButtonThread(QThread):
    responseSignal = Signal(str)
    errorSignal = Signal(str)
    finishedSignal = Signal()
    stop_requested = False

    def __init__(self, user_question, chunks_only, selected_database, parent=None):
        super(SubmitButtonThread, self).__init__(parent)
        self.user_question = user_question
        self.chunks_only = chunks_only
        self.selected_database = selected_database

    def run(self):
        try:
            response = server_connector.ask_local_chatgpt(self.user_question, self.chunks_only, self.selected_database)
            for response_chunk in response:
                if SubmitButtonThread.stop_requested:
                    my_cprint(f"stop_requested set to true...exiting", "green")
                    break
                self.responseSignal.emit(response_chunk)
            self.finishedSignal.emit()
        except Exception as err:
            self.errorSignal.emit(f"Connection to server failed: {str(err)}. Please ensure the external server is running.")
            print(err)

    @classmethod
    def request_stop(cls):
        cls.stop_requested = True

def local_model_process(conn, model_name):
    model_instance = module_chat.choose_model(model_name)
    query_vector_db = None
    current_database = None

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
                    formatted_contexts = format_contexts_and_metadata(contexts, metadata_list)
                    conn.send(("response", formatted_contexts))
                    conn.send(("finished", None))
                    continue

                prepend_string = "Here are the contexts to base your answer on, but I need to reiterate only base your response on these contexts and do not use outside knowledge that you may have been trained with."
                augmented_query = f"{prepend_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + user_question

                response = module_chat.generate_response(model_instance, augmented_query)
                conn.send(("response", response))

                with open('chat_history.txt', 'w', encoding='utf-8') as f:
                    f.write(response)

                citations = format_metadata_as_citations(metadata_list)
                conn.send(("citations", citations))

                conn.send(("finished", None))

            elif message_type == "exit":
                conn.send(("finished", "exit"))
                break

        except EOFError:
            print("Connection closed by main process.")
            break
        except Exception as e:
            print(f"Error in local_model_process: {e}")
            conn.send(("error", str(e)))
            conn.send(("finished", None))

    cleanup_resources(model_instance, model_instance.tokenizer)

def format_contexts_and_metadata(contexts, metadata_list):
    formatted_contexts = []
    for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
        file_name = metadata.get('file_name', 'Unknown')
        formatted_context = (
            f"---------- Context {index} | From File: {file_name} ----------\n"
            f"{context}\n"
        )
        formatted_contexts.append(formatted_context)
    return "\n".join(formatted_contexts)

def format_metadata_as_citations(metadata_list):
    citations = [Path(metadata['file_path']).name for metadata in metadata_list]
    unique_citations = set(citations)
    return "\n".join(unique_citations)

class GuiSignals(QObject):
    response_signal = Signal(str)
    citations_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()

class DatabaseQueryTab(QWidget):
    def __init__(self):
        super(DatabaseQueryTab, self).__init__()
        self.config_path = Path(__file__).resolve().parent / 'config.yaml'
        self.model_process = None
        self.model_pipe = None
        self.gui_signals = GuiSignals()
        self.current_model_name = None
        self.initWidgets()
        self.setup_signals()

    def initWidgets(self):
        layout = QVBoxLayout(self)

        self.read_only_text = QTextEdit()
        self.read_only_text.setReadOnly(True)
        layout.addWidget(self.read_only_text, 5)

        hbox1_layout = QHBoxLayout()

        # selected_database_label = QLabel("Database:")
        # hbox1_layout.addWidget(selected_database_label)

        self.database_pulldown = RefreshingComboBox(self)
        self.database_pulldown.addItems(self.load_created_databases())
        hbox1_layout.addWidget(self.database_pulldown)

        self.model_source_combo = QComboBox()
        self.model_source_combo.addItems(["LM Studio", "Local Model"])
        self.model_source_combo.setCurrentText("LM Studio")  # Set default
        self.model_source_combo.currentTextChanged.connect(self.on_model_source_changed)
        hbox1_layout.addWidget(self.model_source_combo)

        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(model_info['model'] for model_info in CHAT_MODELS.values())
        self.model_combo_box.setCurrentText("Zephyr - 1.6b")  # default model
        self.model_combo_box.setEnabled(True)  # Initially enabled as local model is default
        hbox1_layout.addWidget(self.model_combo_box)

        self.eject_button = QPushButton("Eject Local Model")
        self.eject_button.clicked.connect(self.eject_model)
        self.eject_button.setEnabled(False)
        hbox1_layout.addWidget(self.eject_button)

        if not torch.cuda.is_available():
            self.model_source_combo.setItemData(1, 0, Qt.UserRole - 1)
            tooltip_text = "The Local Model option requires GPU-acceleration."
            self.model_source_combo.setItemData(1, tooltip_text, Qt.ToolTipRole)
            self.model_combo_box.setEnabled(False)
            self.model_combo_box.setToolTip(tooltip_text)
            disabled_style = "QComboBox:disabled { color: #707070; }"
            self.model_combo_box.setStyleSheet(disabled_style)

        layout.addLayout(hbox1_layout)

        self.text_input = QTextEdit()
        layout.addWidget(self.text_input, 1)

        hbox2_layout = QHBoxLayout()

        self.copy_response_button = QPushButton("Copy Response")
        self.copy_response_button.clicked.connect(self.on_copy_response_clicked)
        hbox2_layout.addWidget(self.copy_response_button)

        self.bark_button = QPushButton("Speak Response")
        self.bark_button.clicked.connect(self.on_bark_button_clicked)
        self.bark_button.setToolTip(SPEAK_RESPONSE_TOOLTIP)
        hbox2_layout.addWidget(self.bark_button)
        
        self.chunks_only_checkbox = QCheckBox("Chunks Only")
        self.chunks_only_checkbox.setToolTip(CHUNKS_ONLY_TOOLTIP)
        hbox2_layout.addWidget(self.chunks_only_checkbox)

        self.record_button = QPushButton("Voice Recorder")
        self.record_button.clicked.connect(self.toggle_recording)
        hbox2_layout.addWidget(self.record_button)
        
        self.submit_button = QPushButton("Submit Question")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        hbox2_layout.addWidget(self.submit_button)

        layout.addLayout(hbox2_layout)

        self.is_recording = False
        self.voice_recorder = VoiceRecorder(self)

    def setup_signals(self):
        self.gui_signals.response_signal.connect(self.update_response_local_model)
        self.gui_signals.citations_signal.connect(self.display_citations_in_widget)
        self.gui_signals.error_signal.connect(self.show_error_message)
        self.gui_signals.finished_signal.connect(self.on_submission_finished)

    def on_model_source_changed(self, text):
        if text == "Local Model":
            self.model_combo_box.setEnabled(torch.cuda.is_available())
            self.eject_button.setEnabled(self.model_process is not None)
        else:  # "LM Studio"
            self.model_combo_box.setEnabled(False)
            self.eject_button.setEnabled(False)
    
    def load_created_databases(self):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return list(config.get('created_databases', {}).keys())
        return []

    def on_submit_button_clicked(self):
        script_dir = Path(__file__).resolve().parent
        is_valid, error_message = check_preconditions_for_submit_question(script_dir)
        if not is_valid:
            QMessageBox.warning(self, "Error", error_message)
            return

        self.cumulative_response = ""
        self.submit_button.setDisabled(True)
        user_question = self.text_input.toPlainText()
        chunks_only = self.chunks_only_checkbox.isChecked()
        
        selected_database = self.database_pulldown.currentText()

        if self.model_source_combo.currentText() == "LM Studio":
            self.submit_thread = SubmitButtonThread(user_question, chunks_only, selected_database, self)
            self.submit_thread.responseSignal.connect(self.update_response_lm_studio)
            self.submit_thread.errorSignal.connect(self.show_error_message)
            self.submit_thread.finishedSignal.connect(self.on_submission_finished)
            self.submit_thread.start()
        else:  # Local Model
            selected_model = self.model_combo_box.currentText()
            
            if self.current_model_name != selected_model:
                self.terminate_current_process()
                self.start_new_process(selected_model)

            self.model_pipe.send(("question", (user_question, chunks_only, selected_model, selected_database)))

            # Adjust the state of the eject button based on the chunks only checkbox
            if chunks_only:
                self.eject_button.setEnabled(True)
            else:
                self.eject_button.setEnabled(False)

    def terminate_current_process(self):
        if self.model_process:
            self.model_pipe.send(("exit", None))
            self.model_process.join(timeout=5)  # Wait for up to 5 seconds
            if self.model_process.is_alive():
                self.model_process.terminate()
            self.model_process = None
            self.model_pipe = None

    def start_new_process(self, model_name):
        parent_conn, child_conn = Pipe()
        self.model_pipe = parent_conn
        self.model_process = Process(target=local_model_process, args=(child_conn, model_name))
        self.model_process.start()
        self.current_model_name = model_name
        threading.Thread(target=self.listen_for_response, args=(parent_conn,)).start()
        self.eject_button.setEnabled(True)

    def eject_model(self):
        self.terminate_current_process()
        self.current_model_name = None
        self.eject_button.setEnabled(False)

    def listen_for_response(self, conn):
        while True:
            if conn.poll():
                message_type, message = conn.recv()
                if message_type == "response":
                    self.gui_signals.response_signal.emit(message)
                elif message_type == "citations":
                    self.gui_signals.citations_signal.emit(message)
                elif message_type == "error":
                    self.gui_signals.error_signal.emit(message)
                elif message_type == "finished":
                    self.gui_signals.finished_signal.emit()
                    if message == "exit":
                        break

    def on_model_loaded(self):
        self.eject_button.setEnabled(True)
    
    def display_citations_in_widget(self, citations):
        if citations:
            self.read_only_text.append("\n\nCitations:\n" + citations)
        else:
            self.read_only_text.append("\n\nNo citations found.")
    
    def on_copy_response_clicked(self):
        clipboard = QApplication.clipboard()
        response_text = self.read_only_text.toPlainText()
        if response_text:
            clipboard.setText(response_text)
            QMessageBox.information(self, "Information", "Response copied to clipboard.")
        else:
            QMessageBox.warning(self, "Warning", "No response to copy.")

    def on_bark_button_clicked(self):
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / 'config.yaml'

        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
            tts_config = config.get('tts', {})

        tts_model = tts_config.get('model', '').lower()

        if tts_model not in ['googletts', 'chattts'] and not torch.cuda.is_available():
            QMessageBox.warning(self, "Error", "The Text to Speech backend you selected requires GPU-acceleration.")
            return

        if not (script_dir / 'chat_history.txt').exists():
            QMessageBox.warning(self, "Error", "No response to play.")
            return

        tts_thread = threading.Thread(target=self.run_tts_module)
        tts_thread.daemon = True
        tts_thread.start()

    def run_tts_module(self):
        run_tts(self.config_path, input_text_file)

    def toggle_recording(self):
        if self.is_recording:
            self.voice_recorder.stop_recording()
            self.record_button.setText("Voice Recorder")
        else:
            self.voice_recorder.start_recording()
            self.record_button.setText("Recording...")
        self.is_recording = not self.is_recording

    def update_response_lm_studio(self, response_chunk):
        self.cumulative_response += response_chunk
        self.read_only_text.setPlainText(self.cumulative_response)
        self.submit_button.setDisabled(False)


    def update_response_local_model(self, response):
        self.read_only_text.setPlainText(response)
        self.submit_button.setDisabled(False)
        if not self.chunks_only_checkbox.isChecked():
            self.eject_button.setEnabled(True)

    def display_citations(self, citations):
        if citations:
            QMessageBox.information(self, "Citations", f"The following sources were used:\n\n{citations}")
        else:
            QMessageBox.information(self, "Citations", "No citations found.")

    def show_error_message(self, error_message):
        QMessageBox.warning(self, "Error", error_message)
        self.submit_button.setDisabled(False)

    def on_submission_finished(self):
        self.submit_button.setDisabled(False)

    def update_transcription(self, transcription_text):
        self.text_input.setPlainText(transcription_text)