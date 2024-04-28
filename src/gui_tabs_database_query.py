from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout, QMessageBox, QApplication, QComboBox, QRadioButton, QLabel
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QThread, Signal
import torch
import base64
import yaml
from constants import CHUNKS_ONLY_TOOLTIP, SPEAK_RESPONSE_TOOLTIP, IMAGE_STOP_SIGN
import server_connector
from database_interactions import QueryVectorDB
from voice_recorder_module import VoiceRecorder
from tts_module import BarkAudio, WhisperSpeechAudio
import threading
from pathlib import Path
from utilities import check_preconditions_for_submit_question, my_cprint
from constants import CHAT_MODELS
import module_chat

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

    def __init__(self, user_question, chunks_only, parent=None):
        super(SubmitButtonThread, self).__init__(parent)
        self.user_question = user_question
        self.chunks_only = chunks_only

    def run(self):
        try:
            response = server_connector.ask_local_chatgpt(self.user_question, self.chunks_only)
            for response_chunk in response:
                if SubmitButtonThread.stop_requested:
                    my_cprint(f"stop_requested set to true...exiting", "green")
                    break
                self.responseSignal.emit(response_chunk)
            self.finishedSignal.emit()
        except Exception as err:
            self.errorSignal.emit("Connection to server failed. Please ensure the external server is running.")
            print(err)

    @classmethod
    def request_stop(cls):
        cls.stop_requested = True

class LocalModelThread(QThread):
    responseSignal = Signal(str)
    citationsSignal = Signal(str)
    errorSignal = Signal(str)
    finishedSignal = Signal()

    def __init__(self, user_question, chunks_only, selected_model, parent=None):
        super(LocalModelThread, self).__init__(parent)
        self.user_question = user_question
        self.chunks_only = chunks_only
        self.selected_model = selected_model

    def run(self):
        try:
            query_vector_db = QueryVectorDB('config.yaml')
            contexts, metadata_list = query_vector_db.search(self.user_question)

            if not contexts:
                self.errorSignal.emit("No relevant contexts found.")
                self.finishedSignal.emit()
                return

            if self.chunks_only:
                formatted_contexts = self.format_contexts_and_metadata(contexts, metadata_list)
                self.responseSignal.emit(formatted_contexts)
                self.finishedSignal.emit()
                return

            prepend_string = "Here are the contexts to base your answer on, but I need to reiterate only base your response on these contexts and do not use outside knowledge that you may have been trained with."
            augmented_query = f"{prepend_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + self.user_question

            model_function = getattr(module_chat, CHAT_MODELS[self.selected_model]['function'])
            response = model_function(augmented_query, None, None)
            self.responseSignal.emit(response)

            with open('chat_history.txt', 'w', encoding='utf-8') as f:
                f.write(response)

            citations = self.format_metadata_as_citations(metadata_list)
            self.citationsSignal.emit("\n\n" + citations)

            self.finishedSignal.emit()
        except Exception as err:
            self.errorSignal.emit(str(err))

    def format_contexts_and_metadata(self, contexts, metadata_list):
        formatted_contexts = []
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = metadata.get('file_name', 'Unknown')
            formatted_context = (
                f"---------- Context {index} | From File: {file_name} ----------\n"
                f"{context}\n"
            )
            formatted_contexts.append(formatted_context)
        return "\n".join(formatted_contexts)

    def format_metadata_as_citations(self, metadata_list):
        citations = [Path(metadata['file_path']).name for metadata in metadata_list]
        unique_citations = set(citations)
        return "\n".join(unique_citations)

class DatabaseQueryTab(QWidget):
    def __init__(self):
        super(DatabaseQueryTab, self).__init__()
        self.config_path = Path(__file__).resolve().parent / 'config.yaml'
        self.initWidgets()

    def initWidgets(self):
        layout = QVBoxLayout(self)

        self.read_only_text = QTextEdit()
        self.read_only_text.setReadOnly(True)
        layout.addWidget(self.read_only_text, 4)

        hbox1_layout = QHBoxLayout()

        selected_database_label = QLabel("Selected Database:")
        hbox1_layout.addWidget(selected_database_label)

        self.database_pulldown = RefreshingComboBox(self)
        self.database_pulldown.addItems(self.load_created_databases())
        self.database_pulldown.currentIndexChanged.connect(self.on_database_selected)
        hbox1_layout.addWidget(self.database_pulldown)

        self.chunks_only_checkbox = QCheckBox("Chunks Only")
        self.chunks_only_checkbox.setToolTip(CHUNKS_ONLY_TOOLTIP)
        hbox1_layout.addWidget(self.chunks_only_checkbox)

        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(model_info['model'] for model_info in CHAT_MODELS.values())
        self.model_combo_box.setCurrentText("Neural-Chat - 7b")  # Set the default model
        hbox1_layout.addWidget(self.model_combo_box)

        self.local_model_radio = QRadioButton("Local Model")
        self.lm_studio_radio = QRadioButton("LM Studio")
        self.lm_studio_radio.setChecked(True)
        self.lm_studio_radio.toggled.connect(lambda checked: not checked and self.local_model_radio.setChecked(True))
        self.local_model_radio.toggled.connect(lambda checked: not checked and self.lm_studio_radio.setChecked(True))
        hbox1_layout.addWidget(self.local_model_radio)
        hbox1_layout.addWidget(self.lm_studio_radio)

        # Disable the Local Model radio button and model combo box if CUDA is not available
        if not torch.cuda.is_available():
            self.local_model_radio.setEnabled(False)
            self.model_combo_box.setEnabled(False)
            tooltip_text = "The Local Model option is only supported with GPU acceleration."
            self.local_model_radio.setToolTip(tooltip_text)
            self.model_combo_box.setToolTip(tooltip_text)
            # Applying style sheet to grey out disabled widgets
            disabled_style = "QRadioButton:disabled, QComboBox:disabled { color: #707070; }"
            self.local_model_radio.setStyleSheet(disabled_style)
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

        self.record_button = QPushButton("Voice Recorder (click to record)")
        self.record_button.clicked.connect(self.toggle_recording)
        hbox2_layout.addWidget(self.record_button)

        self.submit_button = QPushButton("Submit Question")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        hbox2_layout.addWidget(self.submit_button)

        layout.addLayout(hbox2_layout)

        self.is_recording = False
        self.voice_recorder = VoiceRecorder(self)


    def on_database_selected(self, index):
        selected_database = self.database_pulldown.itemText(index)
        self.update_config_selected_database(selected_database)
    
    def update_config_selected_database(self, database_name):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            config['database']['database_to_search'] = database_name
            
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(config, file)
    
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

        if self.lm_studio_radio.isChecked():
            self.submit_thread = SubmitButtonThread(user_question, chunks_only, self)
            self.submit_thread.responseSignal.connect(self.update_response_lm_studio)
            self.submit_thread.errorSignal.connect(self.show_error_message)
            self.submit_thread.finishedSignal.connect(self.on_submission_finished)
            self.submit_thread.start()
        else:
            selected_model = self.model_combo_box.currentText()
            self.submit_thread = LocalModelThread(user_question, chunks_only, selected_model, self)
            self.submit_thread.responseSignal.connect(self.update_response_local_model)
            self.submit_thread.citationsSignal.connect(self.display_citations_in_widget)
            self.submit_thread.errorSignal.connect(self.show_error_message)
            self.submit_thread.finishedSignal.connect(self.on_submission_finished)
            self.submit_thread.start()

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
        # Check if PyTorch with CUDA is available
        if not torch.cuda.is_available():
            QMessageBox.warning(self, "Error", "Text to Speech is currently only supported with GPU acceleration.")
            return

        script_dir = Path(__file__).resolve().parent
        if not (script_dir / 'chat_history.txt').exists():
            QMessageBox.warning(self, "Error", "No response to play.")
            return
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            tts_model = config.get('tts', {}).get('model', 'bark')
        
        if tts_model == 'bark':
            tts_thread = threading.Thread(target=self.run_bark_module)
        elif tts_model == 'whisperspeech':
            tts_thread = threading.Thread(target=self.run_whisperspeech_module)
        else:
            QMessageBox.warning(self, "Error", "Invalid TTS model specified in the configuration.")
            return
        
        tts_thread.daemon = True
        tts_thread.start()

    def run_whisperspeech_module(self):
        whisperspeech_audio = WhisperSpeechAudio()
        whisperspeech_audio.run()

    def run_bark_module(self):
        bark_audio = BarkAudio()
        bark_audio.run()

    def toggle_recording(self):
        if self.is_recording:
            self.voice_recorder.stop_recording()
            self.record_button.setText("Record Question (click to record)")
        else:
            self.voice_recorder.start_recording()
            self.record_button.setText("Recording... (click to stop)")
        self.is_recording = not self.is_recording

    def update_response_lm_studio(self, response_chunk):
        self.cumulative_response += response_chunk
        self.read_only_text.setPlainText(self.cumulative_response)
        self.submit_button.setDisabled(False)

    def update_response_local_model(self, response):
        self.read_only_text.setPlainText(response)
        self.submit_button.setDisabled(False)

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