import os
import logging
import signal
import threading
from pathlib import Path

import torch
import yaml
from PySide6.QtCore import QThread, Signal, QObject, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout, QMessageBox,
                               QApplication, QComboBox, QLabel)

from chat_lm_studio import LMStudioChatThread
from chat_local_model import LocalModelChat
from constants import CHUNKS_ONLY_TOOLTIP, SPEAK_RESPONSE_TOOLTIP, CHAT_MODELS
from module_tts import run_tts
from module_voice_recorder import VoiceRecorder
from utilities import check_preconditions_for_submit_question, my_cprint

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log')

current_dir = Path(__file__).resolve().parent
input_text_file = str(current_dir / 'chat_history.txt')

class RefreshingComboBox(QComboBox):
    def __init__(self, parent=None):
        super(RefreshingComboBox, self).__init__(parent)

    def showPopup(self):
        self.clear()
        self.addItems(self.parent().load_created_databases())
        super(RefreshingComboBox, self).showPopup()

class GuiSignals(QObject):
    response_signal = Signal(str)
    citations_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()

class DatabaseQueryTab(QWidget):
    def __init__(self):
        super(DatabaseQueryTab, self).__init__()
        self.config_path = Path(__file__).resolve().parent / 'config.yaml'
        self.lm_studio_chat_thread = None
        self.local_model_chat = LocalModelChat()
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
        self.model_combo_box.setEnabled(False)  # Initially disabled as LM Studio is default
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
        self.local_model_chat.signals.response_signal.connect(self.update_response_local_model)
        self.local_model_chat.signals.citations_signal.connect(self.display_citations_in_widget)
        self.local_model_chat.signals.error_signal.connect(self.show_error_message)
        self.local_model_chat.signals.finished_signal.connect(self.on_submission_finished)
        self.local_model_chat.signals.model_loaded_signal.connect(self.on_model_loaded)
        self.local_model_chat.signals.model_unloaded_signal.connect(self.on_model_unloaded)  # New line

    def on_model_source_changed(self, text):
        if text == "Local Model":
            self.model_combo_box.setEnabled(torch.cuda.is_available())
            self.eject_button.setEnabled(self.local_model_chat.is_model_loaded())
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
            self.lm_studio_chat_thread = LMStudioChatThread(user_question, chunks_only, selected_database)
            self.lm_studio_chat_thread.lm_studio_chat.signals.response_signal.connect(self.update_response_lm_studio)
            self.lm_studio_chat_thread.lm_studio_chat.signals.error_signal.connect(self.show_error_message)
            self.lm_studio_chat_thread.lm_studio_chat.signals.finished_signal.connect(self.on_submission_finished)
            self.lm_studio_chat_thread.lm_studio_chat.signals.citation_signal.connect(self.display_citations_in_widget)
            self.lm_studio_chat_thread.start()
        else:  # Local Model
            selected_model = self.model_combo_box.currentText()
            try:
                if selected_model != self.local_model_chat.current_model:
                    if self.local_model_chat.is_model_loaded():
                        self.local_model_chat.terminate_current_process()
                    self.local_model_chat.start_model_process(selected_model)
                self.local_model_chat.start_chat(user_question, chunks_only, selected_model, selected_database)
            except Exception as e:
                logging.exception(f"Error starting or using local model: {e}")
                self.show_error_message(f"Error with local model: {str(e)}")
                self.submit_button.setDisabled(False)

    def eject_model(self):
        if self.local_model_chat.is_model_loaded():
            try:
                self.local_model_chat.eject_model()
            except Exception as e:
                logging.exception(f"Error during model ejection: {e}")
            finally:
                self.eject_button.setEnabled(False)
                self.model_combo_box.setEnabled(True)
        else:
            logging.info("No model is currently loaded.")

    def on_model_unloaded(self):
        self.eject_button.setEnabled(False)
        self.model_combo_box.setEnabled(True)

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

    def update_response_local_model(self, response):
        self.read_only_text.setPlainText(response)
        if not self.chunks_only_checkbox.isChecked():
            self.eject_button.setEnabled(True)

    def show_error_message(self, error_message):
        QMessageBox.warning(self, "Error", error_message)
        self.submit_button.setDisabled(False)

    def on_submission_finished(self):
        self.submit_button.setDisabled(False)

    def update_transcription(self, transcription_text):
        self.text_input.setPlainText(transcription_text)

    def cleanup(self):
        if self.local_model_chat.is_model_loaded():
            self.local_model_chat.eject_model()
        print("Cleanup completed")