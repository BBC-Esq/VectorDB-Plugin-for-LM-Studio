import logging
from pathlib import Path
import multiprocessing

import torch
import yaml
from PySide6.QtCore import QThread, Signal, QObject, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout, QMessageBox,
                               QApplication, QComboBox, QLabel, QTextBrowser)

from chat_lm_studio import LMStudioChatThread
from chat_local_model import LocalModelChat
from constants import CHAT_MODELS
from module_voice_recorder import VoiceRecorder
from utilities import check_preconditions_for_submit_question, my_cprint
from constants import TOOLTIPS
from database_interactions import QueryVectorDB, process_chunks_only_query

current_dir = Path(__file__).resolve().parent
input_text_file = str(current_dir / 'chat_history.txt')


class ChunksOnlyThread(QThread):
    chunks_ready = Signal(str)

    def __init__(self, query, database_name):
        super().__init__()
        self.query = query
        self.database_name = database_name
        self.process = None

    def run(self):
        try:
            result_queue = multiprocessing.Queue()
            
            self.process = multiprocessing.Process(
                target=process_chunks_only_query,
                args=(self.database_name, self.query, result_queue)
            )
            self.process.start()
            
            result = result_queue.get()
            self.chunks_ready.emit(result)
            
            self.process.join()
            self.process = None
            
        except Exception as e:
            logging.exception(f"Error in chunks only thread: {e}")
            self.chunks_ready.emit(f"Error querying database: {str(e)}")

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

def run_tts_in_process(config_path, input_text_file):
    from module_tts import run_tts  # Import here to avoid potential circular imports
    run_tts(config_path, input_text_file)
    my_cprint("TTS models removed from memory.", "red")


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

class CustomTextBrowser(QTextBrowser):
    '''
    Inherits from QTextBrowser but overrides the doSetSource method to ensure that "http," "https," and "file" schemes are opened
    with the system's default program while other types are still opened however the QTextBrowswer normally does.
    
    The following examples allow for additional handling and/or make the opening internal or external based on the type of link.
        
    1. Handle mailto links within the application to provide a custom email interface
    2. Blocking or handling all other schemes internally
    3. Log every http and https link click to a file or database before opening it in the default browser.
    4. Display a warning message or confirmation dialog for ftp links before proceeding
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)

    def doSetSource(self, name, type):
        if name.scheme() == 'file':
            QDesktopServices.openUrl(QUrl.fromLocalFile(name.toLocalFile()))
        elif name.scheme() in ['http', 'https']:
            QDesktopServices.openUrl(name)
        else:
            super().doSetSource(name, type)

class DatabaseQueryTab(QWidget):
    def __init__(self):
        super(DatabaseQueryTab, self).__init__()
        self.config_path = Path(__file__).resolve().parent / 'config.yaml'
        self.lm_studio_chat_thread = None
        self.local_model_chat = LocalModelChat()
        self.gui_signals = GuiSignals()
        self.current_model_name = None
        self.database_query_thread = None
        self.initWidgets()
        self.setup_signals()

    def initWidgets(self):
        layout = QVBoxLayout(self)

        self.response_widget = CustomTextBrowser()
        self.response_widget.setOpenExternalLinks(True)
        layout.addWidget(self.response_widget, 5)

        self.token_count_label = QLabel("")
        layout.addWidget(self.token_count_label)

        hbox1_layout = QHBoxLayout()

        self.database_pulldown = RefreshingComboBox(self)
        self.database_pulldown.setToolTip(TOOLTIPS["DATABASE_SELECT"])
        self.database_pulldown.addItems(self.load_created_databases())
        hbox1_layout.addWidget(self.database_pulldown)

        self.model_source_combo = QComboBox()
        self.model_source_combo.setToolTip(TOOLTIPS["MODEL_BACKEND_SELECT"])
        self.model_source_combo.addItems(["LM Studio", "Local Model"])
        self.model_source_combo.setCurrentText("LM Studio")
        self.model_source_combo.currentTextChanged.connect(self.on_model_source_changed)
        hbox1_layout.addWidget(self.model_source_combo)

        self.model_combo_box = QComboBox()
        self.model_combo_box.setToolTip(TOOLTIPS["LOCAL_MODEL_SELECT"])

        if torch.cuda.is_available():
            for model_info in CHAT_MODELS.values():
                index = self.model_combo_box.count()
                self.model_combo_box.addItem(model_info['model'])
                gb_value = round(model_info['vram'] / 1024, 1)
                tooltip = f"Uses ~{gb_value} GB memory"
                self.model_combo_box.setItemData(index, tooltip, Qt.ToolTipRole)
            self.model_combo_box.setEnabled(True)
        else:
            self.model_combo_box.addItem(CHAT_MODELS['Zephyr - 1.6b']['model'])
            self.model_combo_box.setToolTip("The Local Model option requires GPU-acceleration.")

        if self.model_combo_box.count() > 0:
            self.model_combo_box.setCurrentIndex(0)

        hbox1_layout.addWidget(self.model_combo_box)

        self.eject_button = QPushButton("Eject Local Model")
        self.eject_button.setToolTip(TOOLTIPS["EJECT_LOCAL_MODEL"])
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
        self.text_input.setToolTip(TOOLTIPS["QUESTION_INPUT"])
        layout.addWidget(self.text_input, 1)

        hbox2_layout = QHBoxLayout()

        self.copy_response_button = QPushButton("Copy Response")
        self.copy_response_button.setToolTip(TOOLTIPS["COPY_RESPONSE"])
        self.copy_response_button.clicked.connect(self.on_copy_response_clicked)
        hbox2_layout.addWidget(self.copy_response_button)

        self.bark_button = QPushButton("Speak Response")
        self.bark_button.setToolTip(TOOLTIPS["SPEAK_RESPONSE"])
        self.bark_button.clicked.connect(self.on_bark_button_clicked)
        hbox2_layout.addWidget(self.bark_button)
        
        self.chunks_only_checkbox = QCheckBox("Chunks Only")
        self.chunks_only_checkbox.setToolTip(TOOLTIPS["CHUNKS_ONLY"])
        hbox2_layout.addWidget(self.chunks_only_checkbox)

        self.record_button = QPushButton("Voice Recorder")
        self.record_button.setToolTip(TOOLTIPS["VOICE_RECORDER"])
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
        self.local_model_chat.signals.model_unloaded_signal.connect(self.on_model_unloaded)
        self.local_model_chat.signals.token_count_signal.connect(self.update_token_count_label)

    def update_token_count_label(self, token_count_string):
        self.token_count_label.setText(token_count_string)

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
                databases = list(config.get('created_databases', {}).keys())
                return [db for db in databases if db != "user_manual"]
        return []

    def on_submit_button_clicked(self):
        script_dir = Path(__file__).resolve().parent
        is_valid, error_message = check_preconditions_for_submit_question(script_dir)
        if not is_valid:
            QMessageBox.warning(self, "Error", error_message)
            return

        self.response_widget.clear()
        self.token_count_label.clear()
        
        self.response_widget.clear()
        self.response_widget.setPlainText("")
        self.response_widget.setHtml("")
        self.response_widget.document().clear()
        cursor = self.response_widget.textCursor()
        cursor.clearSelection()
        self.response_widget.setTextCursor(cursor)
        
        self.cumulative_response = ""
        self.submit_button.setDisabled(True)
        user_question = self.text_input.toPlainText()
        chunks_only = self.chunks_only_checkbox.isChecked()
        
        selected_database = self.database_pulldown.currentText()
        if chunks_only: # only get chunks
            self.database_query_thread = ChunksOnlyThread(user_question, selected_database)
            self.database_query_thread.chunks_ready.connect(self.display_chunks)
            self.database_query_thread.finished.connect(self.on_database_query_finished)
            self.database_query_thread.start()
        else: # lm studio or local model
            if self.model_source_combo.currentText() == "LM Studio":
                self.lm_studio_chat_thread = LMStudioChatThread(user_question, selected_database)
                self.lm_studio_chat_thread.lm_studio_chat.signals.response_signal.connect(self.update_response_lm_studio)
                self.lm_studio_chat_thread.lm_studio_chat.signals.error_signal.connect(self.show_error_message)
                self.lm_studio_chat_thread.lm_studio_chat.signals.finished_signal.connect(self.on_submission_finished)
                self.lm_studio_chat_thread.lm_studio_chat.signals.citation_signal.connect(self.display_citations_in_widget)
                self.lm_studio_chat_thread.start()
            else:  # local models
                selected_model = self.model_combo_box.currentText()
                try:
                    if selected_model != self.local_model_chat.current_model:
                        if self.local_model_chat.is_model_loaded():
                            self.local_model_chat.terminate_current_process()
                        self.local_model_chat.start_model_process(selected_model)
                    self.local_model_chat.start_chat(user_question, selected_model, selected_database)
                except Exception as e:
                    logging.exception(f"Error starting or using local model: {e}")
                    self.show_error_message(f"Error with local model: {str(e)}")
                    self.submit_button.setDisabled(False)

    def display_chunks(self, chunks):
        self.response_widget.setPlainText(chunks)

    def on_database_query_finished(self):
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
            logging.warning("No model is currently loaded.")

    def on_model_loaded(self):
        self.eject_button.setEnabled(True)
        self.eject_button.setText(f"Eject {self.local_model_chat.current_model}")

    def on_model_unloaded(self):
        self.eject_button.setEnabled(False)
        self.eject_button.setText("Eject Local Model")


    def display_citations_in_widget(self, citations):
        if citations:
            self.response_widget.append(f"<br>Citation Links:{citations}")
        else:
            self.response_widget.append("\n\nNo citations found.")

    def on_copy_response_clicked(self):
        clipboard = QApplication.clipboard()
        response_text = self.response_widget.toPlainText()
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

        self.run_tts_module()

    def run_tts_module(self):
        process = multiprocessing.Process(target=run_tts_in_process, args=(str(self.config_path), input_text_file))
        process.start()

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
        self.response_widget.setPlainText(self.cumulative_response)
        self.response_widget.verticalScrollBar().setValue(
            self.response_widget.verticalScrollBar().maximum()
        )

    def update_response_local_model(self, response):
        current_text = self.response_widget.toPlainText()
        self.response_widget.setPlainText(current_text + response)
        self.response_widget.verticalScrollBar().setValue(
            self.response_widget.verticalScrollBar().maximum()
        )
        if not self.chunks_only_checkbox.isChecked():
            self.eject_button.setEnabled(True)

    def show_error_message(self, error_message):
        # error message if the contexts exceed the chat model's limit
        if "exceed the chat model's context limit" in error_message:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(error_message)
            msg_box.setWindowTitle("Context Limit Exceeded")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
        # all other error messages
        else:
            QMessageBox.warning(self, "Error", error_message)
        self.submit_button.setDisabled(False)

    def on_submission_finished(self):
        self.submit_button.setDisabled(False)

    def update_transcription(self, transcription_text):
        self.text_input.setPlainText(transcription_text)

    def cleanup(self):
        if self.local_model_chat.is_model_loaded():
            self.local_model_chat.eject_model()
        if self.database_query_thread and self.database_query_thread.isRunning():
            self.database_query_thread.wait()
        print("Cleanup completed")