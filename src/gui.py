from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget, QTextEdit, QSplitter, QFrame,
    QStyleFactory, QLabel, QGridLayout, QMenuBar, QCheckBox, QHBoxLayout, QMessageBox, QPushButton
)
from PySide6.QtGui import QIcon, QPixmap, QClipboard
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QByteArray
import os
from pathlib import Path
import torch
import yaml
import sys
import platform
import threading
import base64
from initialize import main as initialize_system
from metrics_bar import MetricsBar
from gui_tabs import create_tabs
import voice_recorder_module
import server_connector
from utilities import list_theme_files, make_theme_changer, load_stylesheet, check_preconditions_for_submit_question
from bark_module import BarkAudio
from constants import CHUNKS_ONLY_TOOLTIP, SPEAK_RESPONSE_TOOLTIP, IMAGE_STOP_SIGN
import openai
import multiprocessing

class SubmitButtonThread(QThread):
    responseSignal = Signal(str)
    errorSignal = Signal(str)
    finishedSignal = Signal()
    stop_requested = False

    def __init__(self, user_question, chunks_only, parent=None, callback=None):
        super(SubmitButtonThread, self).__init__(parent)
        self.user_question = user_question
        self.chunks_only = chunks_only
        self.callback = callback

    def run(self):
        try:
            response = server_connector.ask_local_chatgpt(self.user_question, self.chunks_only)
            for response_chunk in response:
                if SubmitButtonThread.stop_requested:
                    break
                self.responseSignal.emit(response_chunk)
            if self.callback:
                self.callback()
            self.finishedSignal.emit()
        except openai.error.APIConnectionError as err:
            self.errorSignal.emit("Connection to server failed. Please ensure the external server is running.")
            print(err)

    @classmethod
    def request_stop(cls):
        cls.stop_requested = True

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()

        initialize_system()
        self.cumulative_response = ""
        self.metrics_bar = MetricsBar()
        self.compute_device = self.metrics_bar.determine_compute_device()
        self.init_ui()
        self.init_menu()

    def is_nvidia_gpu(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "nvidia" in gpu_name.lower()

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.setWindowTitle('LM Studio ChromaDB Plugin - www.chintellalaw.com')
        # GUI dimensions
        self.setGeometry(300, 300, 1077, 1077)
        self.setMinimumSize(450, 510)

        # LEFT FRAME
        self.left_frame = QFrame()
        grid_layout = QGridLayout()

        tab_widget = create_tabs()
        grid_layout.addWidget(tab_widget, 0, 0, 1, 2)

        self.left_frame.setLayout(grid_layout)
        main_splitter.addWidget(self.left_frame)

        # RIGHT FRAME
        right_frame = QFrame()
        right_vbox = QVBoxLayout()

        self.read_only_text = QTextEdit()
        self.read_only_text.setReadOnly(True)

        self.text_input = QTextEdit()

        # stretch factors
        right_vbox.addWidget(self.read_only_text, 4)
        right_vbox.addWidget(self.text_input, 1)

        submit_stop_layout = QHBoxLayout()
        self.submit_button = QPushButton("Submit Questions")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        submit_stop_layout.addWidget(self.submit_button)

        self.stop_button = QPushButton()
        self.stop_button.setDisabled(True)
        stop_icon_data = base64.b64decode(IMAGE_STOP_SIGN)
        stop_icon_pixmap = QPixmap()
        stop_icon_pixmap.loadFromData(stop_icon_data)
        self.stop_button.setIcon(QIcon(stop_icon_pixmap))
        self.stop_button.clicked.connect(self.on_stop_button_clicked)
        submit_stop_layout.addWidget(self.stop_button)

        # stretch factors
        submit_stop_layout.setStretchFactor(self.submit_button, 6)
        submit_stop_layout.setStretchFactor(self.stop_button, 1)

        right_vbox.addLayout(submit_stop_layout)

        row_two_layout = QHBoxLayout()
        
        self.test_embeddings_checkbox = QCheckBox("Chunks Only")
        self.test_embeddings_checkbox.setToolTip(CHUNKS_ONLY_TOOLTIP)
        row_two_layout.addWidget(self.test_embeddings_checkbox)
        
        self.copy_response_button = QPushButton("Copy Response")
        self.copy_response_button.clicked.connect(self.on_copy_response_clicked)
        row_two_layout.addWidget(self.copy_response_button)

        bark_button = QPushButton("Bark Response")
        bark_button.setToolTip(SPEAK_RESPONSE_TOOLTIP)
        bark_button.clicked.connect(self.on_bark_button_clicked)
        row_two_layout.addWidget(bark_button)

        # stretch factors
        row_two_layout.setStretchFactor(self.test_embeddings_checkbox, 2)
        row_two_layout.setStretchFactor(self.copy_response_button, 3)
        row_two_layout.setStretchFactor(bark_button, 3)

        right_vbox.addLayout(row_two_layout)

        button_row_widget = self.create_button_row()
        right_vbox.addWidget(button_row_widget)

        right_frame.setLayout(right_vbox)
        main_splitter.addWidget(right_frame)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_splitter)

        main_layout.addWidget(self.metrics_bar)

        self.metrics_bar.setMaximumHeight(75 if self.is_nvidia_gpu() else 30)

    def on_copy_response_clicked(self):
        response_text = self.read_only_text.toPlainText()
        clipboard = QApplication.clipboard()
        if response_text:
            clipboard.setText(response_text)
            QMessageBox.information(self, "Information", "Response copied to clipboard.")
        else:
            QMessageBox.warning(self, "Warning", "No response to copy.")
    
    def init_menu(self):
        self.menu_bar = QMenuBar(self)
        self.theme_menu = self.menu_bar.addMenu('Themes')

        self.theme_files = list_theme_files()

        for theme in self.theme_files:
            action = self.theme_menu.addAction(theme)
            action.triggered.connect(make_theme_changer(theme))

    def resizeEvent(self, event):
        self.left_frame.setMaximumWidth(self.width() * 0.5)
        self.left_frame.setMinimumWidth(self.width() * 0.3)
        super().resizeEvent(event)

    def on_submit_button_clicked(self):
        self.stop_button.setDisabled(False)
        SubmitButtonThread.stop_requested = False
        script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

        is_valid, error_message = check_preconditions_for_submit_question(script_dir)
        if not is_valid:
            QMessageBox.warning(self, "Error", error_message)
            return

        self.submit_button.setDisabled(True)
        self.submit_button.setText("Processing...")
        user_question = self.text_input.toPlainText()
        chunks_only_state = self.test_embeddings_checkbox.isChecked()
        self.submit_button_thread = SubmitButtonThread(user_question, chunks_only_state, self)
        self.cumulative_response = ""
        self.submit_button_thread.responseSignal.connect(self.update_response)
        self.submit_button_thread.errorSignal.connect(self.show_error_message)
        self.submit_button_thread.finishedSignal.connect(self.on_task_completed)
        self.submit_button_thread.start()

        self.reset_timer = QTimer(self)
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.enable_submit_button)
        self.reset_timer.start(3000)

    def on_stop_button_clicked(self):
        SubmitButtonThread.request_stop()
        self.stop_button.setDisabled(True)

    def on_task_completed(self):
        self.stop_button.setDisabled(True)

    def enable_submit_button(self):
        self.submit_button.setDisabled(False)
        self.submit_button.setText("Submit Question")

    def update_response(self, response):
        self.cumulative_response += response
        self.read_only_text.setPlainText(self.cumulative_response)
        self.submit_button.setDisabled(False)

    def show_error_message(self, message):
        QMessageBox.warning(self, "Error", message)
        self.stop_button.setDisabled(True)

    def update_transcription(self, text):
        self.text_input.setPlainText(text)
    
    def closeEvent(self, event):
        self.metrics_bar.stop_metrics_collector()
        event.accept()
    
    def create_button_row(self):
        voice_recorder = voice_recorder_module.VoiceRecorder(self)
        self.is_recording = False

        def toggle_recording():
            if self.is_recording:
                voice_recorder.stop_recording()
                record_button.setText("Record Question (click to record)")
            else:
                voice_recorder.start_recording()
                record_button.setText("Recording...(click to stop recording)")
            self.is_recording = not self.is_recording

        record_button = QPushButton("Record Question (click to record)")
        record_button.clicked.connect(toggle_recording)

        hbox = QHBoxLayout()
        hbox.addWidget(record_button)

        hbox.setStretchFactor(record_button, 3)

        row_widget = QWidget()
        row_widget.setLayout(hbox)

        return row_widget

    def on_bark_button_clicked(self):
        script_dir = Path(__file__).resolve().parent
        chat_history_path = script_dir / 'chat_history.txt'

        if not chat_history_path.exists():
            QMessageBox.warning(self, "Error", 
                                "No response to play.")
            return

        bark_thread = threading.Thread(target=self.run_bark_module)
        bark_thread.daemon = True
        bark_thread.start()

    def run_bark_module(self):
        bark_audio = BarkAudio()
        bark_audio.run()
    
def main():
    multiprocessing.set_start_method('spawn')

if __name__ == '__main__':
    main()
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('fusion'))
    stylesheet = load_stylesheet('custom_stylesheet_steel_ocean.css')
    app.setStyleSheet(stylesheet)
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())
