from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget, QTextEdit, QSplitter, QFrame,
    QStyleFactory, QLabel, QGridLayout, QMenuBar, QCheckBox, QHBoxLayout, QMessageBox, QPushButton
)
from PySide6.QtGui import QIcon, QPixmap
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

class SubmitButtonThread(QThread):
    responseSignal = Signal(str)
    stop_requested = False

    def __init__(self, user_question, parent=None, callback=None):
        super(SubmitButtonThread, self).__init__(parent)
        self.user_question = user_question
        self.callback = callback

    def run(self):
        try:
            response = server_connector.ask_local_chatgpt(self.user_question)
            for response_chunk in response:
                if SubmitButtonThread.stop_requested:
                    break
                self.responseSignal.emit(response_chunk)
            if self.callback:
                self.callback()
        except Exception as err:
            self.errorSignal.emit()
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
        self.load_config()
        self.init_menu()

    def is_nvidia_gpu(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "nvidia" in gpu_name.lower()

    def load_config(self):
        script_dir = Path(__file__).resolve().parent
        config_path = os.path.join(script_dir, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.test_embeddings_checkbox.setChecked(config.get('test_embeddings', False))

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.setWindowTitle('LM Studio ChromaDB Plugin - www.chintellalaw.com')
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

        right_vbox.addWidget(self.read_only_text, 4)
        right_vbox.addWidget(self.text_input, 1)

        # Horizontal layout for submit and stop buttons
        submit_stop_layout = QHBoxLayout()
        self.submit_button = QPushButton("Submit Questions")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        submit_stop_layout.addWidget(self.submit_button)

        self.stop_button = QPushButton()
        stop_icon_data = base64.b64decode(IMAGE_STOP_SIGN)
        stop_icon_pixmap = QPixmap()
        stop_icon_pixmap.loadFromData(stop_icon_data)
        self.stop_button.setIcon(QIcon(stop_icon_pixmap))
        self.stop_button.clicked.connect(self.on_stop_button_clicked)
        submit_stop_layout.addWidget(self.stop_button)

        submit_stop_layout.setStretchFactor(self.submit_button, 5)
        submit_stop_layout.setStretchFactor(self.stop_button, 1)

        right_vbox.addLayout(submit_stop_layout)

        # Horizontal layout for bark and new stop button
        bark_new_stop_layout = QHBoxLayout()
        bark_button = QPushButton("Bark Response")
        bark_button.setToolTip(SPEAK_RESPONSE_TOOLTIP)
        bark_button.clicked.connect(self.on_bark_button_clicked)
        bark_new_stop_layout.addWidget(bark_button)

        new_stop_button = QPushButton()
        new_stop_button.setIcon(QIcon(stop_icon_pixmap))  # Using the same icon
        # No functionality assigned yet
        bark_new_stop_layout.addWidget(new_stop_button)

        bark_new_stop_layout.setStretchFactor(bark_button, 5)
        bark_new_stop_layout.setStretchFactor(new_stop_button, 1)

        right_vbox.addLayout(bark_new_stop_layout)

        # Test Embeddings checkbox
        self.test_embeddings_checkbox = QCheckBox("Chunks Only")
        self.test_embeddings_checkbox.setToolTip(CHUNKS_ONLY_TOOLTIP)
        self.test_embeddings_checkbox.stateChanged.connect(self.on_test_embeddings_changed)
        right_vbox.addWidget(self.test_embeddings_checkbox)

        # Create and add button row for recording
        button_row_widget = self.create_button_row()
        right_vbox.addWidget(button_row_widget)

        right_frame.setLayout(right_vbox)
        main_splitter.addWidget(right_frame)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_splitter)

        # Metrics bar
        main_layout.addWidget(self.metrics_bar)
        self.metrics_bar.setMaximumHeight(75 if self.is_nvidia_gpu() else 30)

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
        SubmitButtonThread.stop_requested = False
        script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

        # check preconditions
        is_valid, error_message = check_preconditions_for_submit_question(script_dir)
        if not is_valid:
            QMessageBox.warning(self, "Error", error_message)
            return

        self.submit_button.setDisabled(True)
        self.submit_button.setText("Processing...")
        user_question = self.text_input.toPlainText()
        self.submit_button_thread = SubmitButtonThread(user_question, self)
        self.cumulative_response = ""
        self.submit_button_thread.responseSignal.connect(self.update_response)
        self.submit_button_thread.start()

        # 3 second timer
        self.reset_timer = QTimer(self)
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.enable_submit_button)
        self.reset_timer.start(3000)

    def on_stop_button_clicked(self):
        SubmitButtonThread.request_stop()

    def enable_submit_button(self):
        self.submit_button.setDisabled(False)
        self.submit_button.setText("Submit Questions")

    def on_test_embeddings_changed(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.yaml')

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        config['test_embeddings'] = self.test_embeddings_checkbox.isChecked()

        with open(config_path, 'w') as file:
            yaml.dump(config, file)

    def update_response(self, response):
        self.cumulative_response += response
        self.read_only_text.setPlainText(self.cumulative_response)
        self.submit_button.setDisabled(False)

    def update_transcription(self, text):
        self.text_input.setPlainText(text)
    
    def closeEvent(self, event):
        self.metrics_bar.stop_metrics_collector()
        event.accept()
    
    def create_button_row(self):
        voice_recorder = voice_recorder_module.VoiceRecorder(self)

        def start_recording():
            voice_recorder.start_recording()

        def stop_recording():
            voice_recorder.stop_recording()

        start_button = QPushButton("Start Recording")
        start_button.clicked.connect(start_recording)

        stop_button = QPushButton("Stop Recording")
        stop_button.clicked.connect(stop_recording)

        hbox = QHBoxLayout()
        hbox.addWidget(start_button)
        hbox.addWidget(stop_button)

        hbox.setStretchFactor(start_button, 3)
        hbox.setStretchFactor(stop_button, 3)

        row_widget = QWidget()
        row_widget.setLayout(hbox)

        return row_widget

    def on_bark_button_clicked(self):
        script_dir = Path(__file__).resolve().parent
        chat_history_path = script_dir / 'chat_history.txt'

        if not chat_history_path.exists():
            QMessageBox.warning(self, "Error", 
                                "You must connect to LM Studio and get a response first before attempting to hear the response.")
            return

        bark_thread = threading.Thread(target=self.run_bark_module)
        bark_thread.daemon = True
        bark_thread.start()

    def run_bark_module(self):
        bark_audio = BarkAudio()
        bark_audio.run()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('fusion'))
    stylesheet = load_stylesheet('custom_stylesheet_steel_ocean.css')
    app.setStyleSheet(stylesheet)
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())
