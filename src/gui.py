from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QTabWidget,
    QTextEdit, QSplitter, QFrame, QStyleFactory, QLabel, QGridLayout, QMenuBar, QCheckBox, QHBoxLayout
)
from PySide6.QtCore import Qt
import os
import torch
import yaml
import sys
from initialize import main as initialize_system
from metrics_bar import MetricsBar
from download_model import download_embedding_model
from select_model import select_embedding_model_directory
from choose_documents import choose_documents_directory, see_documents_directory
from choose_documents import choose_documents_directory
import create_database
from gui_tabs import create_tabs
from gui_threads import CreateDatabaseThread, SubmitButtonThread
import voice_recorder_module
from utilities import list_theme_files, make_theme_changer, load_stylesheet

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.cumulative_response = ""
        initialize_system()
        self.metrics_bar = MetricsBar()
        self.compute_device = self.metrics_bar.determine_compute_device()
        os_name = self.metrics_bar.get_os_name()
        self.submit_button = None
        self.init_ui()
        self.init_menu()
        self.load_config()

    def is_nvidia_gpu(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "nvidia" in gpu_name.lower()
        return False
    
    def load_config(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.test_embeddings_checkbox.setChecked(config.get('test_embeddings', False))

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.setWindowTitle('LM Studio ChromaDB Plugin - www.chintellalaw.com')
        self.setGeometry(300, 300, 975, 975)
        self.setMinimumSize(450, 510)

        # LEFT FRAME
        self.left_frame = QFrame()
        grid_layout = QGridLayout()
        
        tab_widget = create_tabs()
        grid_layout.addWidget(tab_widget, 0, 0, 1, 2)
        
        # Buttons data
        button_data = [
            ("Download Embedding Model", lambda: download_embedding_model(self)),
            ("Choose Embedding Model Directory", select_embedding_model_directory),
            ("Choose Documents for Database", choose_documents_directory),
            ("See Currently Chosen Documents", see_documents_directory),
            ("Create Vector Database", self.on_create_button_clicked)
        ]
        button_positions = [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
        
        # Create and add buttons
        for position, (text, handler) in zip(button_positions, button_data):
            button = QPushButton(text)
            button.clicked.connect(handler)
            grid_layout.addWidget(button, *position)

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

        self.submit_button = QPushButton("Submit Questions")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        right_vbox.addWidget(self.submit_button)

        self.test_embeddings_checkbox = QCheckBox("Test Embeddings")
        self.test_embeddings_checkbox.stateChanged.connect(self.on_test_embeddings_changed)
        right_vbox.addWidget(self.test_embeddings_checkbox)

        # Create and add button row
        button_row_widget = self.create_button_row(self.on_submit_button_clicked)
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

    def on_create_button_clicked(self):
        self.create_database_thread = CreateDatabaseThread(self)
        self.create_database_thread.start()

    def enable_submit_button(self):
        self.submit_button.setDisabled(False)
        self.submit_button.setText("Submit Questions")

    def on_submit_button_clicked(self):
        if self.submit_button.isEnabled():
            self.submit_button.setDisabled(True)
            self.submit_button.setText("Processing...")
            user_question = self.text_input.toPlainText()
            self.submit_button_thread = SubmitButtonThread(user_question, self, self.enable_submit_button)
            self.cumulative_response = ""
            self.submit_button_thread.responseSignal.connect(self.update_response)
            self.submit_button_thread.errorSignal.connect(self.enable_submit_button)
            self.submit_button_thread.start()

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

    def create_button_row(self, submit_handler):
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('fusion'))
    stylesheet = load_stylesheet('custom_stylesheet_steel_ocean.css')
    app.setStyleSheet(stylesheet)
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())
