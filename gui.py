from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QTabWidget,
    QTextEdit, QSplitter, QFrame, QStyleFactory, QLabel, QHBoxLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
import yaml
import server_connector
import os
from download_model import download_embedding_model
from select_model import select_embedding_model_directory
from choose_documents import choose_documents_directory
import create_database
from metrics_gpu import GPU_Monitor
from metrics_system import SystemMonitor
from initialize import determine_compute_device, is_nvidia_gpu, get_os_name
from voice_recorder_module import VoiceRecorder
from gui_tabs import create_tabs
from gui_threads import CreateDatabaseThread, SubmitButtonThread

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

tabs_config = config.get('tabs', [])
styles = config.get('styles', {})

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.compute_device = determine_compute_device()
        os_name = get_os_name()

        self.init_ui()

        if self.compute_device != "mps" and os_name == "windows" and is_nvidia_gpu():
            self.gpu_monitor = GPU_Monitor(self.gpu_vram_label, self.gpu_util_label, self)
            self.system_monitor = SystemMonitor(self.cpu_label, self.ram_label, self.ram_usage_label, self)
        else:
            self.gpu_monitor = None
            self.system_monitor = None

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.setWindowTitle('LM Studio ChromaDB Plugin - www.chintellalaw.com')
        self.setGeometry(300, 300, 850, 900)
        self.setMinimumSize(550, 610)

        self.left_frame = QFrame()
        left_vbox = QVBoxLayout()

        tab_widget = create_tabs()
        left_vbox.addWidget(tab_widget)

        button_data = [
            ("Download Embedding Model", lambda: download_embedding_model(self)),
            ("Select Embedding Model Directory", select_embedding_model_directory),
            ("Choose Documents for Database", choose_documents_directory),
            ("Create Vector Database", self.on_create_button_clicked)
        ]
        for text, handler in button_data:
            button = QPushButton(text)
            button.setStyleSheet(styles.get('button', ''))
            button.clicked.connect(handler)
            left_vbox.addWidget(button)

        self.left_frame.setLayout(left_vbox)
        self.left_frame.setStyleSheet(styles.get('frame', ''))
        main_splitter.addWidget(self.left_frame)

        right_frame = QFrame()
        right_vbox = QVBoxLayout()
        self.read_only_text = QTextEdit()
        self.read_only_text.setReadOnly(True)
        self.read_only_text.setStyleSheet(styles.get('text', ''))

        self.text_input = QTextEdit()
        self.text_input.setStyleSheet(styles.get('input', ''))
        submit_button = QPushButton("Submit Question")
        submit_button.setStyleSheet(styles.get('button', ''))
        submit_button.clicked.connect(self.on_submit_button_clicked)

        right_vbox.addWidget(self.read_only_text, 4)
        right_vbox.addWidget(self.text_input, 1)
        right_vbox.addWidget(submit_button)
        
        self.recorder = VoiceRecorder()

        self.start_button = QPushButton("Start Recording")
        self.start_button.setStyleSheet(styles.get('button', ''))
        self.start_button.clicked.connect(self.start_recording)
        right_vbox.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setStyleSheet(styles.get('button', ''))
        self.stop_button.clicked.connect(self.stop_recording)
        right_vbox.addWidget(self.stop_button)

        right_frame.setLayout(right_vbox)
        right_frame.setStyleSheet(styles.get('frame', ''))
        main_splitter.addWidget(right_frame)

        metrics_frame = QFrame()
        metrics_frame.setFixedHeight(28)
        metrics_layout = QHBoxLayout(metrics_frame)

        metrics_labels = [
            ("VRAM: N/A", "gpu_vram_label"),
            ("GPU: N/A", "gpu_util_label"),
            ("CPU: N/A", "cpu_label"),
            ("RAM: N/A", "ram_label"),
            ("RAM Usage: N/A", "ram_usage_label")
        ]

        for text, attribute_name in metrics_labels:
            label = QLabel(text)
            setattr(self, attribute_name, label)
            metrics_layout.addWidget(label)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_splitter)
        main_layout.addWidget(metrics_frame)

    def resizeEvent(self, event):
        self.left_frame.setMaximumWidth(self.width() * 0.5)
        super().resizeEvent(event)

    def on_create_button_clicked(self):
        self.create_database_thread = CreateDatabaseThread(self)
        self.create_database_thread.start()

    def on_submit_button_clicked(self):
        user_question = self.text_input.toPlainText()
        self.submit_button_thread = SubmitButtonThread(user_question, self)
        self.submit_button_thread.responseSignal.connect(self.update_response)
        self.submit_button_thread.start()

    def start_recording(self):
        self.recorder.start_recording()

    def stop_recording(self):
        self.recorder.stop_recording()

    def update_response(self, response):
        self.read_only_text.setPlainText(response)

    def closeEvent(self, event):
        if self.gpu_monitor:
            self.gpu_monitor.stop_and_exit_gpu_monitor()
        if self.system_monitor:
            self.system_monitor.stop_and_exit_system_monitor()
        event.accept()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())
