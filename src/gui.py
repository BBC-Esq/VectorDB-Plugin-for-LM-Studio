from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QTabWidget,
    QTextEdit, QSplitter, QFrame, QStyleFactory, QLabel, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
import yaml
import os
from initialize import determine_compute_device, is_nvidia_gpu, get_os_name
from download_model import download_embedding_model
from select_model import select_embedding_model_directory
from choose_documents import choose_documents_directory
import create_database
from metrics_gpu import GPU_Monitor
from metrics_system import SystemMonitor
from gui_tabs import create_tabs
from gui_threads import CreateDatabaseThread, SubmitButtonThread
from metrics_bar import MetricsBar
from button_module import create_button_row

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

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.setWindowTitle('LM Studio ChromaDB Plugin - www.chintellalaw.com')
        self.setGeometry(300, 300, 975, 975)
        self.setMinimumSize(550, 610)

        # Left panel setup with grid layout
        self.left_frame = QFrame()
        grid_layout = QGridLayout()
        
        # Tab widget spanning two columns
        tab_widget = create_tabs()
        grid_layout.addWidget(tab_widget, 0, 0, 1, 2)  # Span two columns
        
        # Button definitions and positions in the grid
        button_data = [
            ("Download Embedding Model", lambda: download_embedding_model(self)),
            ("Set Embedding Model Directory", select_embedding_model_directory),
            ("Choose Documents for Database", choose_documents_directory),
            ("Create Vector Database", self.on_create_button_clicked)
        ]
        button_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        # Create and add buttons to the grid layout
        for position, (text, handler) in zip(button_positions, button_data):
            button = QPushButton(text)
            button.setStyleSheet(styles.get('button', ''))
            button.clicked.connect(handler)
            grid_layout.addWidget(button, *position)

        self.left_frame.setLayout(grid_layout)
        self.left_frame.setStyleSheet(styles.get('frame', ''))
        main_splitter.addWidget(self.left_frame)

        # Right panel setup
        right_frame = QFrame()
        right_vbox = QVBoxLayout()

        self.read_only_text = QTextEdit()
        self.read_only_text.setReadOnly(True)
        self.read_only_text.setStyleSheet(styles.get('text', ''))

        self.text_input = QTextEdit()
        self.text_input.setStyleSheet(styles.get('input', ''))

        right_vbox.addWidget(self.read_only_text, 4)
        right_vbox.addWidget(self.text_input, 1)

        submit_questions_button = QPushButton("Submit Questions")
        submit_questions_button.setStyleSheet(styles.get('button', ''))
        submit_questions_button.clicked.connect(self.on_submit_button_clicked)

        right_vbox.addWidget(submit_questions_button)

        # Define widget containing buttons
        button_row_widget = create_button_row(self.on_submit_button_clicked, styles.get('button', ''))

        # Add widgets from button_module.py
        right_vbox.addWidget(button_row_widget)

        right_frame.setLayout(right_vbox)
        right_frame.setStyleSheet(styles.get('frame', ''))
        main_splitter.addWidget(right_frame)

        self.metrics_bar = MetricsBar()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_splitter)
        main_layout.addWidget(self.metrics_bar)

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

    def update_response(self, response):
        self.read_only_text.setPlainText(response)

    def closeEvent(self, event):
        self.metrics_bar.stop_monitors()
        self.metrics_bar.system_monitor.stop_and_exit_system_monitor()
        self.metrics_bar.gpu_monitor.stop_and_exit_gpu_monitor()
        event.accept()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('fusion'))
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())
