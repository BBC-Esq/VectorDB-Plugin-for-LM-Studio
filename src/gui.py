from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QTabWidget,
    QTextEdit, QSplitter, QFrame, QStyleFactory, QLabel, QGridLayout, QMenuBar, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
import os
import yaml
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
import sys
from utilities import list_theme_files, make_theme_changer, load_stylesheet

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.compute_device = determine_compute_device()
        os_name = get_os_name()
        self.init_ui()
        self.init_menu()

    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.setWindowTitle('LM Studio ChromaDB Plugin - www.chintellalaw.com')
        self.setGeometry(300, 300, 975, 975)
        self.setMinimumSize(450, 510)

        # Left frame setup
        self.left_frame = QFrame()
        grid_layout = QGridLayout()
        
        tab_widget = create_tabs()
        grid_layout.addWidget(tab_widget, 0, 0, 1, 2)
        
        # Button definitions and positions
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
            button.clicked.connect(handler)
            grid_layout.addWidget(button, *position)

        self.left_frame.setLayout(grid_layout)
        main_splitter.addWidget(self.left_frame)

        # Right frame setup
        right_frame = QFrame()
        right_vbox = QVBoxLayout()

        self.read_only_text = QTextEdit()
        self.read_only_text.setReadOnly(True)

        self.text_input = QTextEdit()

        right_vbox.addWidget(self.read_only_text, 4)
        right_vbox.addWidget(self.text_input, 1)

        submit_questions_button = QPushButton("Submit Questions")
        submit_questions_button.clicked.connect(self.on_submit_button_clicked)
        right_vbox.addWidget(submit_questions_button)

        # Add Test Embeddings Checkbox
        self.test_embeddings_checkbox = QCheckBox("Test Embeddings")
        self.test_embeddings_checkbox.stateChanged.connect(self.on_test_embeddings_changed)
        right_vbox.addWidget(self.test_embeddings_checkbox)

        button_row_widget = create_button_row(self.on_submit_button_clicked)
        right_vbox.addWidget(button_row_widget)

        right_frame.setLayout(right_vbox)
        main_splitter.addWidget(right_frame)

        self.metrics_bar = MetricsBar()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_splitter)
        main_layout.addWidget(self.metrics_bar)

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

    def on_submit_button_clicked(self):
        user_question = self.text_input.toPlainText()
        self.submit_button_thread = SubmitButtonThread(user_question, self)
        self.submit_button_thread.responseSignal.connect(self.update_response)
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
        self.read_only_text.setPlainText(response)

    def closeEvent(self, event):
        self.metrics_bar.stop_monitors()
        self.metrics_bar.system_monitor.stop_and_exit_system_monitor()
        self.metrics_bar.gpu_monitor.stop_and_exit_gpu_monitor()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('fusion'))
    stylesheet = load_stylesheet('custom_stylesheet.css')
    app.setStyleSheet(stylesheet)
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())
