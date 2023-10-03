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

styles = {
    "button": 'background-color: #323842; color: light gray; font: 10pt "Segoe UI Historic"; width: 29;',
    "frame": 'background-color: #161b22;',
    "input": 'background-color: #2e333b; color: light gray; font: 13pt "Segoe UI Historic";',
    "text": 'background-color: #092327; color: light gray; font: 12pt "Segoe UI Historic";'
}

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

tabs_config = config.get('tabs', [])

class CreateDatabaseThread(QThread):
    def run(self):
        create_database.main()

class SubmitButtonThread(QThread):
    responseSignal = Signal(str)

    def __init__(self, user_question, parent=None):
        super(SubmitButtonThread, self).__init__(parent)
        self.user_question = user_question

    def run(self):
        response = server_connector.ask_local_chatgpt(self.user_question)
        self.responseSignal.emit(response['answer'])

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
        self.setGeometry(300, 300, 850, 910)
        self.setMinimumSize(550, 610)

        self.left_frame = QFrame()  # Changed here
        left_vbox = QVBoxLayout()
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.South)

        tab_widgets = [QTextEdit(tab.get('placeholder', '')) for tab in tabs_config]
        for i, tab in enumerate(tabs_config):
            tab_widget.addTab(tab_widgets[i], tab.get('name', ''))

        tutorial_tab = QWebEngineView()
        tab_widget.addTab(tutorial_tab, 'Tutorial')
        user_manual_folder = os.path.join(os.path.dirname(__file__), 'User_Manual')
        html_file_path = os.path.join(user_manual_folder, 'number_format.html')
        tutorial_tab.setUrl(QUrl.fromLocalFile(html_file_path))
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

        right_vbox.addWidget(self.read_only_text, 5)
        right_vbox.addWidget(self.text_input, 1)
        right_vbox.addWidget(submit_button)

        right_frame.setLayout(right_vbox)
        right_frame.setStyleSheet(styles.get('frame', ''))
        main_splitter.addWidget(right_frame)

        metrics_frame = QFrame()
        metrics_frame.setFixedHeight(28)
        metrics_layout = QHBoxLayout(metrics_frame)

        self.gpu_vram_label = QLabel("VRAM: N/A")
        self.gpu_util_label = QLabel("GPU: N/A")
        self.cpu_label = QLabel("CPU: N/A")
        self.ram_label = QLabel("RAM: N/A")
        self.ram_usage_label = QLabel("RAM Usage: N/A")

        metrics_layout.addWidget(self.gpu_vram_label)
        metrics_layout.addWidget(self.gpu_util_label)
        metrics_layout.addWidget(self.cpu_label)
        metrics_layout.addWidget(self.ram_label)
        metrics_layout.addWidget(self.ram_usage_label)

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
