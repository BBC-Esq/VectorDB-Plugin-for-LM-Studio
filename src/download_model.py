from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QButtonGroup, QLabel, QGridLayout
import subprocess
import threading
from pathlib import Path
from constants import AVAILABLE_MODELS

class ModelDownloadedSignal(QObject):
    downloaded = Signal(str)

model_downloaded_signal = ModelDownloadedSignal()

class DownloadModelDialog(QDialog):
    def __init__(self, parent=None):
        super(DownloadModelDialog, self).__init__(parent)
        self.setWindowTitle('Download Model')
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        self.available_models = AVAILABLE_MODELS
        self.button_group = QButtonGroup(self)
        self.selected_model = None

        # Add column headers
        model_name_header = QLabel("Model Name")
        self.grid_layout.addWidget(model_name_header, 0, 1)

        description_header = QLabel("Description")
        self.grid_layout.addWidget(description_header, 0, 2)

        def get_model_directory_name(model_name):
            return model_name.replace("/", "--")

        embedding_models_dir = Path('Embedding_Models')
        if not embedding_models_dir.exists():
            embedding_models_dir.mkdir(parents=True)

        existing_directories = set([d.name for d in embedding_models_dir.iterdir() if d.is_dir()])

        for row, model_entry in enumerate(self.available_models, start=1):
            model_name = model_entry['model']
            expected_dir_name = get_model_directory_name(model_name)

            radiobutton = QRadioButton()
            self.grid_layout.addWidget(radiobutton, row, 0)

            model_label = QLabel(model_name)
            self.grid_layout.addWidget(model_label, row, 1)

            description_label = QLabel(model_entry['details']['description'])
            self.grid_layout.addWidget(description_label, row, 2)

            self.button_group.addButton(radiobutton)

        button_layout = QHBoxLayout()
        download_button = QPushButton('Download', self)
        download_button.clicked.connect(self.accept)
        button_layout.addWidget(download_button)

        exit_button = QPushButton('Cancel', self)
        exit_button.clicked.connect(self.reject)
        button_layout.addWidget(exit_button)

        self.grid_layout.addLayout(button_layout, row + 1, 0, 1, 3)

    def accept(self):
        for button in self.button_group.buttons():
            if button.isChecked():
                index = self.button_group.buttons().index(button)
                self.selected_model = self.available_models[index]
                break
        if self.selected_model:
            super().accept()

def download_embedding_model(parent):
    dialog = DownloadModelDialog(parent)
    if dialog.exec_():
        selected_model = dialog.selected_model

        if selected_model:
            model_url = f"https://huggingface.co/{selected_model['model']}"
            target_directory = Path("Embedding_Models") / selected_model['model'].replace("/", "--")

            def download_model():
                subprocess.run(["git", "clone", model_url, str(target_directory)])
                print(f"{selected_model['model']} has been downloaded and is ready to use!")
                model_downloaded_signal.downloaded.emit(selected_model['model'])

            download_thread = threading.Thread(target=download_model)
            download_thread.start()
