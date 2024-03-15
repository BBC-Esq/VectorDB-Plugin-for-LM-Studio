import subprocess
import threading
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QVBoxLayout, QGroupBox, QPushButton, QRadioButton, QButtonGroup
from pathlib import Path
from constants import AVAILABLE_MODELS

class ModelDownloadedSignal(QObject):
    downloaded = Signal(str)

model_downloaded_signal = ModelDownloadedSignal()

class VectorModelsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.available_models = AVAILABLE_MODELS
        self.group_boxes = {}
        self.downloaded_labels = {}
        self.model_radiobuttons = QButtonGroup(self)
        self.model_radiobuttons.setExclusive(True)
        self.stretch_factors = {
            'BAAI': 3,
            'hkunlp': 3,
            'jinaai': 3,
            'sentence-transformers': 8,
            'thenlper': 3,
        }

        embedding_models_dir = Path('Embedding_Models')
        if not embedding_models_dir.exists():
            embedding_models_dir.mkdir(parents=True)

        existing_directories = {d.name for d in embedding_models_dir.iterdir() if d.is_dir()}

        headers = ["Select", "Model Name", "Dimensions", "Max Sequence", "Size (MB)", "Downloaded"]

        row_counter = 1
        for model_name, details in self.available_models.items():
            vendor, model_short_name = model_name.split("/", 1)
            
            group_box = self.group_boxes.get(vendor)
            if group_box is None:
                group_box = QGroupBox(vendor)
                group_layout = QGridLayout()
                group_layout.setVerticalSpacing(0)
                group_layout.setHorizontalSpacing(0)
                group_box.setLayout(group_layout)
                self.group_boxes[vendor] = group_box

                for col, header in enumerate(headers):
                    header_label = QLabel(header)
                    header_label.setAlignment(Qt.AlignCenter)
                    group_layout.addWidget(header_label, 0, col)

            grid = group_box.layout()
            row = grid.rowCount()

            radiobutton = QRadioButton()
            self.model_radiobuttons.addButton(radiobutton, row_counter)
            grid.addWidget(radiobutton, row, 0)

            model_label = QLabel(model_short_name)
            grid.addWidget(model_label, row, 1)

            dimensions_label = QLabel(str(details['dimensions']))
            dimensions_label.setAlignment(Qt.AlignCenter)
            grid.addWidget(dimensions_label, row, 2)

            max_sequence_label = QLabel(str(details['max_sequence']))
            max_sequence_label.setAlignment(Qt.AlignCenter)
            grid.addWidget(max_sequence_label, row, 3)

            size_mb_label = QLabel(str(details['size_mb']))
            size_mb_label.setAlignment(Qt.AlignCenter)
            grid.addWidget(size_mb_label, row, 4)

            expected_dir_name = self.get_model_directory_name(model_name)
            is_downloaded = expected_dir_name in existing_directories
            downloaded_label = QLabel('Yes' if is_downloaded else 'No')
            downloaded_label.setAlignment(Qt.AlignCenter)
            grid.addWidget(downloaded_label, row, 5)
            radiobutton.setEnabled(not is_downloaded)

            self.downloaded_labels[model_name] = downloaded_label

            row_counter += 1

        for vendor, group_box in self.group_boxes.items():
            stretch_factor = self.stretch_factors.get(vendor, 1)
            self.main_layout.addWidget(group_box, stretch_factor)

        self.download_button = QPushButton('Download Selected Model')
        self.download_button.clicked.connect(self.initiate_model_download)
        self.main_layout.addWidget(self.download_button)

        model_downloaded_signal.downloaded.connect(self.update_model_downloaded_status)

    def get_model_directory_name(self, model_name):
        return model_name.replace("/", "--")

    def initiate_model_download(self):
        selected_id = self.model_radiobuttons.checkedId()
        if selected_id != -1:
            model_name = list(self.available_models.keys())[selected_id - 1]
            model_url = f"https://huggingface.co/{model_name}"
            target_directory = Path("Embedding_Models") / self.get_model_directory_name(model_name)

            def download_model():
                subprocess.run(["git", "clone", model_url, str(target_directory)], check=True)
                print(f"{model_name} has been downloaded and is ready to use!")
                model_downloaded_signal.downloaded.emit(model_name)

            download_thread = threading.Thread(target=download_model)
            download_thread.start()

    def update_model_downloaded_status(self, model_name):
        embedding_models_dir = Path('Embedding_Models')
        existing_directories = {d.name for d in embedding_models_dir.iterdir() if d.is_dir()}
        model_directory_name = self.get_model_directory_name(model_name)

        if model_directory_name in existing_directories:
            downloaded_label = self.downloaded_labels.get(model_name)
            if downloaded_label:
                downloaded_label.setText('Yes')

