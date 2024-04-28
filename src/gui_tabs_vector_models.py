import subprocess
import shutil
import threading
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QVBoxLayout, QGroupBox, QPushButton, QRadioButton, QButtonGroup
from pathlib import Path
from constants import AVAILABLE_MODELS, CHAT_MODELS

class ModelDownloadedSignal(QObject):
    downloaded = Signal(str)

model_downloaded_signal = ModelDownloadedSignal()

class VectorModelsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.group_boxes = {}
        self.downloaded_labels = {}
        self.model_radiobuttons = QButtonGroup(self)
        self.model_radiobuttons.setExclusive(True)
        self.stretch_factors = {
            'BAAI': 2,
            'hkunlp': 2,
            'sentence-transformers': 4,
            'thenlper': 2,
        }

        embedding_models_dir = Path('Models')
        if not embedding_models_dir.exists():
            embedding_models_dir.mkdir(parents=True)

        existing_directories = {
            d.name for d in embedding_models_dir.iterdir() if d.is_dir()
        }

        headers = ["Select", "Model Name", "Dimensions", "Max Sequence", "Size (MB)", "Downloaded"]
        column_stretch_factors = [1, 3, 2, 2, 2, 2]

        row_counter = 1
        for vendor, models in AVAILABLE_MODELS.items():
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

            for col, stretch_factor in enumerate(column_stretch_factors):
                group_layout.setColumnStretch(col, stretch_factor)

            for model in models:
                model_name = f"{vendor}/{model['name']}"
                grid = group_box.layout()
                row = grid.rowCount()

                radiobutton = QRadioButton()
                self.model_radiobuttons.addButton(radiobutton, row_counter)
                grid.addWidget(radiobutton, row, 0, alignment=Qt.AlignCenter)

                model_label = QLabel(model['name'])
                grid.addWidget(model_label, row, 1, alignment=Qt.AlignCenter)

                dimensions_label = QLabel(str(model['dimensions']))
                grid.addWidget(dimensions_label, row, 2, alignment=Qt.AlignCenter)

                max_sequence_label = QLabel(str(model['max_sequence']))
                grid.addWidget(max_sequence_label, row, 3, alignment=Qt.AlignCenter)

                size_mb_label = QLabel(str(model['size_mb']))
                grid.addWidget(size_mb_label, row, 4, alignment=Qt.AlignCenter)

                expected_dir_name = self.get_model_directory_name(model_name)
                is_downloaded = expected_dir_name in existing_directories
                downloaded_label = QLabel('Yes' if is_downloaded else 'No')
                grid.addWidget(downloaded_label, row, 5, alignment=Qt.AlignCenter)
                radiobutton.setEnabled(not is_downloaded)

                self.downloaded_labels[model_name] = downloaded_label

                row_counter += 1

        # Add Chat Models group box
        chat_models_group_box = QGroupBox("Chat Models")
        chat_models_layout = QGridLayout()
        chat_models_layout.setVerticalSpacing(0)
        chat_models_layout.setHorizontalSpacing(0)
        chat_models_group_box.setLayout(chat_models_layout)

        chat_headers = ["Select", "Model Name", "VRAM", "Tokens/s", "Context Length", "Downloaded"]
        chat_column_stretch_factors = [1, 3, 2, 2, 2, 2]

        for col, header in enumerate(chat_headers):
            header_label = QLabel(header)
            header_label.setAlignment(Qt.AlignCenter)
            chat_models_layout.addWidget(header_label, 0, col)

        for col, stretch_factor in enumerate(chat_column_stretch_factors):
            chat_models_layout.setColumnStretch(col, stretch_factor)

        for model_data in CHAT_MODELS.values():
            model_name = model_data["model"]
            grid = chat_models_group_box.layout()
            row = grid.rowCount()

            radiobutton = QRadioButton()
            self.model_radiobuttons.addButton(radiobutton, row_counter)
            grid.addWidget(radiobutton, row, 0, alignment=Qt.AlignCenter)

            model_label = QLabel(model_data["model"])
            grid.addWidget(model_label, row, 1, alignment=Qt.AlignCenter)

            vram_label = QLabel(str(model_data["avg_vram_usage"]))
            grid.addWidget(vram_label, row, 2, alignment=Qt.AlignCenter)

            tokens_per_second_label = QLabel(str(model_data["tokens_per_second"]))
            grid.addWidget(tokens_per_second_label, row, 3, alignment=Qt.AlignCenter)

            context_length_label = QLabel(str(model_data["context_length"]))
            grid.addWidget(context_length_label, row, 4, alignment=Qt.AlignCenter)

            expected_dir_name = self.get_model_directory_name(model_data["repo_id"])
            is_downloaded = expected_dir_name in existing_directories
            downloaded_label = QLabel('Yes' if is_downloaded else 'No')
            grid.addWidget(downloaded_label, row, 5, alignment=Qt.AlignCenter)
            radiobutton.setEnabled(not is_downloaded)

            self.downloaded_labels[model_data["repo_id"]] = downloaded_label

            row_counter += 1

        for vendor, group_box in self.group_boxes.items():
            stretch_factor = self.stretch_factors.get(vendor, 1)
            self.main_layout.addWidget(group_box, stretch_factor)

        self.main_layout.addWidget(chat_models_group_box, 4)

        self.download_button = QPushButton('Download Selected Model')
        self.download_button.clicked.connect(self.initiate_model_download)
        self.main_layout.addWidget(self.download_button)

        model_downloaded_signal.downloaded.connect(self.update_model_downloaded_status)

    def get_model_directory_name(self, model_name):
        return model_name.replace("/", "--")

    def initiate_model_download(self):
        selected_id = self.model_radiobuttons.checkedId()
        if selected_id != -1:
            model_name = list(self.downloaded_labels.keys())[selected_id - 1]
            model_url = f"https://huggingface.co/{model_name}"
            target_directory = Path("Models") / self.get_model_directory_name(model_name)

            def download_model():
                subprocess.run(["git", "clone", "--depth", "1", model_url, str(target_directory)], check=True)
                print(f"{model_name} has been downloaded and is ready to use!")
                model_downloaded_signal.downloaded.emit(model_name)

            download_thread = threading.Thread(target=download_model)
            download_thread.start()

    def update_model_downloaded_status(self, model_name):
        models_dir = Path('Models')
        existing_directories = {d.name for d in models_dir.iterdir() if d.is_dir()}
        model_directory_name = self.get_model_directory_name(model_name)

        if model_directory_name in existing_directories:
            downloaded_label = self.downloaded_labels.get(model_name)
            if downloaded_label:
                downloaded_label.setText('Yes')