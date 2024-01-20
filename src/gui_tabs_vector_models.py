from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QVBoxLayout, QGroupBox, QSizePolicy, QPushButton
from pathlib import Path
from constants import AVAILABLE_MODELS
from download_model import download_embedding_model, model_downloaded_signal

class VectorModelsTab(QWidget):
    def __init__(self, parent=None):
        super(VectorModelsTab, self).__init__(parent)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.available_models = AVAILABLE_MODELS
        self.group_boxes = {}
        self.downloaded_labels = {}
        self.stretch_factors = {
            'BAAI': 4,
            'hkunlp': 4,
            'jinaai': 5,
            'sentence-transformers': 14,
            'thenlper': 4,
        }

        def get_model_directory_name(model_name):
            return model_name.replace("/", "--")

        embedding_models_dir = Path('Embedding_Models')
        if not embedding_models_dir.exists():
            embedding_models_dir.mkdir(parents=True)

        existing_directories = set([d.name for d in embedding_models_dir.iterdir() if d.is_dir()])

        headers = ["Model Name", "Dimensions", "Max Sequence", "Size (MB)", "Downloaded"]

        for model_entry in self.available_models:
            model_name = model_entry['model']
            vendor, model_short_name = model_name.split("/", 1)
            
            group_box = self.group_boxes.get(vendor)
            if group_box is None:
                group_box = QGroupBox(vendor)
                group_layout = QGridLayout()
                group_layout.setVerticalSpacing(0)
                group_layout.setHorizontalSpacing(0)
                group_box.setLayout(group_layout)
                self.group_boxes[vendor] = group_box

                group_box.setMinimumSize(0, 0)
                group_box.setSizePolicy(QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored))

                for col, header in enumerate(headers):
                    header_label = QLabel(header)
                    header_label.setAlignment(Qt.AlignCenter)
                    header_label.setMinimumSize(0, 0)
                    group_layout.addWidget(header_label, 0, col)

            grid = group_box.layout()
            row = grid.rowCount()

            expected_dir_name = get_model_directory_name(model_name)
            is_downloaded = expected_dir_name in existing_directories

            model_label = QLabel(model_short_name)
            model_label.setMinimumSize(0, 0)
            grid.addWidget(model_label, row, 0)

            dimensions_label = QLabel(str(model_entry['details']['dimensions']))
            dimensions_label.setAlignment(Qt.AlignCenter)
            dimensions_label.setMinimumSize(0, 0)
            grid.addWidget(dimensions_label, row, 1)

            max_sequence_label = QLabel(str(model_entry['details']['max_sequence']))
            max_sequence_label.setAlignment(Qt.AlignCenter)
            max_sequence_label.setMinimumSize(0, 0)
            grid.addWidget(max_sequence_label, row, 2)

            size_mb_label = QLabel(str(model_entry['details']['size_mb']))
            size_mb_label.setAlignment(Qt.AlignCenter)
            size_mb_label.setMinimumSize(0, 0)
            grid.addWidget(size_mb_label, row, 3)

            downloaded_label = QLabel('Yes' if is_downloaded else 'No')
            downloaded_label.setAlignment(Qt.AlignCenter)
            downloaded_label.setMinimumSize(0, 0)
            grid.addWidget(downloaded_label, row, 4)

            self.downloaded_labels[model_name] = downloaded_label

        for vendor, group_box in self.group_boxes.items():
            stretch_factor = self.stretch_factors.get(vendor, 1)
            self.main_layout.addWidget(group_box, stretch_factor)

        self.download_button = QPushButton('Download Embedding Model')
        self.download_button.clicked.connect(lambda: download_embedding_model(self))
        self.main_layout.addWidget(self.download_button)

        model_downloaded_signal.downloaded.connect(self.update_model_downloaded_status)

        self.main_layout.activate()
        self.update()

    def update_model_downloaded_status(self, model_name):
        embedding_models_dir = Path('Embedding_Models')
        existing_directories = set([d.name for d in embedding_models_dir.iterdir() if d.is_dir()])
        model_directory_name = model_name.replace("/", "--")

        if model_directory_name in existing_directories:
            downloaded_label = self.downloaded_labels.get(model_name)
            if downloaded_label:
                downloaded_label.setText('Yes')
