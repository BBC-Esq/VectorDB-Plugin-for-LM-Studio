import os
import shutil
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import (
    QWidget, QLabel, QGridLayout, QVBoxLayout, QGroupBox, QPushButton, QRadioButton, QButtonGroup
)

from constants import VECTOR_MODELS
from download_model import ModelDownloader, model_downloaded_signal
import webbrowser

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
            'BAAI': 4,
            'hkunlp': 4,
            'sentence-transformers': 6,
            'thenlper': 4,
            'intfloat': 4
        }

        models_dir = Path('Models')
        if not models_dir.exists():
            models_dir.mkdir(parents=True)

        vector_models_dir = models_dir / "Vector"

        existing_vector_directories = {d.name for d in vector_models_dir.iterdir() if d.is_dir()}

        headers = ["Select", "Model Name", "Dimensions", "Max Sequence", "Size (MB)", "Downloaded", "Link"]
        column_stretch_factors = [1, 3, 2, 2, 2, 2, 3]

        def add_centered_widget(grid, widget, row, col):
            grid.addWidget(widget, row, col, alignment=Qt.AlignCenter)

        row_counter = 1
        for vendor, models in VECTOR_MODELS.items():
            group_box = QGroupBox(vendor)
            group_layout = QGridLayout()
            group_layout.setVerticalSpacing(0)
            group_layout.setHorizontalSpacing(0)
            group_box.setLayout(group_layout)
            self.group_boxes[vendor] = group_box

            for col, header in enumerate(headers):
                header_label = QLabel(header)
                header_label.setAlignment(Qt.AlignCenter)
                header_label.setStyleSheet("text-decoration: underline;")
                group_layout.addWidget(header_label, 0, col)

            for col, stretch_factor in enumerate(column_stretch_factors):
                group_layout.setColumnStretch(col, stretch_factor)

            for model in models:
                model_name = f"{vendor}/{model['name']}"
                grid = group_box.layout()
                row = grid.rowCount()

                radiobutton = QRadioButton()
                self.model_radiobuttons.addButton(radiobutton, row_counter)
                add_centered_widget(grid, radiobutton, row, 0)

                add_centered_widget(grid, QLabel(model['name']), row, 1)
                add_centered_widget(grid, QLabel(str(model['dimensions'])), row, 2)
                add_centered_widget(grid, QLabel(str(model['max_sequence'])), row, 3)
                add_centered_widget(grid, QLabel(str(model['size_mb'])), row, 4)

                expected_dir_name = ModelDownloader(model_name, model['type']).get_model_directory_name()
                is_downloaded = expected_dir_name in existing_vector_directories
                downloaded_label = QLabel('Yes' if is_downloaded else 'No')
                add_centered_widget(grid, downloaded_label, row, 5)
                radiobutton.setEnabled(not is_downloaded)

                self.downloaded_labels[model_name] = (downloaded_label, model['type'])

                link = QLabel()
                link.setTextFormat(Qt.RichText)
                link.setText(f'<a href="https://huggingface.co/{model["repo_id"]}">Link</a>')
                link.setOpenExternalLinks(False)
                link.linkActivated.connect(self.open_link)
                add_centered_widget(grid, link, row, 6)

                row_counter += 1

        for vendor, group_box in self.group_boxes.items():
            stretch_factor = self.stretch_factors.get(vendor, 1)
            self.main_layout.addWidget(group_box, stretch_factor)

        self.download_button = QPushButton('Download Selected Model')
        self.download_button.clicked.connect(self.initiate_model_download)
        self.main_layout.addWidget(self.download_button)

        model_downloaded_signal.downloaded.connect(self.update_model_downloaded_status)

    def initiate_model_download(self):
        selected_id = self.model_radiobuttons.checkedId()
        if selected_id != -1:
            model_name, (_, model_type) = list(self.downloaded_labels.items())[selected_id - 1]
            model_downloader = ModelDownloader(model_name, model_type)

            download_thread = threading.Thread(target=lambda: model_downloader.download_model())
            download_thread.start()

    def update_model_downloaded_status(self, model_name, model_type):
        models_dir = Path('Models')
        vector_models_dir = models_dir / "Vector"

        existing_vector_directories = {d.name for d in vector_models_dir.iterdir() if d.is_dir()}

        model_directory_name = ModelDownloader(model_name, model_type).get_model_directory_name()

        if model_type == "vector" and model_directory_name in existing_vector_directories:
            downloaded_label = self.downloaded_labels.get(model_name)[0]
            if downloaded_label:
                downloaded_label.setText('Yes')

    def open_link(self, url):
        webbrowser.open(url)

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    window = VectorModelsTab()
    window.show()
    app.exec()