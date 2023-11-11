from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QButtonGroup, QLabel, QGridLayout
import os
import yaml
import subprocess
import threading

class DownloadModelDialog(QDialog):
    def __init__(self, parent=None):
        super(DownloadModelDialog, self).__init__(parent)
        self.setWindowTitle('Download Model')
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        try:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                self.available_models = config.get('AVAILABLE_MODELS', [])
        except Exception as e:
            print(f"Error loading config.yaml: {e}")
            self.available_models = []

        self.button_group = QButtonGroup(self)
        self.selected_model = None

        # Adding column headers
        model_name_header = QLabel("Model Name")
        self.grid_layout.addWidget(model_name_header, 0, 1)

        description_header = QLabel("Description")
        self.grid_layout.addWidget(description_header, 0, 2)

        dimensions_header = QLabel("Dimensions")
        self.grid_layout.addWidget(dimensions_header, 0, 3)

        max_sequence_header = QLabel("Max Sequence")
        self.grid_layout.addWidget(max_sequence_header, 0, 4)

        size_mb_header = QLabel("Size (MB)")
        self.grid_layout.addWidget(size_mb_header, 0, 5)

        downloaded_header = QLabel("Downloaded")
        self.grid_layout.addWidget(downloaded_header, 0, 6)

        def get_model_directory_name(model_name):
            return model_name.replace("/", "--")

        if not os.path.exists('Embedding_Models'):
            os.makedirs('Embedding_Models')
        existing_directories = set(os.listdir('Embedding_Models'))

        for row, model_entry in enumerate(self.available_models, start=1):
            model_name = model_entry['model']
            expected_dir_name = get_model_directory_name(model_name)
            is_downloaded = expected_dir_name in existing_directories

            radiobutton = QRadioButton()
            self.grid_layout.addWidget(radiobutton, row, 0)

            model_label = QLabel(model_name)
            self.grid_layout.addWidget(model_label, row, 1)

            description_label = QLabel(model_entry['details']['description'])
            self.grid_layout.addWidget(description_label, row, 2)

            dimensions_label = QLabel(str(model_entry['details']['dimensions']))
            self.grid_layout.addWidget(dimensions_label, row, 3)

            max_sequence_label = QLabel(str(model_entry['details']['max_sequence']))
            self.grid_layout.addWidget(max_sequence_label, row, 4)

            size_mb_label = QLabel(str(model_entry['details']['size_mb']))
            self.grid_layout.addWidget(size_mb_label, row, 5)

            downloaded_label = QLabel('Yes' if is_downloaded else 'No')
            self.grid_layout.addWidget(downloaded_label, row, 6)

            self.button_group.addButton(radiobutton)

        button_layout = QHBoxLayout()
        download_button = QPushButton('Download', self)
        download_button.clicked.connect(self.accept)
        button_layout.addWidget(download_button)

        exit_button = QPushButton('Exit', self)
        exit_button.clicked.connect(self.reject)
        button_layout.addWidget(exit_button)

        self.grid_layout.addLayout(button_layout, row + 1, 0, 1, 7)

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
            target_directory = os.path.join("Embedding_Models", selected_model['model'].replace("/", "--"))

            def download_model():
                subprocess.run(["git", "clone", model_url, target_directory])
                print(f"{selected_model['model']} has been downloaded and is ready to use!")

            download_thread = threading.Thread(target=download_model)
            download_thread.start()
