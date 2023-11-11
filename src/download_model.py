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
        self.grid_layout.addWidget(model_name_header, 0, 1)  # Adding header in the second column

        download_status_header = QLabel("Downloaded")
        download_status_header.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(download_status_header, 0, 2)  # Adding header in the third column

        def get_model_directory_name(model_name):
            return model_name.replace("/", "--")

        if not os.path.exists('Embedding_Models'):
            os.makedirs('Embedding_Models')
        existing_directories = set(os.listdir('Embedding_Models'))

        for row, model in enumerate(self.available_models, start=1):
            expected_dir_name = get_model_directory_name(model)
            is_downloaded = expected_dir_name in existing_directories

            radiobutton = QRadioButton()
            self.grid_layout.addWidget(radiobutton, row, 0)  # radio button in the first column

            model_label = QLabel(model)
            self.grid_layout.addWidget(model_label, row, 1)  # model name in the second column

            status_label = QLabel('Yes' if is_downloaded else 'No')
            status_label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(status_label, row, 2)  # status label in the third column

            self.button_group.addButton(radiobutton)

        button_layout = QHBoxLayout()
        download_button = QPushButton('Download', self)
        download_button.clicked.connect(self.accept)
        button_layout.addWidget(download_button)

        exit_button = QPushButton('Exit', self)
        exit_button.clicked.connect(self.reject)
        button_layout.addWidget(exit_button)

        self.grid_layout.addLayout(button_layout, row + 1, 0, 1, 3)  # Span the button layout across three columns

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
            model_url = f"https://huggingface.co/{selected_model}"
            target_directory = os.path.join("Embedding_Models", selected_model.replace("/", "--"))

            def download_model():
                subprocess.run(["git", "clone", model_url, target_directory])
                print(f"{selected_model} has been downloaded and is ready to use!")

            download_thread = threading.Thread(target=download_model)
            download_thread.start()
