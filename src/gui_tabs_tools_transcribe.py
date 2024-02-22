from functools import partial
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QSlider
)
from PySide6.QtCore import Qt
import yaml
from pathlib import Path
from transcribe_module import WhisperTranscriber
import threading
from utilities import my_cprint

class TranscriberToolSettingsTab(QWidget):

    def __init__(self):
        super().__init__()
        self.selected_audio_file = None
        self.config = self.read_config()

        self.gpu_brand = self.config.get('Compute_Device', {}).get('gpu_brand', '').lower()

        self.create_layout()

    def read_config(self):
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def create_layout(self):
        main_layout = QVBoxLayout()

        model_selection_hbox = QHBoxLayout()
        model_selection_hbox.addWidget(QLabel("Whisper Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v2", "medium.en", "small.en"])
        model_selection_hbox.addWidget(self.model_combo)

        model_selection_hbox.addWidget(QLabel("Speed (more memory)"))

        self.slider_label = QLabel("8")
        self.number_slider = QSlider(Qt.Horizontal)
        self.number_slider.setMinimum(1)
        self.number_slider.setMaximum(100)
        self.number_slider.setValue(8)
        self.number_slider.valueChanged.connect(self.update_slider_label)

        model_selection_hbox.addWidget(self.number_slider)
        model_selection_hbox.addWidget(self.slider_label)

        main_layout.addLayout(model_selection_hbox)

        hbox = QHBoxLayout()
        self.select_file_button = QPushButton("Select Audio File")
        self.select_file_button.clicked.connect(self.select_audio_file)
        hbox.addWidget(self.select_file_button)

        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.start_transcription)
        hbox.addWidget(self.transcribe_button)

        main_layout.addLayout(hbox)

        self.file_path_label = QLabel("No file currently selected")
        main_layout.addWidget(self.file_path_label)

        self.setLayout(main_layout)

    def update_slider_label(self, value):
        self.slider_label.setText(str(value))

    def update_config_file(self):
        with open('config.yaml', 'w') as file:
            yaml.dump(self.config, file)

    def select_audio_file(self):
        current_dir = Path(__file__).resolve().parent
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", str(current_dir))
        if file_name:
            file_path = Path(file_name)
            short_path = "..." + str(Path(file_path.parent.name) / file_path.name)
            self.file_path_label.setText(short_path)
            self.selected_audio_file = file_name

    def start_transcription(self):
        if not self.selected_audio_file:
            print("Please select an audio file.")
            return

        selected_model = self.model_combo.currentText()
        selected_batch_size = int(self.slider_label.text())

        def transcription_thread():
            my_cprint("Starting transcription process...", 'green')
            transcriber = WhisperTranscriber(model_identifier=selected_model, batch_size=selected_batch_size)
            transcriber.start_transcription_process(self.selected_audio_file)
            my_cprint("Transcription created and ready to be input into vector database.", 'green')

        threading.Thread(target=transcription_thread, daemon=True).start()