from functools import partial
from PySide6.QtWidgets import (
    QLabel, QComboBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QApplication, QCheckBox, QFileDialog
)
from PySide6.QtCore import Qt
import yaml
from pathlib import Path
from transcribe_module import TranscribeFile
import threading

class TranscriberToolSettingsTab(QWidget):

    def __init__(self):
        super().__init__()
        self.selected_audio_file = None
        self.config = self.read_config()

        compute_device_config = self.config.get('Compute_Device', {}) or {}
        gpu_brand = compute_device_config.get('gpu_brand', '')
        self.gpu_brand = gpu_brand.lower() if gpu_brand is not None else ''

        self.default_device = self.config.get('transcribe_file', {}).get('device', 'cpu').lower()
        self.default_quant = self.config.get('transcribe_file', {}).get('quant', '')
        self.default_model = self.config.get('transcribe_file', {}).get('model', '')
        self.timestamps_enabled = self.config.get('transcribe_file', {}).get('timestamps', False)

        self.create_layout()

    def read_config(self):
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def create_layout(self):
        main_layout = QVBoxLayout()

        # First row of widgets
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["whisper-small.en", "whisper-medium.en", "whisper-large-v2"])
        self.model_combo.setCurrentText(self.default_model)
        self.model_combo.currentTextChanged.connect(self.update_model_in_config)
        hbox1.addWidget(self.model_combo)

        hbox1.addWidget(QLabel("Quant"))
        self.quantization_combo = QComboBox()
        hbox1.addWidget(self.quantization_combo)

        hbox1.addWidget(QLabel("Device"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("cpu")
        if self.gpu_brand == "nvidia":
            self.device_combo.addItem("cuda")
        index = self.device_combo.findText(self.default_device, Qt.MatchFixedString)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
        self.device_combo.currentTextChanged.connect(self.device_selection_changed)
        hbox1.addWidget(self.device_combo)

        main_layout.addLayout(hbox1)

        self.populate_quant_combo(self.default_device)
        self.quantization_combo.currentTextChanged.connect(self.update_quant_in_config)

        # Second row of widgets
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Timestamps"))
        self.timestamp_checkbox = QCheckBox()
        self.timestamp_checkbox.setChecked(self.timestamps_enabled)
        self.timestamp_checkbox.stateChanged.connect(self.update_timestamps_in_config)
        hbox2.addWidget(self.timestamp_checkbox)

        main_layout.addLayout(hbox2)

        # Third row of widgets
        hbox3 = QHBoxLayout()
        self.select_file_button = QPushButton("Select Audio File")
        self.select_file_button.clicked.connect(self.select_audio_file)
        hbox3.addWidget(self.select_file_button)

        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.start_transcription)
        hbox3.addWidget(self.transcribe_button)

        main_layout.addLayout(hbox3)

        self.file_path_label = QLabel("No file currently selected")
        main_layout.addWidget(self.file_path_label)

        self.setLayout(main_layout)

    def populate_quant_combo(self, device_type):
        self.quantization_combo.clear()
        quantizations = self.config.get('Supported_CTranslate2_Quantizations', {})
        device_type_key = 'GPU' if device_type == 'cuda' else 'CPU'
        if device_type_key in quantizations:
            self.quantization_combo.addItems(quantizations[device_type_key])
        self.set_default_quant()

    def set_default_quant(self):
        index = self.quantization_combo.findText(self.default_quant, Qt.MatchFixedString)
        if index >= 0:
            self.quantization_combo.setCurrentIndex(index)
        else:
            if self.quantization_combo.count() > 0:
                self.quantization_combo.setCurrentIndex(0)
        self.update_quant_in_config(self.quantization_combo.currentText())

    def update_config_file(self):
        with open('config.yaml', 'w') as file:
            yaml.dump(self.config, file)

    def device_selection_changed(self, new_device):
        self.config['transcribe_file']['device'] = new_device.lower()
        self.update_config_file()
        self.populate_quant_combo(new_device.lower())

    def update_quant_in_config(self, new_quant):
        self.config['transcribe_file']['quant'] = new_quant
        self.update_config_file()

    def update_timestamps_in_config(self):
        self.config['transcribe_file']['timestamps'] = self.timestamp_checkbox.isChecked()
        self.update_config_file()

    def update_model_in_config(self, new_model):
        self.config['transcribe_file']['model'] = new_model
        self.update_config_file()

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

        def transcription_thread():
            transcriber = TranscribeFile(self.selected_audio_file)
            transcriber.start_transcription()

        threading.Thread(target=transcription_thread, daemon=True).start()
        print(f"Transcription process for {Path(self.selected_audio_file).name} started.")

if __name__ == "__main__":
    app = QApplication([])
    window = TranscriberToolSettingsTab()
    window.show()
    app.exec()
