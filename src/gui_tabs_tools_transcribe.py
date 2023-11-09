from PySide6.QtWidgets import (
    QLabel, QComboBox, QWidget, QGridLayout, QPushButton, QFileDialog, QCheckBox, QApplication
)
from PySide6.QtCore import QThread, Qt
import ctranslate2
import yaml
# Import the TranscribeFile class from the transcribe_module.py script
from transcribe_module import TranscribeFile

class TranscriptionThread(QThread):
    def __init__(self, transcriber):
        super().__init__()
        self.transcriber = transcriber

    def run(self):
        try:
            # Run the transcription process
            self.transcriber.transcribe_to_file()
            print("Transcription completed and saved in 'Docs_for_DB' directory.")
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
        except Exception as e:
            print(f"An error occurred during transcription: {e}")

class TranscriberToolSettingsTab(QWidget):
    
    def __init__(self):
        super().__init__()
        self.selected_audio_file = None  # Keep track of the selected audio file
        self.create_layout()
        self.load_config()

    def load_config(self):
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
            transcriber_data = config_data.get('transcribe_file', {})

        initial_device_value = transcriber_data.get('device', None)
        if initial_device_value is not None:
            self.device_combo.setCurrentText(str(initial_device_value))

        self.update_quantization(self.device_combo, self.quantization_combo)
        initial_quant_value = transcriber_data.get('quant', None)
        if initial_quant_value is not None:
            self.quantization_combo.setCurrentText(str(initial_quant_value))

        update_map = {
            'model': (self.model_combo, 'setCurrentText', str)
        }
        for setting, (widget, setter, caster) in update_map.items():
            initial_value = transcriber_data.get(setting, None)
            if initial_value is not None:
                getattr(widget, setter)(caster(initial_value))

    def has_cuda_device(self):
        cuda_device_count = ctranslate2.get_cuda_device_count()
        return cuda_device_count > 0

    def get_supported_quantizations(self, device_type):
        types = ctranslate2.get_supported_compute_types(device_type)
        return [q for q in types if q != 'int16']

    def update_quantization(self, device_combo, quantization_combo):
        if device_combo.currentText() == "cpu":
            quantizations = self.get_supported_quantizations("cpu")
        else:
            quantizations = self.get_supported_quantizations("cuda")
        quantization_combo.clear()
        quantization_combo.addItems(quantizations)

    def create_layout(self):
        layout = QGridLayout()

        # Model
        model_label = QLabel("Model")
        layout.addWidget(model_label, 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2"])
        layout.addWidget(self.model_combo, 0, 1)

        # Quantization
        quantization_label = QLabel("Quant")
        layout.addWidget(quantization_label, 0, 2)
        self.quantization_combo = QComboBox()
        layout.addWidget(self.quantization_combo, 0, 3)

        # Device
        device_label = QLabel("Device")
        layout.addWidget(device_label, 0, 4)
        self.device_combo = QComboBox()
        device_options = ["cpu"]
        if self.has_cuda_device():
            device_options.append("cuda")
        self.device_combo.addItems(device_options)
        layout.addWidget(self.device_combo, 0, 5)

        # Timestamp and Translate Labels with Checkboxes
        timestamp_label = QLabel("Timestamps")
        layout.addWidget(timestamp_label, 1, 0)
        self.timestamp_checkbox = QCheckBox()
        layout.addWidget(self.timestamp_checkbox, 1, 1)
        translate_label = QLabel("Translate")
        layout.addWidget(translate_label, 1, 2)
        self.translate_checkbox = QCheckBox()
        layout.addWidget(self.translate_checkbox, 1, 3)

        # Language Label and ComboBox
        language_label = QLabel("Language")
        layout.addWidget(language_label, 1, 4)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Option 1", "Option 2", "Option 3"])
        layout.addWidget(self.language_combo, 1, 5)

        # Select Audio File Button
        self.select_file_button = QPushButton("Select Audio File")
        self.select_file_button.clicked.connect(self.select_audio_file)
        layout.addWidget(self.select_file_button, 2, 0, 1, 3)

        # Transcribe/Translate Button (with functionality)
        self.transcribe_translate_button = QPushButton("Transcribe/Translate")
        self.transcribe_translate_button.clicked.connect(self.start_transcription)
        layout.addWidget(self.transcribe_translate_button, 2, 3, 1, 3)

        # Update Settings Button
        self.update_settings_button = QPushButton("Update Settings")
        self.update_settings_button.clicked.connect(self.save_settings)
        layout.addWidget(self.update_settings_button, 3, 0, 1, 6)

        self.device_combo.currentTextChanged.connect(lambda: self.update_quantization(self.device_combo, self.quantization_combo))
        self.update_quantization(self.device_combo, self.quantization_combo)

        self.setLayout(layout)
    
    def select_audio_file(self):
        audio_file_filter = "Audio Files (*.mp3 *.wav *.flac *.mp4 *.wma *.mpeg *.mpga *.m4a *.webm *.ogg *.oga *.)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", audio_file_filter)
        if file_name:
            self.selected_audio_file = file_name
            self.update_config(file_to_transcribe=file_name)
    
    def save_settings(self):
        settings = {
            "model": self.model_combo.currentText(),
            "quant": self.quantization_combo.currentText(),
            "device": self.device_combo.currentText(),
            "timestamps": self.timestamp_checkbox.isChecked(),
            "translate": self.translate_checkbox.isChecked(),
            "language": self.language_combo.currentText()
        }
        self.update_config(**settings)

    def update_config(self, file_to_transcribe=None, **kwargs):
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
            transcribe_data = config_data.get('transcribe_file', {})

        if file_to_transcribe:
            transcribe_data['file'] = file_to_transcribe
        for key, value in kwargs.items():
            transcribe_data[key] = value

        config_data['transcribe_file'] = transcribe_data

        with open('config.yaml', 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def start_transcription(self):
        if self.selected_audio_file:
            # Create an instance of the TranscribeFile class
            transcriber = TranscribeFile()
            # Create a QThread object and move the transcriber to it
            self.transcription_thread = TranscriptionThread(transcriber)
            # Start the thread which will call the run method
            self.transcription_thread.start()
        else:
            print("Please select an audio file first.")

# Only used if ran standalone
if __name__ == "__main__":
    app = QApplication([])
    window = TranscriberToolSettingsTab()
    window.show()
    app.exec()
