from PySide6.QtWidgets import (
    QLabel, QComboBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QApplication
)
from PySide6.QtCore import Qt
import ctranslate2
import yaml
from transcribe_module import TranscribeFile

class TranscriberToolSettingsTab(QWidget):
    
    def __init__(self):
        super().__init__()
        self.selected_audio_file = None
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
        main_layout = QVBoxLayout()

        # First horizontal layout
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2"])
        hbox1.addWidget(self.model_combo)

        hbox1.addWidget(QLabel("Quant"))
        self.quantization_combo = QComboBox()
        hbox1.addWidget(self.quantization_combo)

        hbox1.addWidget(QLabel("Device"))
        device_options = ["cpu"] + ["cuda"] if self.has_cuda_device() else []
        self.device_combo = QComboBox()
        self.device_combo.addItems(device_options)
        self.device_combo.currentTextChanged.connect(lambda: self.update_quantization(self.device_combo, self.quantization_combo))
        self.update_quantization(self.device_combo, self.quantization_combo)
        hbox1.addWidget(self.device_combo)

        main_layout.addLayout(hbox1)

        # Second horizontal layout
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Timestamps"))
        self.timestamp_checkbox = QCheckBox()
        hbox2.addWidget(self.timestamp_checkbox)

        hbox2.addWidget(QLabel("Translate"))
        self.translate_checkbox = QCheckBox()
        hbox2.addWidget(self.translate_checkbox)

        hbox2.addWidget(QLabel("Language"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Option 1", "Option 2", "Option 3"])
        hbox2.addWidget(self.language_combo)

        main_layout.addLayout(hbox2)

        # Third horizontal layout
        hbox3 = QHBoxLayout()
        self.select_file_button = QPushButton("Select Audio File")
        self.select_file_button.clicked.connect(self.select_audio_file)
        hbox3.addWidget(self.select_file_button)

        self.transcribe_translate_button = QPushButton("Transcribe/Translate")
        self.transcribe_translate_button.clicked.connect(self.start_transcription)
        hbox3.addWidget(self.transcribe_translate_button)

        main_layout.addLayout(hbox3)

        # Update settings button (without a horizontal layout)
        self.update_settings_button = QPushButton("Update Settings")
        self.update_settings_button.clicked.connect(self.save_settings)
        main_layout.addWidget(self.update_settings_button)

        self.setLayout(main_layout)
    
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
            transcriber = TranscribeFile(self.selected_audio_file)
            transcriber.start_transcription_thread()
        else:
            print("Please select an audio file first.")

if __name__ == "__main__":
    app = QApplication([])
    window = TranscriberToolSettingsTab()
    window.show()
    app.exec()
