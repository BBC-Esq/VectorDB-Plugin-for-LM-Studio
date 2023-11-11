from PySide6.QtWidgets import (
    QLabel, QComboBox, QWidget, QGridLayout, QPushButton, QFileDialog, QCheckBox, QApplication
)
from PySide6.QtCore import QThread, Qt
import ctranslate2
import yaml
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
        layout = QGridLayout()

        def add_widget(widget_class, text, row, column, colspan=1, signal_slot=None, items=None):
            widget = widget_class()
            if widget_class in [QLabel, QPushButton]:
                widget.setText(text)
            if signal_slot:
                widget.clicked.connect(signal_slot)
            if widget_class is QComboBox and items:
                widget.addItems(items)
            layout.addWidget(widget, row, column, 1, colspan)
            return widget

        # Model
        add_widget(QLabel, "Model", 0, 0)
        self.model_combo = add_widget(QComboBox, None, 0, 1, items=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2"])

        # Quantization
        add_widget(QLabel, "Quant", 0, 2)
        self.quantization_combo = add_widget(QComboBox, None, 0, 3)

        # Device
        add_widget(QLabel, "Device", 0, 4)
        device_options = ["cpu"] + ["cuda"] if self.has_cuda_device() else []
        self.device_combo = add_widget(QComboBox, None, 0, 5, items=device_options)

        # Timestamp and Translate Labels with Checkboxes
        add_widget(QLabel, "Timestamps", 1, 0)
        self.timestamp_checkbox = add_widget(QCheckBox, None, 1, 1)
        add_widget(QLabel, "Translate", 1, 2)
        self.translate_checkbox = add_widget(QCheckBox, None, 1, 3)

        # Language Label and ComboBox
        add_widget(QLabel, "Language", 1, 4)
        self.language_combo = add_widget(QComboBox, None, 1, 5, items=["Option 1", "Option 2", "Option 3"])

        # Select Audio File Button
        self.select_file_button = add_widget(QPushButton, "Select Audio File", 2, 0, 3, signal_slot=self.select_audio_file)

        # Transcribe/Translate Button
        self.transcribe_translate_button = add_widget(QPushButton, "Transcribe/Translate", 2, 3, 3, signal_slot=self.start_transcription)

        # Update Settings Button
        self.update_settings_button = add_widget(QPushButton, "Update Settings", 3, 0, 6, signal_slot=self.save_settings)

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
            transcriber = TranscribeFile()
            self.transcription_thread = TranscriptionThread(transcriber)
            self.transcription_thread.start()
        else:
            print("Please select an audio file first.")

# Only used if ran standalone
if __name__ == "__main__":
    app = QApplication([])
    window = TranscriberToolSettingsTab()
    window.show()
    app.exec()
