from PySide6.QtWidgets import QLabel, QComboBox, QWidget, QHBoxLayout, QVBoxLayout, QCheckBox
import yaml
from pathlib import Path

class BarkModelSettingsTab(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initialize_layout()
        self.load_config_and_set_values()
        self.connect_signals()

    def initialize_layout(self):
        main_layout = QVBoxLayout()

        # First row
        first_row_layout = QHBoxLayout()
        self.model_size_label = QLabel("Model")
        first_row_layout.addWidget(self.model_size_label)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["normal", "small"])
        first_row_layout.addWidget(self.model_size_combo)

        self.quant_label = QLabel("Quant")
        first_row_layout.addWidget(self.quant_label)
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["float32", "float16"])
        first_row_layout.addWidget(self.quant_combo)

        self.speaker_label = QLabel("Speaker")
        first_row_layout.addWidget(self.speaker_label)
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
                                     "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
                                     "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
                                     "v2/en_speaker_9"])
        first_row_layout.addWidget(self.speaker_combo)

        main_layout.addLayout(first_row_layout)

        # Second row
        second_row_layout = QHBoxLayout()
        self.better_transformer_label = QLabel("Better Transformer")
        second_row_layout.addWidget(self.better_transformer_label)
        self.better_transformer_checkbox = QCheckBox()
        second_row_layout.addWidget(self.better_transformer_checkbox)

        self.cpu_offload_label = QLabel("CPU Offload")
        second_row_layout.addWidget(self.cpu_offload_label)
        self.cpu_offload_checkbox = QCheckBox()
        second_row_layout.addWidget(self.cpu_offload_checkbox)

        main_layout.addLayout(second_row_layout)

        self.setLayout(main_layout)

    def load_config_and_set_values(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                config = None
        else:
            config = None

        bark_config = config.get('bark', {}) if config else {}

        self.model_size_combo.setCurrentText(bark_config.get('size', "small"))
        self.quant_combo.setCurrentText(bark_config.get('model_precision', "float16"))
        self.speaker_combo.setCurrentText(bark_config.get('speaker', "v2/en_speaker_6"))
        self.better_transformer_checkbox.setChecked(bark_config.get('use_better_transformer', True))
        self.cpu_offload_checkbox.setChecked(bark_config.get('enable_cpu_offload', False))

    def connect_signals(self):
        self.model_size_combo.currentTextChanged.connect(self.update_config)
        self.quant_combo.currentTextChanged.connect(self.update_config)
        self.speaker_combo.currentTextChanged.connect(self.update_config)
        self.better_transformer_checkbox.stateChanged.connect(self.update_config)
        self.cpu_offload_checkbox.stateChanged.connect(self.update_config)

    def update_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                config = {}
        else:
            config = {}

        # Update only the 'bark' section of the config
        bark_config = config.get('bark', {})
        bark_config['size'] = self.model_size_combo.currentText()
        bark_config['model_precision'] = self.quant_combo.currentText()
        bark_config['speaker'] = self.speaker_combo.currentText()
        bark_config['use_better_transformer'] = self.better_transformer_checkbox.isChecked()
        bark_config['enable_cpu_offload'] = self.cpu_offload_checkbox.isChecked()
        config['bark'] = bark_config

        with open(config_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)