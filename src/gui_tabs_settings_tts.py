import yaml
from pathlib import Path

from PySide6.QtWidgets import QLabel, QComboBox, QWidget, QGridLayout, QRadioButton, QButtonGroup, QMessageBox

from constants import WHISPER_SPEECH_MODELS

class BarkModelSettingsTab(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initialize_layout()
        self.load_config_and_set_values()
        self.connect_signals()
        self.update_widgets_state()

    def initialize_layout(self):
        main_layout = QGridLayout()

        self.radio_button_group = QButtonGroup(self)

        self.use_bark_radio = QRadioButton("Bark (GPU)")
        self.radio_button_group.addButton(self.use_bark_radio)

        self.use_whisper_radio = QRadioButton("WhisperSpeech (GPU)")
        self.use_whisper_radio.setChecked(True)
        self.radio_button_group.addButton(self.use_whisper_radio)

        # self.use_chattts_radio = QRadioButton("ChatTTS - (CPU/CPU)")
        # self.radio_button_group.addButton(self.use_chattts_radio)

        self.use_googletts_radio = QRadioButton("Google TTS - (CPU)")
        self.radio_button_group.addButton(self.use_googletts_radio)

        main_layout.addWidget(self.use_bark_radio, 0, 0, 1, 2)

        self.model_size_label = QLabel("Model")
        main_layout.addWidget(self.model_size_label, 0, 2)

        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["normal", "small"])
        self.model_size_combo.setMinimumWidth(100)
        main_layout.addWidget(self.model_size_combo, 0, 3)

        self.speaker_label = QLabel("Speaker")
        main_layout.addWidget(self.speaker_label, 0, 4)

        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems([
            "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
            "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
            "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
            "v2/en_speaker_9"])
        self.speaker_combo.setMinimumWidth(100)
        main_layout.addWidget(self.speaker_combo, 0, 5)

        main_layout.addWidget(self.use_whisper_radio, 1, 0, 1, 2)
        
        self.s2a_label = QLabel("S2A Model:")
        main_layout.addWidget(self.s2a_label, 1, 2)

        self.s2a_combo = QComboBox()
        self.s2a_combo.addItems(list(WHISPER_SPEECH_MODELS["s2a"].keys()))
        self.s2a_combo.setCurrentText(list(WHISPER_SPEECH_MODELS["s2a"].keys())[0])  # Set default
        self.s2a_combo.setMinimumWidth(100)
        main_layout.addWidget(self.s2a_combo, 1, 3)

        self.t2s_label = QLabel("T2S Model:")
        main_layout.addWidget(self.t2s_label, 1, 4)

        self.t2s_combo = QComboBox()
        self.t2s_combo.addItems(list(WHISPER_SPEECH_MODELS["t2s"].keys()))
        self.t2s_combo.setCurrentText(list(WHISPER_SPEECH_MODELS["t2s"].keys())[0])  # Set default
        self.t2s_combo.setMinimumWidth(100)
        main_layout.addWidget(self.t2s_combo, 1, 5)

        # main_layout.addWidget(self.use_chattts_radio, 2, 0, 1, 2)
        main_layout.addWidget(self.use_googletts_radio, 3, 0, 1, 2)

        self.setLayout(main_layout)

    def load_config_and_set_values(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Configuration Error",
                    f"Error loading configuration file: {e}"
                )
                config = None
        else:
            config = None

        bark_config = config.get('bark', {}) if config else {}

        self.model_size_combo.setCurrentText(bark_config.get('size', "small"))
        self.speaker_combo.setCurrentText(bark_config.get('speaker', "v2/en_speaker_6"))

        tts_config = config.get('tts', {}) if config else {}
        tts_model = tts_config.get('model', 'whisperspeech')
        if tts_model == 'bark':
            self.use_bark_radio.setChecked(True)
        # elif tts_model == 'chattts':
            # self.use_chattts_radio.setChecked(True)
        elif tts_model == 'googletts':
            self.use_googletts_radio.setChecked(True)
        else:
            self.use_whisper_radio.setChecked(True)

        s2a_model = tts_config.get('s2a', "s2a-q4-tiny-en+pl.model")
        t2s_model = tts_config.get('t2s', "t2s-tiny-en+pl.model")

        self.s2a_combo.setCurrentText(
            next(
                (k for k, v in WHISPER_SPEECH_MODELS["s2a"].items() if v == s2a_model),
                list(WHISPER_SPEECH_MODELS["s2a"].keys())[0]
            )
        )

        self.t2s_combo.setCurrentText(
            next(
                (k for k, v in WHISPER_SPEECH_MODELS["t2s"].items() if v == t2s_model),
                list(WHISPER_SPEECH_MODELS["t2s"].keys())[0]
            )
        )

        self.update_whisperspeech_config()

    def connect_signals(self):
        self.model_size_combo.currentTextChanged.connect(self.update_config)
        self.speaker_combo.currentTextChanged.connect(self.update_config)
        self.use_bark_radio.toggled.connect(self.update_widgets_state)
        self.use_whisper_radio.toggled.connect(self.update_widgets_state)
        # self.use_chattts_radio.toggled.connect(self.update_widgets_state)
        self.use_googletts_radio.toggled.connect(self.update_widgets_state)
        self.use_bark_radio.toggled.connect(self.update_tts_model)
        self.use_whisper_radio.toggled.connect(self.update_tts_model)
        # self.use_chattts_radio.toggled.connect(self.update_tts_model)
        self.use_googletts_radio.toggled.connect(self.update_tts_model)
        self.s2a_combo.currentTextChanged.connect(self.update_whisperspeech_config)
        self.t2s_combo.currentTextChanged.connect(self.update_whisperspeech_config)

    def update_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Configuration Error",
                    f"Error updating configuration: {e}"
                )
                config = {}
        else:
            config = {}

        bark_config = config.get('bark', {})
        bark_config['size'] = self.model_size_combo.currentText()
        bark_config['speaker'] = self.speaker_combo.currentText()
        config['bark'] = bark_config

        with open(config_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def update_widgets_state(self):
        bark_enabled = self.use_bark_radio.isChecked()
        whisper_enabled = self.use_whisper_radio.isChecked()
        self.model_size_label.setEnabled(bark_enabled)
        self.model_size_combo.setEnabled(bark_enabled)
        self.speaker_label.setEnabled(bark_enabled)
        self.speaker_combo.setEnabled(bark_enabled)
        self.s2a_label.setEnabled(whisper_enabled)
        self.s2a_combo.setEnabled(whisper_enabled)
        self.t2s_label.setEnabled(whisper_enabled)
        self.t2s_combo.setEnabled(whisper_enabled)

    def update_tts_model(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Configuration Error",
                    f"Error updating TTS model settings: {e}"
                )
                config = {}
        else:
            config = {}

        tts_config = config.get('tts', {})
        if self.use_bark_radio.isChecked():
            tts_config['model'] = 'bark'
        elif self.use_chattts_radio.isChecked():
            tts_config['model'] = 'chattts'
        elif self.use_googletts_radio.isChecked():
            tts_config['model'] = 'googletts'
        else:
            tts_config['model'] = 'whisperspeech'
        config['tts'] = tts_config

        with open(config_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def update_whisperspeech_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Configuration Error",
                    f"Error updating WhisperSpeech settings: {e}"
                )
                config = {}
        else:
            config = {}

        tts_config = config.get('tts', {})
        tts_config['model'] = 'whisperspeech'
        tts_config['s2a'] = WHISPER_SPEECH_MODELS["s2a"][self.s2a_combo.currentText()][0]
        tts_config['t2s'] = WHISPER_SPEECH_MODELS["t2s"][self.t2s_combo.currentText()][0]
        config['tts'] = tts_config

        with open(config_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)