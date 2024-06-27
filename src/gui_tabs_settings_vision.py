import yaml
from pathlib import Path
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QGridLayout, QVBoxLayout, QComboBox, QWidget
from constants import VISION_MODELS

def is_cuda_available():
    return torch.cuda.is_available()

class VisionSettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)
        gridLayout = QGridLayout()
        mainVLayout.addLayout(gridLayout)

        label_model = QLabel("Model")
        gridLayout.addWidget(label_model, 0, 1)
        gridLayout.setAlignment(label_model, Qt.AlignCenter)

        label_size = QLabel("Size")
        gridLayout.addWidget(label_size, 0, 3)
        gridLayout.setAlignment(label_size, Qt.AlignCenter)

        label_quant = QLabel("Quant")
        gridLayout.addWidget(label_quant, 0, 5)
        gridLayout.setAlignment(label_quant, Qt.AlignCenter)

        self.modelComboBox = QComboBox()
        self.populate_model_combobox()
        gridLayout.addWidget(self.modelComboBox, 0, 2)

        self.sizeLabel = QLabel()
        gridLayout.addWidget(self.sizeLabel, 0, 4)

        self.quantLabel = QLabel()
        gridLayout.addWidget(self.quantLabel, 0, 6)

        self.modelComboBox.currentIndexChanged.connect(self.updateModelInfo)

        self.set_initial_model()

    def populate_model_combobox(self):
        cuda_available = is_cuda_available()
        available_models = []
        
        for model, info in VISION_MODELS.items():
            if cuda_available or not info.get('requires_cuda', True):
                available_models.append(model)
        
        self.modelComboBox.addItems(available_models)

    def set_initial_model(self):
        config = self.read_config()
        saved_model = config.get('vision', {}).get('chosen_model')

        if saved_model and saved_model in [self.modelComboBox.itemText(i) for i in range(self.modelComboBox.count())]:
            index = self.modelComboBox.findText(saved_model)
            self.modelComboBox.setCurrentIndex(index)
        else:
            self.modelComboBox.setCurrentIndex(0)
        
        self.updateModelInfo()

    def updateModelInfo(self):
        chosen_model = self.modelComboBox.currentText()
        self.updateConfigFile('chosen_model', chosen_model)
        
        model_info = VISION_MODELS[chosen_model]
        self.sizeLabel.setText(model_info['size'])
        self.quantLabel.setText(model_info['precision'])

    def read_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file)
            except Exception:
                pass
        return {}

    def updateConfigFile(self, key, value):
        current_config = self.read_config()
        vision_config = current_config.get('vision', {})
        if vision_config.get(key) != value:
            vision_config[key] = value
            current_config['vision'] = vision_config
            config_file_path = Path('config.yaml')
            with open(config_file_path, 'w', encoding='utf-8') as file:
                yaml.dump(current_config, file)