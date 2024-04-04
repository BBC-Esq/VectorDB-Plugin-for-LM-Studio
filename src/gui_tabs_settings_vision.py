from PySide6.QtWidgets import QLabel, QGridLayout, QVBoxLayout, QComboBox, QWidget
from PySide6.QtCore import Qt
import yaml
from pathlib import Path
from constants import VISION_MODELS

class VisionSettingsTab(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)

        # Options layout
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
        self.modelComboBox.addItems(list(VISION_MODELS.keys()))
        gridLayout.addWidget(self.modelComboBox, 0, 2)

        self.sizeComboBox = QComboBox()
        gridLayout.addWidget(self.sizeComboBox, 0, 4)

        self.quantComboBox = QComboBox()
        gridLayout.addWidget(self.quantComboBox, 0, 6)

        self.modelComboBox.currentIndexChanged.connect(self.updateChosenModel)
        self.sizeComboBox.currentIndexChanged.connect(self.updateChosenSize)
        self.quantComboBox.currentIndexChanged.connect(self.updateChosenQuant)

        # Initial widget states
        self.updateComboBoxes()
        self.updateChosenModel()

    def updateComboBoxes(self):
        model = self.modelComboBox.currentText()
        sizes = VISION_MODELS[model]['available_sizes']
        quants = VISION_MODELS[model]['available_quants']

        self.sizeComboBox.clear()
        self.quantComboBox.clear()

        self.sizeComboBox.addItems(sizes)
        self.quantComboBox.addItems(quants)

    def updateChosenModel(self):
        chosen_model = self.modelComboBox.currentText()
        self.updateConfigFile('chosen_model', chosen_model)
        self.updateComboBoxes()

    def updateChosenSize(self):
        chosen_size = self.sizeComboBox.currentText()
        self.updateConfigFile('chosen_size', chosen_size)

    def updateChosenQuant(self):
        chosen_quant = self.quantComboBox.currentText()
        self.updateConfigFile('chosen_quant', chosen_quant)

    def updateConfigFile(self, key, value):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r', encoding='utf-8') as file:
                    current_config = yaml.safe_load(file)
            except Exception:
                current_config = {}

            vision_config = current_config.get('vision', {})
            vision_config[key] = value
            current_config['vision'] = vision_config

            with open(config_file_path, 'w', encoding='utf-8') as file:
                yaml.dump(current_config, file)