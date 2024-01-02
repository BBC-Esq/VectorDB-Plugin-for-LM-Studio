from PySide6.QtWidgets import QLabel, QGridLayout, QVBoxLayout, QComboBox, QCheckBox, QSpinBox, QWidget
from PySide6.QtCore import Qt
import yaml
import os

class VisionSettingsTab(QWidget):
    def __init__(self):
        super().__init__()

        # Read YAML file
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

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
        gridLayout.addWidget(label_size, 0, 2)
        gridLayout.setAlignment(label_size, Qt.AlignCenter)

        label_quant = QLabel("Quant")
        gridLayout.addWidget(label_quant, 0, 3)
        gridLayout.setAlignment(label_quant, Qt.AlignCenter)

        label_flash = QLabel("Flash Attn 2")
        gridLayout.addWidget(label_flash, 0, 4)
        gridLayout.setAlignment(label_flash, Qt.AlignCenter)

        label_batch = QLabel("Batch")
        gridLayout.addWidget(label_batch, 0, 5)
        gridLayout.setAlignment(label_batch, Qt.AlignCenter)

        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(["llava", "bakllava", "cogvlm"])
        gridLayout.addWidget(self.modelComboBox, 1, 1)

        self.sizeComboBox = QComboBox()
        gridLayout.addWidget(self.sizeComboBox, 1, 2)

        self.quantComboBox = QComboBox()
        gridLayout.addWidget(self.quantComboBox, 1, 3)

        flashCheckBox = QCheckBox()
        gridLayout.addWidget(flashCheckBox, 1, 4)
        gridLayout.setAlignment(flashCheckBox, Qt.AlignCenter)
        flashCheckBox.setEnabled(False)

        batchSpinBox = QSpinBox()
        gridLayout.addWidget(batchSpinBox, 1, 5)
        gridLayout.setAlignment(batchSpinBox, Qt.AlignCenter)
        batchSpinBox.setEnabled(False)

        self.modelComboBox.currentIndexChanged.connect(self.updateChosenModel)
        self.sizeComboBox.currentIndexChanged.connect(self.updateChosenSize)
        self.quantComboBox.currentIndexChanged.connect(self.updateChosenQuant)

        # Initial widget states
        self.updateComboBoxes()
        self.updateChosenModel()

    def updateComboBoxes(self):
        model = self.modelComboBox.currentText()
        sizes = self.config['vision'][model]['available_sizes']
        quants = self.config['vision'][model]['available_quants']

        self.sizeComboBox.clear()
        self.quantComboBox.clear()
        self.sizeComboBox.addItems(sizes)
        self.quantComboBox.addItems(quants)

    def updateChosenModel(self):
        chosen_model = self.modelComboBox.currentText()
        self.config['vision']['chosen_model'] = chosen_model
        self.updateConfigFile()
        self.updateComboBoxes()

    def updateChosenSize(self):
        chosen_size = self.sizeComboBox.currentText()
        self.config['vision']['chosen_size'] = chosen_size
        self.updateConfigFile()

    def updateChosenQuant(self):
        chosen_quant = self.quantComboBox.currentText()
        self.config['vision']['chosen_quant'] = chosen_quant
        self.updateConfigFile()

    def updateConfigFile(self):
        config_file_path = 'config.yaml'
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as file:
                    current_config = yaml.safe_load(file)
            except Exception as e:
                current_config = {}

        # Update only the 'vision' section of the config
        vision_config = current_config.get('vision', {})
        vision_config['chosen_model'] = self.config['vision']['chosen_model']
        vision_config['chosen_size'] = self.config['vision']['chosen_size']
        vision_config['chosen_quant'] = self.config['vision']['chosen_quant']
        current_config['vision'] = vision_config

        with open(config_file_path, 'w') as file:
            yaml.dump(current_config, file)