from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QSizePolicy, QComboBox
from PySide6.QtGui import QIntValidator
import yaml

class ChunkSettingsTab(QWidget):
    def __init__(self):
        super(ChunkSettingsTab, self).__init__()

        with open('config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            self.database_config = config_data['database']
            self.compute_device_options = config_data['Compute_Device']['available']
            self.database_creation_device = config_data['Compute_Device']['database_creation']

        v_layout = QVBoxLayout()

        self.device_label = QLabel(f"Create Device: {self.database_creation_device}")
        self.device_combo = QComboBox()
        self.device_combo.addItems(self.compute_device_options)
        if self.database_creation_device in self.compute_device_options:
            self.device_combo.setCurrentIndex(self.compute_device_options.index(self.database_creation_device))

        device_layout = QHBoxLayout()
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.device_combo)
        v_layout.addLayout(device_layout)

        # Chunk settings
        chunk_settings_layout = QHBoxLayout()
        self.field_data = {}
        self.label_data = {}
        chunk_settings_group = ['chunk_overlap', 'chunk_size']

        for setting in chunk_settings_group:
            current_value = self.database_config.get(setting, '')
            edit = QLineEdit()
            edit.setPlaceholderText(f"Enter new {setting}...")
            edit.setValidator(QIntValidator())
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            label = QLabel(f"{setting.replace('_', ' ').capitalize()}: {current_value}")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            chunk_settings_layout.addWidget(label)
            chunk_settings_layout.addWidget(edit)
            self.field_data[setting] = edit
            self.label_data[setting] = label

        v_layout.addLayout(chunk_settings_layout)
        self.setLayout(v_layout)

    def update_config(self):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        new_device = self.device_combo.currentText()
        if new_device != self.database_creation_device:
            settings_changed = True
            config_data['Compute_Device']['database_creation'] = new_device
            self.database_creation_device = new_device
            self.device_label.setText(f"Create Device: {new_device}")

        for setting, widget in self.field_data.items():
            new_value = widget.text()
            if new_value and new_value != str(self.database_config.get(setting, '')):
                settings_changed = True
                config_data['database'][setting] = int(new_value)
                self.label_data[setting].setText(f"{setting.replace('_', ' ').capitalize()}: {new_value}")
                widget.clear()

        if settings_changed:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f)

        return settings_changed
