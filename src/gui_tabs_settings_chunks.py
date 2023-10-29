from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
import yaml

class ChunkSettingsTab(QWidget):
    def __init__(self):
        super(ChunkSettingsTab, self).__init__()

        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)

        main_layout = QVBoxLayout()
        self.field_data = {}
        self.label_data = {}

        settings_group = ['chunk_overlap', 'chunk_size']

        layout = QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)

        row = 0
        for setting in settings_group:
            current_value = config_data.get(setting, '')

            edit = QLineEdit()
            edit.setPlaceholderText(f"Enter new {setting}...")
            edit.setValidator(QIntValidator())
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            label = QLabel(f"{setting.replace('_', ' ').capitalize()}: {current_value}")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(label, row, 0)
            layout.addWidget(edit, row, 1)

            self.field_data[setting] = edit
            self.label_data[setting] = label
            row += 1

        main_layout.addLayout(layout)
        self.setLayout(main_layout)

    def update_config(self):
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        for setting, widget in self.field_data.items():
            new_value = widget.text()

            if new_value and new_value != str(config_data.get(setting, '')):
                settings_changed = True
                config_data[setting] = int(new_value)
                self.label_data[setting].setText(f"{setting.replace('_', ' ').capitalize()}: {new_value}")
                widget.clear()

        if settings_changed:
            with open('config.yaml', 'w') as f:
                yaml.safe_dump(config_data, f)
        
        return settings_changed
