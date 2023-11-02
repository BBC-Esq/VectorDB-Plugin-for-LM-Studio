from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QSizePolicy
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

        layout = QHBoxLayout()

        for setting in settings_group:
            current_value = config_data.get(setting, '')

            edit = QLineEdit()
            edit.setPlaceholderText(f"Enter new {setting}...")
            edit.setValidator(QIntValidator())
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            label = QLabel(f"{setting.replace('_', ' ').capitalize()}: {current_value}")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(label)
            layout.addWidget(edit)

            self.field_data[setting] = edit
            self.label_data[setting] = label

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
