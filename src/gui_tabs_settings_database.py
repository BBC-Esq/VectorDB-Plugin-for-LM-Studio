from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QHBoxLayout, QSizePolicy
from PySide6.QtGui import QIntValidator, QDoubleValidator
import yaml

class DatabaseSettingsTab(QWidget):
    def __init__(self):
        super(DatabaseSettingsTab, self).__init__()

        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)['database']

        h_layout = QHBoxLayout()
        self.field_data = {}
        self.label_data = {}

        settings_group = ['similarity', 'contexts']

        for setting in settings_group:
            current_value = config_data.get(setting, '')

            edit = QLineEdit()
            edit.setPlaceholderText(f"Enter new {setting}...")
            
            if setting == 'similarity':
                edit.setValidator(QDoubleValidator())
            else:
                edit.setValidator(QIntValidator())
                
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            label = QLabel(f"{setting.replace('_', ' ').capitalize()}: {current_value}")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            h_layout.addWidget(label)
            h_layout.addWidget(edit)

            self.field_data[setting] = edit
            self.label_data[setting] = label

        self.setLayout(h_layout)

    def update_config(self):
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        for setting, widget in self.field_data.items():
            new_value = widget.text()

            if new_value and new_value != str(config_data['database'].get(setting, '')):
                settings_changed = True
                
                if setting == 'similarity':
                    config_data['database'][setting] = float(new_value)
                else:
                    config_data['database'][setting] = int(new_value)
                    
                self.label_data[setting].setText(f"{setting.replace('_', ' ').capitalize()}: {new_value}")
                widget.clear()

        if settings_changed:
            with open('config.yaml', 'w') as f:
                yaml.safe_dump(config_data, f)

        return settings_changed
