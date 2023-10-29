from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QMessageBox, QSizePolicy
from PySide6.QtGui import QIntValidator, QDoubleValidator
import yaml

class ServerSettingsTab(QWidget):
    def __init__(self):
        super(ServerSettingsTab, self).__init__()

        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)
            self.connection_str = config_data.get('server', {}).get('connection_str', '')
            self.current_port = self.connection_str.split(":")[-1].split("/")[0]
            self.current_max_tokens = config_data.get('server', {}).get('model_max_tokens', '')
            self.current_temperature = config_data.get('server', {}).get('model_temperature', '')
            self.current_prefix = config_data.get('server', {}).get('prefix', '')
            self.current_suffix = config_data.get('server', {}).get('suffix', '')

        settings_dict = {
            'port': {"placeholder": "Enter new port...", "validator": QIntValidator(), "current": self.current_port},
            'max_tokens': {"placeholder": "Enter new max tokens...", "validator": QIntValidator(), "current": self.current_max_tokens},
            'temperature': {"placeholder": "Enter new model temperature...", "validator": QDoubleValidator(), "current": self.current_temperature},
            'prefix': {"placeholder": "Enter new prefix...", "validator": None, "current": self.current_prefix},
            'suffix': {"placeholder": "Enter new suffix...", "validator": None, "current": self.current_suffix}
        }

        self.widgets = {}
        layout = QGridLayout()

        row = 0
        for setting, setting_config in settings_dict.items():
            label = QLabel(f"{setting.capitalize()}: {setting_config['current']}")
            edit = QLineEdit()
            edit.setPlaceholderText(setting_config['placeholder'])
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if setting_config['validator']:
                edit.setValidator(setting_config['validator'])

            if setting in ['prefix', 'suffix']:
                edit.textEdited.connect(
                    lambda text, edit=edit: edit.setText(''.join([c for c in text if not c.isdigit()]))
                )

            layout.addWidget(label, row, 0)
            layout.addWidget(edit, row, 1)

            self.widgets[setting] = {"label": label, "edit": edit}
            row += 1

        self.setLayout(layout)

    def update_config(self):
        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)

        updated = False
        for setting, widget in self.widgets.items():
            new_value = widget['edit'].text()
            if new_value:
                updated = True
                if setting == 'port':
                    config_data['server']['connection_str'] = self.connection_str.replace(self.current_port, new_value)
                elif setting in ['max_tokens', 'temperature']:
                    config_data['server'][f'model_{setting}'] = int(new_value) if setting == 'max_tokens' else float(new_value)
                else:
                    config_data['server'][setting] = new_value

                widget['label'].setText(f"{setting.capitalize()}: {new_value}")
                widget['edit'].clear()

        if updated:
            with open('config.yaml', 'w') as file:
                yaml.safe_dump(config_data, file)

        return updated
