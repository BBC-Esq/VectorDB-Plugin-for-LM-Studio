from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QMessageBox, QSizePolicy, QCheckBox
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
            self.prompt_format_disabled = config_data.get('server', {}).get('prompt_format_disabled', False)

        settings_dict = {
            'port': {"placeholder": "Enter new port...", "validator": QIntValidator(), "current": self.current_port},
            'max_tokens': {"placeholder": "Enter new max tokens...", "validator": QIntValidator(), "current": self.current_max_tokens},
            'temperature': {"placeholder": "Enter new model temperature...", "validator": QDoubleValidator(), "current": self.current_temperature},
            'prefix': {"placeholder": "Enter new prefix...", "validator": None, "current": self.current_prefix},
            'suffix': {"placeholder": "Enter new suffix...", "validator": None, "current": self.current_suffix}
        }

        self.widgets = {}
        layout = QGridLayout()

        layout.addWidget(self.create_label('port', settings_dict), 0, 0)
        layout.addWidget(self.create_edit('port', settings_dict), 0, 1)
        layout.addWidget(self.create_label('max_tokens', settings_dict), 1, 0)
        layout.addWidget(self.create_edit('max_tokens', settings_dict), 1, 1)
        layout.addWidget(self.create_label('temperature', settings_dict), 1, 2)
        layout.addWidget(self.create_edit('temperature', settings_dict), 1, 3)

        prompt_format_label = QLabel("Prompt Format:")
        layout.addWidget(prompt_format_label, 2, 0)

        disable_label = QLabel("Disable")
        layout.addWidget(disable_label, 2, 2)

        self.disable_checkbox = QCheckBox()
        self.initial_disable_state = self.prompt_format_disabled
        self.disable_checkbox.setChecked(self.prompt_format_disabled)
        layout.addWidget(self.disable_checkbox, 2, 3)

        layout.addWidget(self.create_label('prefix', settings_dict), 3, 0)
        layout.addWidget(self.create_edit('prefix', settings_dict), 3, 1)
        layout.addWidget(self.create_label('suffix', settings_dict), 3, 2)
        layout.addWidget(self.create_edit('suffix', settings_dict), 3, 3)

        self.setLayout(layout)

    def create_label(self, setting, settings_dict):
        label = QLabel(f"{setting.capitalize()}: {settings_dict[setting]['current']}")
        self.widgets[setting] = {"label": label}
        return label

    def create_edit(self, setting, settings_dict):
        edit = QLineEdit()
        edit.setPlaceholderText(settings_dict[setting]['placeholder'])
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if settings_dict[setting]['validator']:
            edit.setValidator(settings_dict[setting]['validator'])
        if setting in ['prefix', 'suffix']:
            edit.textEdited.connect(
                lambda text, edit=edit: edit.setText(''.join([c for c in text if not c.isdigit()]))
            )
        self.widgets[setting]['edit'] = edit
        return edit

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

        current_disable_state = self.disable_checkbox.isChecked()
        if current_disable_state != self.initial_disable_state:
            updated = True
            config_data['server']['prompt_format_disabled'] = current_disable_state
            self.initial_disable_state = current_disable_state

        if updated:
            with open('config.yaml', 'w') as file:
                yaml.safe_dump(config_data, file)

        return updated
