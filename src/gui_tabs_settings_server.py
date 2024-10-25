import yaml
from pathlib import Path

from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QGridLayout, QMessageBox, QSizePolicy, QCheckBox, QComboBox, QMessageBox)

from constants import PROMPT_FORMATS, TOOLTIPS

class ServerSettingsTab(QWidget):
    def __init__(self):
        super(ServerSettingsTab, self).__init__()

        try:
            with open('config.yaml', 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
                self.server_config = self.config_data.get('server', {})
                self.connection_str = self.server_config.get('connection_str', '')
                if ':' in self.connection_str and '/' in self.connection_str:
                    self.current_port = self.connection_str.split(":")[-1].split("/")[0]
                else:
                    self.current_port = ''
                self.current_max_tokens = self.server_config.get('model_max_tokens', '')
                self.current_temperature = self.server_config.get('model_temperature', '')
                self.current_prefix = self.server_config.get('prefix', '')
                self.current_suffix = self.server_config.get('suffix', '')
                self.prompt_format_disabled = self.server_config.get('prompt_format_disabled', False)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}"
            )
            self.server_config = {}
            self.connection_str = ''
            self.current_port = ''
            self.current_max_tokens = ''
            self.current_temperature = ''
            self.current_prefix = ''
            self.current_suffix = ''
            self.prompt_format_disabled = False

        settings_dict = {
            'port': {
                "placeholder": "Port...",
                "validator": QIntValidator(1, 65535),
                "current": self.current_port
            },
            'max_tokens': {
                "placeholder": "Max tokens (-1 or ≥25)...",
                "validator": QIntValidator(-1, 1000000),
                "current": self.current_max_tokens
            },
            'temperature': {
                "placeholder": "Temperature (0.0 - 2.0)...",
                "validator": QDoubleValidator(0.0, 2.0, 4),
                "current": self.current_temperature
            },
            'prefix': {
                "placeholder": "Prefix...",
                "validator": None,
                "current": self.current_prefix
            },
            'suffix': {
                "placeholder": "Suffix...",
                "validator": None,
                "current": self.current_suffix
            }
        }

        self.widgets = {}
        layout = QGridLayout()

        # Port
        port_label = self.create_label('port', settings_dict)
        port_label.setToolTip(TOOLTIPS["PORT"])
        layout.addWidget(port_label, 0, 0)
        port_edit = self.create_edit('port', settings_dict)
        port_edit.setToolTip(TOOLTIPS["PORT"])
        layout.addWidget(port_edit, 0, 1)

        # Max Tokens
        max_tokens_label = self.create_label('max_tokens', settings_dict)
        max_tokens_label.setToolTip(TOOLTIPS["MAX_TOKENS"])
        layout.addWidget(max_tokens_label, 1, 0)
        max_tokens_edit = self.create_edit('max_tokens', settings_dict)
        max_tokens_edit.setToolTip(TOOLTIPS["MAX_TOKENS"])
        layout.addWidget(max_tokens_edit, 1, 1)

        # Temperature
        temp_label = self.create_label('temperature', settings_dict)
        temp_label.setToolTip(TOOLTIPS["TEMPERATURE"])
        layout.addWidget(temp_label, 1, 2)
        temp_edit = self.create_edit('temperature', settings_dict)
        temp_edit.setToolTip(TOOLTIPS["TEMPERATURE"])
        layout.addWidget(temp_edit, 1, 3)

        # Prompt Format
        prompt_format_label = QLabel("Prompt Format:")
        prompt_format_label.setToolTip(TOOLTIPS["PREFIX_SUFFIX"])
        layout.addWidget(prompt_format_label, 2, 0)

        self.prompt_format_combobox = QComboBox()
        self.prompt_format_combobox.addItems([
            "", "ChatML", "Llama2/Mistral", "Neural Chat/SOLAR", "Orca2", "StableLM-Zephyr"
        ])
        self.prompt_format_combobox.setToolTip(TOOLTIPS["PREFIX_SUFFIX"])
        layout.addWidget(self.prompt_format_combobox, 2, 1)
        self.prompt_format_combobox.currentIndexChanged.connect(self.update_prefix_suffix)

        # Disable Prompt Formatting
        disable_label = QLabel("Disable Prompt Formatting:")
        disable_label.setToolTip(TOOLTIPS["DISABLE_PROMPT_FORMATTING"])
        layout.addWidget(disable_label, 2, 2)

        self.disable_checkbox = QCheckBox()
        self.disable_checkbox.setChecked(self.prompt_format_disabled)
        self.disable_checkbox.setToolTip(TOOLTIPS["DISABLE_PROMPT_FORMATTING"])
        layout.addWidget(self.disable_checkbox, 2, 3)

        # Prefix
        prefix_label = self.create_label('prefix', settings_dict)
        prefix_label.setToolTip(TOOLTIPS["PREFIX_SUFFIX"])
        layout.addWidget(prefix_label, 3, 0)
        prefix_edit = self.create_edit('prefix', settings_dict)
        prefix_edit.setToolTip(TOOLTIPS["PREFIX_SUFFIX"])
        layout.addWidget(prefix_edit, 3, 1)

        # Suffix
        suffix_label = self.create_label('suffix', settings_dict)
        suffix_label.setToolTip(TOOLTIPS["PREFIX_SUFFIX"])
        layout.addWidget(suffix_label, 4, 0)
        suffix_edit = self.create_edit('suffix', settings_dict)
        suffix_edit.setToolTip(TOOLTIPS["PREFIX_SUFFIX"])
        layout.addWidget(suffix_edit, 4, 1)

        self.setLayout(layout)

    def create_label(self, setting, settings_dict):
        label_text = f"{setting.replace('_', ' ').capitalize()}: {settings_dict[setting]['current']}"
        label = QLabel(label_text)
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

    def refresh_labels(self):
        self.widgets['prefix']['label'].setText(f"Prefix: {self.server_config.get('prefix', '')}")
        self.widgets['suffix']['label'].setText(f"Suffix: {self.server_config.get('suffix', '')}")

    def update_prefix_suffix(self, index):
        option = self.prompt_format_combobox.currentText()
        if option in PROMPT_FORMATS:
            prefix = PROMPT_FORMATS[option]["prefix"]
            suffix = PROMPT_FORMATS[option]["suffix"]
            self.widgets['prefix']['edit'].setText(prefix)
            self.widgets['suffix']['edit'].setText(suffix)
        else:
            self.widgets['prefix']['edit'].clear()
            self.widgets['suffix']['edit'].clear()

    def update_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with config_file_path.open('r', encoding='utf-8') as file:
                    config_data = yaml.safe_load(file)
                    self.server_config = config_data.get('server', {})
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Configuration",
                    f"An error occurred while loading the configuration: {e}"
                )
                return False
        else:
            QMessageBox.critical(
                self,
                "Configuration File Missing",
                "The configuration file 'config.yaml' does not exist."
            )
            return False

        settings_changed = False
        errors = []

        new_query_device = self.prompt_format_combobox.currentText()
        device_changed = new_query_device != self.server_config.get('database_query', '')

        new_port_text = self.widgets['port']['edit'].text().strip()
        if new_port_text:
            try:
                new_port = int(new_port_text)
                if not (1 <= new_port <= 65535):
                    raise ValueError("Port must be between 1 and 65535.")
            except ValueError:
                errors.append("Port must be an integer between 1 and 65535.")
        else:
            new_port = self.current_port

        new_max_tokens_text = self.widgets['max_tokens']['edit'].text().strip()
        if new_max_tokens_text:
            try:
                if new_max_tokens_text == "-1":
                    new_max_tokens = -1
                else:
                    new_max_tokens = int(new_max_tokens_text)
                    if new_max_tokens < 25:
                        raise ValueError("Max tokens must be -1 or an integer ≥25.")
            except ValueError:
                errors.append("Max tokens must be -1 or an integer ≥25.")
        else:
            new_max_tokens = self.current_max_tokens

        new_temperature_text = self.widgets['temperature']['edit'].text().strip()
        if new_temperature_text:
            try:
                new_temperature = float(new_temperature_text)
                if not (0.0 <= new_temperature <= 2.0):
                    raise ValueError("Temperature must be between 0.0 and 2.0.")
            except ValueError:
                errors.append("Temperature must be a number between 0.0 and 2.0.")
        else:
            new_temperature = self.current_temperature

        new_prefix = self.widgets['prefix']['edit'].text().strip()
        new_suffix = self.widgets['suffix']['edit'].text().strip()

        new_prompt_format_disabled = self.disable_checkbox.isChecked()

        if errors:
            error_message = "\n".join(errors)
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"The following errors occurred:\n{error_message}"
            )
            return False

        if device_changed:
            config_data['Compute_Device']['database_query'] = new_query_device
            settings_changed = True

        if new_port_text and new_port != self.current_port:
            if ':' in self.connection_str and '/' in self.connection_str:
                new_connection_str = self.connection_str.replace(self.current_port, str(new_port))
                config_data['server']['connection_str'] = new_connection_str
                settings_changed = True
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Connection String",
                    "The existing connection string format is invalid. Unable to update port."
                )
                return False

        if new_max_tokens_text and new_max_tokens != self.server_config.get('model_max_tokens', 0):
            config_data['server']['model_max_tokens'] = new_max_tokens
            settings_changed = True

        if new_temperature_text and new_temperature != self.server_config.get('model_temperature', 0.0):
            config_data['server']['model_temperature'] = new_temperature
            settings_changed = True

        if new_prefix and new_prefix != self.server_config.get('prefix', ''):
            config_data['server']['prefix'] = new_prefix
            settings_changed = True

        if new_suffix and new_suffix != self.server_config.get('suffix', ''):
            config_data['server']['suffix'] = new_suffix
            settings_changed = True

        if new_prompt_format_disabled != self.server_config.get('prompt_format_disabled', False):
            config_data['server']['prompt_format_disabled'] = new_prompt_format_disabled
            settings_changed = True

        if settings_changed:
            try:
                with config_file_path.open('w', encoding='utf-8') as file:
                    yaml.safe_dump(config_data, file)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Configuration",
                    f"An error occurred while saving the configuration: {e}"
                )
                return False

            if device_changed:
                self.server_config['database_query'] = new_query_device
                self.widgets['port']['label'].setText(f"Device: {new_query_device}")

            if new_port_text:
                self.connection_str = config_data['server']['connection_str']
                self.current_port = str(new_port)
                self.widgets['port']['label'].setText(f"Port: {new_port}")

            if new_max_tokens_text:
                self.server_config['model_max_tokens'] = new_max_tokens
                self.widgets['max_tokens']['label'].setText(f"Max Tokens: {new_max_tokens}")

            if new_temperature_text:
                self.server_config['model_temperature'] = new_temperature
                self.widgets['temperature']['label'].setText(f"Temperature: {new_temperature}")

            if new_prefix:
                self.server_config['prefix'] = new_prefix
                self.widgets['prefix']['label'].setText(f"Prefix: {new_prefix}")

            if new_suffix:
                self.server_config['suffix'] = new_suffix
                self.widgets['suffix']['label'].setText(f"Suffix: {new_suffix}")

            if new_prompt_format_disabled != self.prompt_format_disabled:
                self.prompt_format_disabled = new_prompt_format_disabled

            self.widgets['port']['edit'].clear()
            self.widgets['max_tokens']['edit'].clear()
            self.widgets['temperature']['edit'].clear()
            self.widgets['prefix']['edit'].clear()
            self.widgets['suffix']['edit'].clear()

        return settings_changed

    def reset_search_term(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with config_file_path.open('r', encoding='utf-8') as file:
                    config_data = yaml.safe_load(file)
                    self.server_config = config_data.get('server', {})
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Configuration",
                    f"An error occurred while loading the configuration: {e}"
                )
                return
        else:
            QMessageBox.critical(
                self,
                "Configuration File Missing",
                "The configuration file 'config.yaml' does not exist."
            )
            return

        config_data['server']['search_term'] = ''

        try:
            with config_file_path.open('w', encoding='utf-8') as file:
                yaml.safe_dump(config_data, file)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"An error occurred while saving the configuration: {e}"
            )
            return

        self.server_config['search_term'] = ''
        self.widgets['prefix']['label'].setText(f"Prefix: {self.server_config.get('prefix', '')}")
        self.widgets['suffix']['label'].setText(f"Suffix: {self.server_config.get('suffix', '')}")
        self.widgets['port']['label'].setText(f"Port: {self.current_port}")
        self.widgets['max_tokens']['label'].setText(f"Max Tokens: {self.current_max_tokens}")
        self.widgets['temperature']['label'].setText(f"Temperature: {self.current_temperature}")
        self.widgets['prefix']['edit'].clear()
        self.widgets['suffix']['edit'].clear()