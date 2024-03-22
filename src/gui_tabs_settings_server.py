from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QMessageBox, QSizePolicy, QCheckBox, QComboBox
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtCore import Qt
import yaml
from pathlib import Path

class ServerSettingsTab(QWidget):
    def __init__(self):
        super(ServerSettingsTab, self).__init__()

        with open('config.yaml', 'r') as file:
            self.config_data = yaml.safe_load(file)
            self.connection_str = self.config_data.get('server', {}).get('connection_str', '')
            self.current_port = self.connection_str.split(":")[-1].split("/")[0]
            self.current_max_tokens = self.config_data.get('server', {}).get('model_max_tokens', '')
            self.current_temperature = self.config_data.get('server', {}).get('model_temperature', '')
            self.current_prefix = self.config_data.get('server', {}).get('prefix', '')
            self.current_suffix = self.config_data.get('server', {}).get('suffix', '')
            self.prompt_format_disabled = self.config_data.get('server', {}).get('prompt_format_disabled', False)

        settings_dict = {
            'port': {"placeholder": "Enter new port...", "validator": QIntValidator(), "current": self.current_port},
            'max_tokens': {"placeholder": "Enter new max tokens...", "validator": QIntValidator(), "current": self.current_max_tokens},
            'temperature': {"placeholder": "Enter new model temperature...", "validator": QDoubleValidator(), "current": self.current_temperature},
            'prefix': {"placeholder": "Enter new prefix...", "validator": None, "current": self.current_prefix},
            'suffix': {"placeholder": "Enter new suffix...", "validator": None, "current": self.current_suffix}
        }

        self.widgets = {}
        layout = QGridLayout()

        # Prompt Format and its combo box moved to the far left (positions 0 and 1 in the first column)
        prompt_format_label = QLabel("Prompt Format:")
        layout.addWidget(prompt_format_label, 0, 0)
        layout.setAlignment(prompt_format_label, Qt.AlignCenter)

        self.prompt_format_combobox = QComboBox()
        self.prompt_format_combobox.addItems(["", "ChatML", "Llama2/Mistral", "Neural Chat/SOLAR", "Orca2", "StableLM-Zephyr"])
        layout.addWidget(self.prompt_format_combobox, 1, 0)
        layout.setAlignment(self.prompt_format_combobox, Qt.AlignCenter)
        self.prompt_format_combobox.currentIndexChanged.connect(self.update_prefix_suffix)

        # Disable Prompt and its checkbox moved next to the Prompt Format (positions 0 and 1 in the second column)
        disable_label = QLabel("Disable Prompt:")
        layout.addWidget(disable_label, 0, 1)
        layout.setAlignment(disable_label, Qt.AlignCenter)

        self.disable_checkbox = QCheckBox()
        self.disable_checkbox.setChecked(self.prompt_format_disabled)
        layout.addWidget(self.disable_checkbox, 1, 1)
        layout.setAlignment(self.disable_checkbox, Qt.AlignCenter)

        # Moving port, max tokens, and temperature next in the order (columns 2, 3, and 4 respectively)
        port_label = self.create_label('port', settings_dict)
        layout.addWidget(port_label, 0, 2)
        layout.setAlignment(port_label, Qt.AlignCenter)

        layout.addWidget(self.create_edit('port', settings_dict), 1, 2)
        layout.setAlignment(self.create_edit('port', settings_dict), Qt.AlignCenter)

        max_tokens_label = self.create_label('max_tokens', settings_dict)
        layout.addWidget(max_tokens_label, 0, 3)
        layout.setAlignment(max_tokens_label, Qt.AlignCenter)

        layout.addWidget(self.create_edit('max_tokens', settings_dict), 1, 3)
        layout.setAlignment(self.create_edit('max_tokens', settings_dict), Qt.AlignCenter)

        temperature_label = self.create_label('temperature', settings_dict)
        layout.addWidget(temperature_label, 0, 4)
        layout.setAlignment(temperature_label, Qt.AlignCenter)

        layout.addWidget(self.create_edit('temperature', settings_dict), 1, 4)
        layout.setAlignment(self.create_edit('temperature', settings_dict), Qt.AlignCenter)

        # Keeping prefix and suffix at the bottom and making them span the entire row
        layout.addWidget(self.create_label('prefix', settings_dict), 3, 0)
        layout.addWidget(self.create_edit('prefix', settings_dict), 3, 1, 1, 4)  # Spanning 4 columns for alignment
        layout.addWidget(self.create_label('suffix', settings_dict), 4, 0)
        layout.addWidget(self.create_edit('suffix', settings_dict), 4, 1, 1, 4)  # Spanning 4 columns for alignment

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

    def refresh_labels(self):
        self.widgets['prefix']['label'].setText(f"Prefix: {self.config_data.get('server', {}).get('prefix', '')}")
        self.widgets['suffix']['label'].setText(f"Suffix: {self.config_data.get('server', {}).get('suffix', '')}")
    
    def update_prefix_suffix(self, index):
        option = self.prompt_format_combobox.currentText()
        key_mapping = {
            "ChatML": ("prefix_chat_ml", "suffix_chat_ml"),
            "Llama2/Mistral": ("prefix_llama2_and_mistral", "suffix_llama2_and_mistral"),
            "Neural Chat/SOLAR": ("prefix_neural_chat", "suffix_neural_chat"),
            "Orca2": ("prefix_orca2", "suffix_orca2"),
            "StableLM-Zephyr": ("prefix_stablelm-zephyr", "suffix_stablelm-zephyr"),
        }
        prefix_key, suffix_key = key_mapping.get(option, ("", ""))
        self.widgets['prefix']['edit'].setText(self.config_data.get('server', {}).get(prefix_key, ''))
        self.widgets['suffix']['edit'].setText(self.config_data.get('server', {}).get(suffix_key, ''))

    def update_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with config_file_path.open('r') as file:
                    config_data = yaml.safe_load(file)
            except Exception as e:
                config_data = {}

        updated = False
        for setting, widget in self.widgets.items():
            new_value = widget['edit'].text()
            if new_value:
                updated = True
                if setting == 'port':
                    config_data['server']['connection_str'] = self.connection_str.replace(self.current_port, new_value)
                    self.widgets['port']['label'].setText(f"Port: {new_value}")
                elif setting == 'max_tokens':
                    config_data['server']['model_max_tokens'] = int(new_value)
                    self.widgets['max_tokens']['label'].setText(f"Max Tokens: {new_value}")
                elif setting == 'temperature':
                    config_data['server']['model_temperature'] = float(new_value)
                    self.widgets['temperature']['label'].setText(f"Temperature: {new_value}")
                else:
                    config_data['server'][setting] = new_value
                    self.widgets[setting]['label'].setText(f"{setting.capitalize()}: {new_value}")

                widget['edit'].clear()

        checkbox_state = self.disable_checkbox.isChecked()
        if checkbox_state != config_data.get('server', {}).get('prompt_format_disabled', False):
            config_data['server']['prompt_format_disabled'] = checkbox_state
            updated = True

        if updated:
            with config_file_path.open('w') as file:
                yaml.safe_dump(config_data, file)

        return updated