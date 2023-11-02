from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QSizePolicy, QComboBox
from PySide6.QtGui import QIntValidator, QDoubleValidator
import yaml

class DatabaseSettingsTab(QWidget):
    def __init__(self):
        super(DatabaseSettingsTab, self).__init__()

        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
            self.database_config = config_data['database']
            self.compute_device_options = config_data['Compute_Device']['available']
            self.database_creation_device = config_data['Compute_Device']['database_creation']
            self.database_query_device = config_data['Compute_Device']['database_query']

        v_layout, h_layout1, h_layout2 = QVBoxLayout(), QHBoxLayout(), QHBoxLayout()
        h_layout_device = QHBoxLayout()

        self.field_data = {}
        self.label_data = {}

        # Create Device ComboBoxes
        device_labels = [f"Create Device {self.database_creation_device}", f"Query Device {self.database_query_device}"]
        self.device_combos = [QComboBox() for _ in device_labels]

        for label_text, combo in zip(device_labels, self.device_combos):
            label = QLabel(label_text)
            combo.addItems(self.compute_device_options)
            combo.setCurrentIndex(self.compute_device_options.index(label_text.split()[-1]))
            h_layout_device.addWidget(label)
            h_layout_device.addWidget(combo)

        v_layout.addLayout(h_layout_device)

        # Database settings
        database_settings_group = ['similarity', 'contexts', 'chunk_overlap', 'chunk_size']

        for setting in database_settings_group:
            current_value = self.database_config.get(setting, '')
            edit = QLineEdit()
            edit.setPlaceholderText(f"Enter new {setting}...")
            edit.setValidator(QDoubleValidator() if setting == 'similarity' else QIntValidator())
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            label = QLabel(f"{setting.replace('_', ' ').capitalize()}: {current_value}")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            target_layout = h_layout1 if setting in ['similarity', 'contexts'] else h_layout2
            target_layout.addWidget(label)
            target_layout.addWidget(edit)
            self.field_data[setting] = edit
            self.label_data[setting] = label

        v_layout.addLayout(h_layout1)
        v_layout.addLayout(h_layout2)
        self.setLayout(v_layout)
        
        message_label = QLabel("<b><u>Only use gpu-acceleration for database creation.</u></b>")
        v_layout.addWidget(message_label)
        
        self.setLayout(v_layout)

    def update_config(self):
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        # Update device configurations
        device_types = ["creation", "query"]
        for device_type, combo in zip(device_types, self.device_combos):
            current_device = config_data['Compute_Device'][f'database_{device_type}']
            new_device = combo.currentText()
            if current_device != new_device:
                config_data['Compute_Device'][f'database_{device_type}'] = new_device
                settings_changed = True

        # Update database settings
        for setting, widget in self.field_data.items():
            new_value = widget.text()
            current_value = config_data['database'].get(setting, '')
            if new_value and new_value != str(current_value):
                settings_changed = True
                config_data['database'][setting] = float(new_value) if setting == 'similarity' else int(new_value)
                self.label_data[setting].setText(f"{setting.replace('_', ' ').capitalize()}: {new_value}")
                widget.clear()

        if settings_changed:
            with open('config.yaml', 'w') as f:
                yaml.safe_dump(config_data, f)

        return settings_changed
