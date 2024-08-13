import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QGroupBox, QVBoxLayout, QSizePolicy

class ModelsSettingsTab(QWidget):
    def __init__(self):
        super(ModelsSettingsTab, self).__init__()

        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)

        main_layout = QVBoxLayout()
        self.field_data = {}

        for category, sub_dict in config_data['embedding-models'].items():
            if category in ['bge', 'instructor']:
                group_box = QGroupBox(category)
                layout = QGridLayout()

                row = 0
                for setting, current_value in sub_dict.items():
                    full_key = f"{category}-{setting}"
                    
                    edit = QLineEdit()
                    edit.setPlaceholderText(f"Enter new {setting.lower()}...")
                    edit.textChanged.connect(self.validate_text_only)
                    edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                    label = QLabel(f"{setting}: {current_value}")
                    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                    layout.addWidget(edit, row, 0)
                    layout.addWidget(label, row + 1, 0)

                    self.field_data[full_key] = edit
                    row += 2

                group_box.setLayout(layout)
                group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                main_layout.addWidget(group_box)

        self.setLayout(main_layout)

    def validate_text_only(self, text):
        if not text.isalpha():
            sender = self.sender()
            sender.setText(''.join(filter(str.isalpha, text)))

    def update_config(self):
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        for full_key, widget in self.field_data.items():
            category, setting = full_key.split('-')
            new_value = widget.text()

            if new_value and new_value != config_data['embedding-models'][category][setting]:
                settings_changed = True
                config_data['embedding-models'][category][setting] = new_value
                widget.clear()

        if settings_changed:
            with open('config.yaml', 'w') as f:
                yaml.safe_dump(config_data, f)

        return settings_changed
