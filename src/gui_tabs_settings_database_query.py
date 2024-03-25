from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QSizePolicy, QComboBox, QPushButton
from PySide6.QtGui import QIntValidator, QDoubleValidator
import yaml

class DatabaseSettingsTab(QWidget):
    def __init__(self):
        super(DatabaseSettingsTab, self).__init__()

        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
            self.database_config = config_data['database']
            self.compute_device_options = config_data['Compute_Device']['available']
            self.database_query_device = config_data['Compute_Device']['database_query']
            self.search_term = config_data['database'].get('search_term', '')
            self.document_type = config_data['database'].get('document_types', '')

        grid_layout = QGridLayout()

        self.field_data = {}
        self.label_data = {}

        self.query_device_label = QLabel(f"Query Device: {self.database_query_device}")
        self.query_device_combo = QComboBox()
        self.query_device_combo.addItems(self.compute_device_options)
        if self.database_query_device in self.compute_device_options:
            self.query_device_combo.setCurrentIndex(self.compute_device_options.index(self.database_query_device))
        grid_layout.addWidget(self.query_device_label, 0, 0)
        grid_layout.addWidget(self.query_device_combo, 0, 1)

        # Add similarity settings
        similarity_value = self.database_config.get('similarity', '')
        self.similarity_edit = QLineEdit()
        self.similarity_edit.setPlaceholderText("Enter new similarity...")
        self.similarity_edit.setValidator(QDoubleValidator())
        self.similarity_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.similarity_label = QLabel(f"Similarity: {similarity_value}")
        self.similarity_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.similarity_label, 1, 0)
        grid_layout.addWidget(self.similarity_edit, 1, 1)
        self.field_data['similarity'] = self.similarity_edit
        self.label_data['similarity'] = self.similarity_label

        # Add contexts settings
        contexts_value = self.database_config.get('contexts', '')
        self.contexts_edit = QLineEdit()
        self.contexts_edit.setPlaceholderText("Enter new contexts...")
        self.contexts_edit.setValidator(QIntValidator())
        self.contexts_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.contexts_label = QLabel(f"Contexts: {contexts_value}")
        self.contexts_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.contexts_label, 1, 3)
        grid_layout.addWidget(self.contexts_edit, 1, 4)
        self.field_data['contexts'] = self.contexts_edit
        self.label_data['contexts'] = self.contexts_label

        self.search_term_edit = QLineEdit()
        self.search_term_edit.setPlaceholderText("Enter new search term...")
        self.search_term_edit.setText(self.search_term)
        self.search_term_label = QLabel(f"Search Term Filter: {self.search_term}")
        self.filter_button = QPushButton("Clear Filter")
        self.filter_button.clicked.connect(self.reset_search_term)
        grid_layout.addWidget(self.search_term_label, 2, 0)
        grid_layout.addWidget(self.search_term_edit, 2, 1)
        grid_layout.addWidget(self.filter_button, 2, 2)

        self.file_type_combo = QComboBox()
        file_type_items = ["All Files", "Images Only", "Documents Only", "Audio Only"]
        self.file_type_combo.addItems(file_type_items)

        if self.document_type == 'image':
            default_index = file_type_items.index("Images Only")
        elif self.document_type == 'document':
            default_index = file_type_items.index("Documents Only")
        elif self.document_type == 'audio':
            default_index = file_type_items.index("Audio Only")
        else:
            default_index = file_type_items.index("All Files")
        self.file_type_combo.setCurrentIndex(default_index)

        grid_layout.addWidget(QLabel("File Type:"), 2, 3)
        grid_layout.addWidget(self.file_type_combo, 2, 4)

        self.setLayout(grid_layout)

    def update_config(self):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        new_query_device = self.query_device_combo.currentText()
        if new_query_device != config_data['Compute_Device'].get('database_query', ''):
            settings_changed = True
            config_data['Compute_Device']['database_query'] = new_query_device
            self.query_device_label.setText(f"Query Device: {new_query_device}")

        database_settings = {'search_term': self.search_term_edit, **self.field_data}

        for setting, widget in database_settings.items():
            new_value = widget.text()
            if new_value and new_value != str(config_data['database'].get(setting, '')):
                settings_changed = True
                config_data['database'][setting] = float(new_value) if setting == 'similarity' else new_value
                if setting == 'search_term':
                    self.search_term_label.setText(f"Search Term: {new_value}")
                else:
                    self.label_data[setting].setText(f"{setting.replace('_', ' ').capitalize()}: {new_value}")
                widget.clear()

        file_type_map = {
            "All Files": '',
            "Images Only": 'image',
            "Documents Only": 'document',
            "Audio Only": 'audio' # add items to map here
        }

        file_type_selection = self.file_type_combo.currentText()
        document_type_value = file_type_map[file_type_selection]

        if document_type_value != config_data['database'].get('document_types', ''):
            settings_changed = True
            config_data['database']['document_types'] = document_type_value

        if settings_changed:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f)

        return settings_changed

    def reset_search_term(self):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        config_data['database']['search_term'] = ''
        
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_data, f)

        self.search_term_label.setText("Search Term: ")
        self.search_term_edit.clear()