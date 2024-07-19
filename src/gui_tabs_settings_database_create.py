import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QSizePolicy, QComboBox

class ChunkSettingsTab(QWidget):
    def __init__(self):
        super(ChunkSettingsTab, self).__init__()
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            self.database_config = config_data['database']
            self.compute_device_options = config_data['Compute_Device']['available']
            self.database_creation_device = config_data['Compute_Device']['database_creation']
        grid_layout = QGridLayout()
        
        # Device selection and current setting
        self.device_label = QLabel("Device:")
        grid_layout.addWidget(self.device_label, 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(self.compute_device_options)
        if self.database_creation_device in self.compute_device_options:
            self.device_combo.setCurrentIndex(self.compute_device_options.index(self.database_creation_device))
        self.device_combo.setMinimumWidth(100)
        grid_layout.addWidget(self.device_combo, 0, 2)
        self.current_device_label = QLabel(f"{self.database_creation_device}")
        grid_layout.addWidget(self.current_device_label, 0, 1)
        
        # Chunk size and current setting (moved to the left)
        self.chunk_size_label = QLabel("Chunk Size (# characters):")
        grid_layout.addWidget(self.chunk_size_label, 0, 3)
        self.chunk_size_edit = QLineEdit()
        self.chunk_size_edit.setPlaceholderText("Enter new chunk_size...")
        self.chunk_size_edit.setValidator(QIntValidator())
        grid_layout.addWidget(self.chunk_size_edit, 0, 5)
        current_size = self.database_config.get('chunk_size', '')
        self.current_size_label = QLabel(f"{current_size}")
        grid_layout.addWidget(self.current_size_label, 0, 4)
        
        # Chunk overlap and current setting (moved to the right)
        self.chunk_overlap_label = QLabel("Overlap (# characters):")
        grid_layout.addWidget(self.chunk_overlap_label, 0, 6)
        self.chunk_overlap_edit = QLineEdit()
        self.chunk_overlap_edit.setPlaceholderText("Enter new chunk_overlap...")
        self.chunk_overlap_edit.setValidator(QIntValidator())
        grid_layout.addWidget(self.chunk_overlap_edit, 0, 8)
        current_overlap = self.database_config.get('chunk_overlap', '')
        self.current_overlap_label = QLabel(f"{current_overlap}")
        grid_layout.addWidget(self.current_overlap_label, 0, 7)
        
        self.setLayout(grid_layout)

    def update_config(self):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        new_device = self.device_combo.currentText()
        if new_device != self.database_creation_device:
            settings_changed = True
            config_data['Compute_Device']['database_creation'] = new_device
            self.database_creation_device = new_device
            self.current_device_label.setText(f"{new_device}")

        new_chunk_overlap = self.chunk_overlap_edit.text()
        if new_chunk_overlap and new_chunk_overlap != str(self.database_config.get('chunk_overlap', '')):
            settings_changed = True
            config_data['database']['chunk_overlap'] = int(new_chunk_overlap)
            self.current_overlap_label.setText(f"{new_chunk_overlap}")

        new_chunk_size = self.chunk_size_edit.text()
        if new_chunk_size and new_chunk_size != str(self.database_config.get('chunk_size', '')):
            settings_changed = True
            config_data['database']['chunk_size'] = int(new_chunk_size)
            self.current_size_label.setText(f"{new_chunk_size}")

        if settings_changed:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f)
            
            self.chunk_overlap_edit.clear()
            self.chunk_size_edit.clear()

        return settings_changed