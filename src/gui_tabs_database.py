from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QTreeView, QFileSystemModel, QMenu, QGroupBox, QComboBox, QLabel
from PySide6.QtGui import QAction
from PySide6.QtCore import QDir, Qt

class DatabaseTab(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)

        # Database name label
        self.db_name_label = QLabel("Displaying database name: XYZ")
        self.layout.addWidget(self.db_name_label)       

       # Group box
        self.documents_group_box = self.create_group_box("Documents")

        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        self.create_buttons()
        self.layout.addLayout(self.buttons_layout)

        # Create Database button
        self.create_db_button = QPushButton("Create Vector Database")
        self.layout.addWidget(self.create_db_button)

    def create_group_box(self, title):
        group_box = QGroupBox(title)
        group_box.setCheckable(True)
        group_box.setChecked(True)
        layout = QVBoxLayout()
        tree_view = QTreeView()
        layout.addWidget(tree_view)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        return group_box

    def create_buttons(self):
        # Pull-down menu
        self.pull_down_menu = QComboBox()
        self.buttons_layout.addWidget(self.pull_down_menu)
        
        # Add docs button
        self.choose_docs_button = QPushButton("Add Documents")
        self.buttons_layout.addWidget(self.choose_docs_button)

        # Choose model directory button
        self.choose_model_dir_button = QPushButton("Choose Model")
        self.buttons_layout.addWidget(self.choose_model_dir_button)
