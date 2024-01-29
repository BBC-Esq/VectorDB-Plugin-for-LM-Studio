from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QTreeView, QFileSystemModel, QMenu, QGroupBox
from PySide6.QtGui import QAction
from PySide6.QtCore import QDir, Qt, QTimer, QThread, Signal
import os
import platform
from pathlib import Path
import yaml
from download_model import download_embedding_model
from choose_documents_and_vector_model import select_embedding_model_directory, choose_documents_directory
import database_interactions
from utilities import check_preconditions_for_db_creation, open_file, delete_file

class CreateDatabaseThread(QThread):
    def run(self):
        create_vector_db = database_interactions.CreateVectorDB()
        create_vector_db.run()

class DatabasesTab(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)

        # Group box
        self.documents_group_box = self.create_group_box("Documents", "Docs_for_DB")
        self.images_group_box = self.create_group_box("Images", "Images_for_DB")

        self.groups = {self.documents_group_box: 1, self.images_group_box: 1}

        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        self.create_buttons()
        self.layout.addLayout(self.buttons_layout)

        # Create Database button
        self.create_db_button = QPushButton("Create Vector Database")
        self.create_db_button.clicked.connect(self.on_create_db_clicked)
        self.layout.addWidget(self.create_db_button)

    def create_group_box(self, title, directory_name):
        group_box = QGroupBox(title)
        group_box.setCheckable(True)
        group_box.setChecked(True)
        layout = QVBoxLayout()
        tree_view = self.setup_directory_view(directory_name)
        layout.addWidget(tree_view)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
        group_box.toggled.connect(lambda checked, gb=group_box: self.toggle_group_box(gb, checked))
        return group_box

    def setup_directory_view(self, directory_name):
        tree_view = QTreeView()
        model = QFileSystemModel()
        tree_view.setModel(model)
        tree_view.setSelectionMode(QTreeView.ExtendedSelection)

        script_dir = Path(__file__).resolve().parent
        directory_path = script_dir / directory_name
        model.setRootPath(str(directory_path))
        tree_view.setRootIndex(model.index(str(directory_path)))

        tree_view.hideColumn(1)
        tree_view.hideColumn(2)
        tree_view.hideColumn(3)

        tree_view.doubleClicked.connect(self.on_double_click)
        tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        tree_view.customContextMenuRequested.connect(self.on_context_menu)

        return tree_view

    def on_double_click(self, index):
        tree_view = self.sender()
        model = tree_view.model()
        file_path = model.filePath(index)
        open_file(file_path)

    def on_context_menu(self, point):
        tree_view = self.sender()
        context_menu = QMenu(self)
        delete_action = QAction("Delete File", self)
        context_menu.addAction(delete_action)

        delete_action.triggered.connect(lambda: self.on_delete_file(tree_view))
        context_menu.exec_(tree_view.viewport().mapToGlobal(point))

    def on_delete_file(self, tree_view):
        selected_indexes = tree_view.selectedIndexes()
        model = tree_view.model()
        for index in selected_indexes:
            if index.column() == 0:
                file_path = model.filePath(index)
                delete_file(file_path)

    def on_create_db_clicked(self):
        self.create_db_button.setDisabled(True)
        QTimer.singleShot(3000, self.reenable_create_db_button)

        checks_passed, message = check_preconditions_for_db_creation(Path(__file__).resolve().parent)
        if not checks_passed:
            if message:
                QMessageBox.warning(self, "Error", message)
            return

        self.create_database_thread = CreateDatabaseThread(self)
        self.create_database_thread.start()

    def reenable_create_db_button(self):
        self.create_db_button.setDisabled(False)

    def toggle_group_box(self, group_box, checked):
        self.tree_views[group_box].setVisible(checked)
        self.adjust_stretch()

    def adjust_stretch(self):
        total_stretch = sum(stretch for group, stretch in self.groups.items() if group.isChecked())
        for group, stretch in self.groups.items():
            self.layout.setStretchFactor(group, stretch if group.isChecked() else 0)

    def create_buttons(self):
        # Choose docs button
        self.choose_docs_button = QPushButton("Choose Files")
        self.choose_docs_button.clicked.connect(choose_documents_directory)
        self.buttons_layout.addWidget(self.choose_docs_button)

        # Choose model directory button
        self.choose_model_dir_button = QPushButton("Choose Model")
        self.choose_model_dir_button.clicked.connect(select_embedding_model_directory)
        self.buttons_layout.addWidget(self.choose_model_dir_button)
