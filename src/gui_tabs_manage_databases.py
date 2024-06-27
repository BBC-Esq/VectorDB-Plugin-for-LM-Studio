import json
import shutil
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTreeView, QFileSystemModel, QMenu,
                               QGroupBox, QLabel, QComboBox, QMessageBox)

from utilities import open_file

class CustomFileSystemModel(QFileSystemModel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and index.column() == 0:
            file_path = self.filePath(index)
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        return json.load(file)['metadata'].get('file_name', 'Unknown')
                except FileNotFoundError:
                    return "File Missing"
                except json.JSONDecodeError:
                    return "Invalid JSON"
                except Exception as e:
                    print(f"Error loading JSON file {file_path}: {e}")
                    return "Error"
        return super().data(index, role)

class RefreshingComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

    def showPopup(self):
        self.clear()
        self.addItems(self.parent().load_created_databases())
        super().showPopup()

class ManageDatabasesTab(QWidget):
    def __init__(self):
        super().__init__()
        self.config_path = Path(__file__).resolve().parent / "config.yaml"
        self.created_databases = self.load_created_databases()

        self.layout = QVBoxLayout(self)

        self.documents_group_box = self.create_group_box_with_directory_view("Files in Selected Database", "Docs_for_DB")
        self.layout.addWidget(self.documents_group_box)

        self.database_info_layout = QHBoxLayout()
        self.database_info_label = QLabel("No database selected.")
        self.database_info_layout.addWidget(self.database_info_label)
        self.layout.addLayout(self.database_info_layout)

        self.buttons_layout = QHBoxLayout()
        self.pull_down_menu = RefreshingComboBox(self)
        self.pull_down_menu.currentIndexChanged.connect(self.update_directory_view_and_info_label)
        self.buttons_layout.addWidget(self.pull_down_menu)
        self.create_buttons()
        self.layout.addLayout(self.buttons_layout)

        self.groups = {self.documents_group_box: 1}

    def load_created_databases(self):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return list(config.get('created_databases', {}).keys())
        return []

    def display_no_databases_message(self):
        self.documents_group_box.hide()
        self.database_info_label.setText("No database selected.")

    def create_group_box_with_directory_view(self, title, directory_name):
        group_box = QGroupBox(title)
        group_box.setCheckable(True)
        group_box.setChecked(True)
        layout = QVBoxLayout()
        self.tree_view = QTreeView()
        self.model = CustomFileSystemModel()
        self.tree_view.setModel(self.model)
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        self.tree_view.doubleClicked.connect(self.on_double_click)
        layout.addWidget(self.tree_view)
        group_box.setLayout(layout)
        return group_box

    def update_directory_view_and_info_label(self, index):
        selected_database = self.pull_down_menu.currentText()
        if selected_database:
            self.documents_group_box.show()
            new_path = Path(__file__).resolve().parent / "Vector_DB" / selected_database / "json"
            if new_path.exists():
                self.model.setRootPath(str(new_path))
                self.tree_view.setRootIndex(self.model.index(str(new_path)))
                if self.config_path.exists():
                    with open(self.config_path, 'r', encoding='utf-8') as file:
                        config = yaml.safe_load(file)
                        db_config = config.get('created_databases', {}).get(selected_database, {})
                        model_path = db_config.get('model', '')
                        model_name = model_path.split('/')[-1]
                        chunk_size = db_config.get('chunk_size', '')
                        chunk_overlap = db_config.get('chunk_overlap', '')
                        info_text = f"{model_name}    |    Chunk Size:  {chunk_size}    |    Chunk Overlap:  {chunk_overlap}"
                        self.database_info_label.setText(info_text)
                else:
                    self.database_info_label.setText("Configuration missing.")
            else:
                self.display_no_databases_message()
        else:
            self.display_no_databases_message()

    def on_double_click(self, index):
        file_path = self.tree_view.model().filePath(index)
        if file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    document = json.load(json_file)
                actual_file_path = document['metadata'].get('file_path')
                if actual_file_path and Path(actual_file_path).exists():
                    open_file(actual_file_path)
                else:
                    raise ValueError("File path is missing or invalid in the document metadata.")
            except (json.JSONDecodeError, ValueError) as e:
                QMessageBox.warning(self, "Error", f"Failed to open file: {e}")
        else:
            open_file(file_path)

    def toggle_group_box(self, group_box, checked):
        self.groups[group_box] = 1 if checked else 0
        self.adjust_stretch()

    def adjust_stretch(self):
        total_stretch = sum(stretch for group, stretch in self.groups.items() if group.isChecked())
        for group, stretch in self.groups.items():
            self.layout.setStretchFactor(group, stretch if group.isChecked() else 0)

    def create_buttons(self):
        self.delete_database_button = QPushButton("Delete Database")
        self.buttons_layout.addWidget(self.delete_database_button)
        self.delete_database_button.clicked.connect(self.delete_selected_database)

    def delete_selected_database(self):
        selected_database = self.pull_down_menu.currentText()
        if not selected_database:
            QMessageBox.warning(self, "Delete Database", "No database selected.")
            return

        reply = QMessageBox.question(self, 'Delete Database',
                                     "This cannot be undone.\nClick OK to proceed or Cancel to back out.",
                                     QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)

        if reply == QMessageBox.Ok:
            self.model.setRootPath('')

            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)

                if 'created_databases' in config and selected_database in config['created_databases']:
                    del config['created_databases'][selected_database]

                config.setdefault('database', {})['database_to_search'] = ''

                with open(self.config_path, 'w', encoding='utf-8') as file:
                    yaml.safe_dump(config, file)

                base_dir = Path(__file__).resolve().parent
                deletion_failed = False
                for folder_name in ["Vector_DB", "Vector_DB_Backup", "Docs_for_DB"]:
                    dir_path = base_dir / folder_name / selected_database
                    if dir_path.exists():
                        shutil.rmtree(dir_path, ignore_errors=True)
                        if dir_path.exists():
                            deletion_failed = True
                            print(f"Failed to delete: {dir_path}")

                if deletion_failed:
                    QMessageBox.warning(self, "Delete Database", "Some files/folders could not be deleted. Please check manually.")
                else:
                    QMessageBox.information(self, "Delete Database", f"Database '{selected_database}' and associated files have been deleted.")
                self.refresh_pull_down_menu()
            else:
                QMessageBox.warning(self, "Delete Database", "Configuration file missing or corrupted.")

    def refresh_pull_down_menu(self):
        self.created_databases = self.load_created_databases()
        self.pull_down_menu.clear()
        self.pull_down_menu.addItems(self.created_databases)
        if not self.created_databases:
            self.display_no_databases_message()