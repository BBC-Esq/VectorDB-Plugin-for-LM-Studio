import shutil
import sqlite3
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTableView, QMenu,
                               QGroupBox, QLabel, QComboBox, QMessageBox, QHeaderView)

from utilities import open_file

class SQLiteTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []
        self._headers = ["File Name"]

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][0]
        elif role == Qt.ForegroundRole:
            return QColor('white')
        return None

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return 1

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return None

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

        self.documents_group_box = self.create_group_box_with_table_view("Files in Selected Database")
        self.layout.addWidget(self.documents_group_box)

        self.database_info_layout = QHBoxLayout()
        self.database_info_label = QLabel("No database selected.")
        self.database_info_layout.addWidget(self.database_info_label)
        self.layout.addLayout(self.database_info_layout)

        self.buttons_layout = QHBoxLayout()
        self.pull_down_menu = RefreshingComboBox(self)
        self.pull_down_menu.currentIndexChanged.connect(self.update_table_view_and_info_label)
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

    def create_group_box_with_table_view(self, title):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        self.table_view = QTableView()
        self.model = SQLiteTableModel()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.doubleClicked.connect(self.on_double_click)
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)
        
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.table_view)
        group_box.setLayout(layout)
        return group_box

    def update_table_view_and_info_label(self, index):
        selected_database = self.pull_down_menu.currentText() if index != -1 else ""
        if selected_database:
            self.documents_group_box.show()
            db_path = Path(__file__).resolve().parent / "Vector_DB" / selected_database / "metadata.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT file_name, file_path FROM document_metadata")
                data = cursor.fetchall()
                conn.close()

                self.model._data = [(row[0], row[1]) for row in data]
                self.model.layoutChanged.emit()

                if self.config_path.exists():
                    with open(self.config_path, 'r', encoding='utf-8') as file:
                        config = yaml.safe_load(file)
                        db_config = config.get('created_databases', {}).get(selected_database, {})
                        model_path = db_config.get('model', '')
                        model_name = Path(model_path).name
                        chunk_size = db_config.get('chunk_size', '')
                        chunk_overlap = db_config.get('chunk_overlap', '')
                        info_text = f"DB name:  \"{selected_database}\"   |   Created with \"{model_name}\"   |   Chunk  size/overlap = {chunk_size} / {chunk_overlap}."
                        self.database_info_label.setText(info_text)
                else:
                    self.database_info_label.setText("Configuration missing.")
            else:
                self.display_no_databases_message()
        else:
            self.display_no_databases_message()

    def on_double_click(self, index):
        selected_database = self.pull_down_menu.currentText()
        if selected_database:
            file_path = self.model._data[index.row()][1]
            if Path(file_path).exists():
                open_file(file_path)
            else:
                QMessageBox.warning(self, "Error", f"File not found at the specified path: {file_path}")
        else:
            QMessageBox.warning(self, "Error", "No database selected.")

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
            self.model.beginResetModel()
            self.model._data = []
            self.model.endResetModel()

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
                for folder_name in ["Vector_DB", "Vector_DB_Backup"]:
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
                self.update_table_view_and_info_label(-1)
            else:
                QMessageBox.warning(self, "Delete Database", "Configuration file missing or corrupted.")

    def refresh_pull_down_menu(self):
        self.created_databases = self.load_created_databases()
        self.pull_down_menu.clear()
        self.pull_down_menu.addItems(self.created_databases)
        if not self.created_databases:
            self.display_no_databases_message()

    def show_context_menu(self, position):
        context_menu = QMenu(self)
        delete_action = QAction("Delete File", self)
        delete_action.triggered.connect(self.delete_selected_file)
        context_menu.addAction(delete_action)
        
        context_menu.exec_(self.table_view.viewport().mapToGlobal(position))

    def delete_selected_file(self):
        # Placeholder function for delete functionality
        print("Delete file functionality will be implemented here.")
        # TODO: Implement actual file deletion logic