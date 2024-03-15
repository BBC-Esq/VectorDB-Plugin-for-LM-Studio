import shutil
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTreeView, QFileSystemModel, QMenu, QGroupBox, QLabel, QComboBox, QMessageBox
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from pathlib import Path
import yaml
from utilities import open_file, delete_file

class CustomFileSystemModel(QFileSystemModel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.column() == 0:
            original_value = super().data(index, role)
            if original_value.endswith('.pkl'):
                return original_value[:-4]
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
        self.layout = QVBoxLayout(self)
        self.config_path = Path(__file__).resolve().parent / "config.yaml"
        self.created_databases = self.load_created_databases()

        self.documents_group_box = self.create_group_box("Files in Selected Database", "Docs_for_DB")
        self.layout.addWidget(self.documents_group_box)

        self.database_info_label = QLabel("No database selected.")
        self.database_info_layout = QHBoxLayout()
        self.database_info_layout.addWidget(self.database_info_label)
        self.layout.addLayout(self.database_info_layout)

        self.buttons_layout = QHBoxLayout()
        self.pull_down_menu = RefreshingComboBox(self)
        self.pull_down_menu.currentIndexChanged.connect(self.update_directory_view)
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

    def create_group_box(self, title, directory_name):
        group_box = QGroupBox(title)
        group_box.setCheckable(True)
        group_box.setChecked(True)
        layout = QVBoxLayout()
        self.tree_view = self.setup_directory_view(directory_name)
        layout.addWidget(self.tree_view)
        group_box.setLayout(layout)
        return group_box

    def setup_directory_view(self, directory_name):
        self.tree_view = QTreeView()
        self.model = CustomFileSystemModel()
        self.tree_view.setModel(self.model)
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)

        script_dir = Path(__file__).resolve().parent
        directory_path = script_dir / directory_name
        self.model.setRootPath(str(directory_path))
        self.tree_view.setRootIndex(self.model.index(str(directory_path)))

        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)

        self.tree_view.doubleClicked.connect(self.on_double_click)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.on_context_menu)

        return self.tree_view

    def update_directory_view(self, index):
        selected_database = self.pull_down_menu.currentText()
        if selected_database:
            self.documents_group_box.show()
            new_path = Path(__file__).resolve().parent / "Docs_for_DB" / selected_database
            if new_path.exists():
                self.model.setRootPath(str(new_path))
                self.tree_view.setRootIndex(self.model.index(str(new_path)))
                self.update_database_info_label(selected_database)
            else:
                self.display_no_databases_message()
        else:
            self.display_no_databases_message()

    def update_database_info_label(self, selected_database):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                db_config = config.get('created_databases', {}).get(selected_database, {})
                model_path = db_config.get('model', '')
                model_name = model_path.split('/')[-1]
                chunk_size = db_config.get('chunk_size', '')
                chunk_overlap = db_config.get('chunk_overlap', '')
                info_text = f"Created with:  {model_name}       Chunk Size:  {chunk_size}       Chunk Overlap:  {chunk_overlap}"
                self.database_info_label.setText(info_text)
        else:
            self.database_info_label.setText("Configuration missing.")

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
                for folder_name in ["Vector_DB", "Vector_DB_Backup", "Docs_for_DB"]:
                    dir_path = base_dir / folder_name / selected_database
                    if dir_path.exists() and dir_path.is_dir():
                        shutil.rmtree(dir_path)

                QMessageBox.information(self, "Delete Database", f"Database '{selected_database}' and associated files have been deleted.")
                self.refresh_pull_down_menu()

    def refresh_pull_down_menu(self):
        self.created_databases = self.load_created_databases()
        self.pull_down_menu.clear()
        self.pull_down_menu.addItems(self.created_databases)
        if not self.created_databases:
            self.display_no_databases_message()