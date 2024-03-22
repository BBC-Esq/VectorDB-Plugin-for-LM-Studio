import subprocess
import os
import yaml
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

def choose_documents_directory():
    allowed_extensions = ['.pdf', '.docx', '.epub', '.txt', '.enex', '.eml', '.msg', '.csv', '.xls', '.xlsx', '.rtf', '.odt',
                          '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.html', '.htm', '.md', '.doc']
    current_dir = Path(__file__).parent.resolve()
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_paths, _ = file_dialog.getOpenFileNames(None, "Choose Documents and Images for Database", str(current_dir))

    if file_paths:
        incompatible_files = []
        compatible_files = []

        for file_path in file_paths:
            extension = Path(file_path).suffix.lower()
            if extension in allowed_extensions:
                compatible_files.append(file_path)
            else:
                incompatible_files.append(Path(file_path).name)

        if incompatible_files:
            dialog_text = "The following files cannot be added here due to their file extension:\n\n" + "\n".join(incompatible_files) + "\n\nHowever, if any of them are audio files you can still add them directly in the Tools Tab."
            dialog_text += "\n\nClick 'Ok' to add the compatible documents only (remembering to add audio files separately) or 'Cancel' to back out completely."
            incompatible_dialog = QDialog()
            incompatible_dialog.resize(800, 600)
            incompatible_dialog.setWindowTitle("Incompatible Files Detected")
            layout = QVBoxLayout()

            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setText(dialog_text)
            layout.addWidget(text_edit)

            button_box = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            button_box.addWidget(ok_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)

            incompatible_dialog.setLayout(layout)

            ok_button.clicked.connect(incompatible_dialog.accept)
            cancel_button.clicked.connect(incompatible_dialog.reject)

            user_choice = incompatible_dialog.exec()

            if user_choice == QDialog.Rejected:
                return

        target_folder = current_dir / "Docs_for_DB"
        for file_path in compatible_files:
            symlink_target = target_folder / Path(file_path).name
            if not target_folder.exists():
                target_folder.mkdir(parents=True, exist_ok=True)
            if symlink_target.exists():
                symlink_target.unlink()
            symlink_target.symlink_to(file_path)

def load_config():
    with open(Path("config.yaml"), 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def select_embedding_model_directory():
    initial_dir = Path('Embedding_Models') if Path('Embedding_Models').exists() else Path.home()
    chosen_directory = QFileDialog.getExistingDirectory(None, "Select Embedding Model Directory", str(initial_dir))
    
    if chosen_directory:
        config_file_path = Path("config.yaml")
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r', encoding='utf-8') as file:
                    config_data = yaml.safe_load(file)
            except Exception:
                config_data = {}

        config_data["EMBEDDING_MODEL_NAME"] = chosen_directory

        with open(config_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_data, file)

        print(f"Selected directory: {chosen_directory}")
