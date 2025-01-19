from pathlib import Path
import yaml
from PySide6.QtWidgets import (QFileDialog, QDialog, QVBoxLayout, QTextEdit, 
                              QPushButton, QHBoxLayout, QMessageBox)
from create_symlinks import create_symlinks_parallel

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.epub', '.txt', '.enex', '.eml', '.msg', '.csv', '.xls', '.xlsx', 
                     '.rtf', '.odt', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.html', 
                     '.htm', '.md', '.doc'}
DOCS_FOLDER = "Docs_for_DB"
CONFIG_FILE = "config.yaml"

def choose_documents_directory():
    current_dir = Path(__file__).parent.resolve()
    target_dir = current_dir / DOCS_FOLDER
    target_dir.mkdir(parents=True, exist_ok=True)

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Selection Type")
    msg_box.setText("Would you like to select a directory or individual files?")
    dir_button = msg_box.addButton("Select Directory", QMessageBox.ActionRole)
    files_button = msg_box.addButton("Select Files", QMessageBox.ActionRole)
    cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

    msg_box.exec()
    clicked_button = msg_box.clickedButton()

    if clicked_button == cancel_button:
        return

    file_dialog = QFileDialog()

    if clicked_button == dir_button:
        # Directory selection mode
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        selected_dir = file_dialog.getExistingDirectory(None, "Choose Directory for Database", str(current_dir))
        if selected_dir:
            selected_dir_path = Path(selected_dir)
            compatible_files = []
            incompatible_files = []

            for file_path in selected_dir_path.iterdir():
                if file_path.is_file():
                    if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                        compatible_files.append(str(file_path))
                    else:
                        incompatible_files.append(file_path.name)

            if incompatible_files:
                if not show_incompatible_files_dialog(incompatible_files):
                    return

            if compatible_files:
                try:
                    count, errors = create_symlinks_parallel(compatible_files, target_dir)
                    if errors:
                        print("Errors occurred while creating symlinks:", errors)
                except Exception as e:
                    print(f"Error creating symlinks: {e}")
    else:
        # File selection mode
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_paths = file_dialog.getOpenFileNames(None, "Choose Documents and Images for Database", str(current_dir))[0]
        if file_paths:
            compatible_files = []
            incompatible_files = []

            for file_path in file_paths:
                path = Path(file_path)
                if path.suffix.lower() in ALLOWED_EXTENSIONS:
                    compatible_files.append(str(path))
                else:
                    incompatible_files.append(path.name)

            if incompatible_files:
                if not show_incompatible_files_dialog(incompatible_files):
                    return

            if compatible_files:
                try:
                    count, errors = create_symlinks_parallel(compatible_files, target_dir)
                    if errors:
                        print("Errors occurred while creating symlinks:", errors)
                except Exception as e:
                    print(f"Error creating symlinks: {e}")

def show_incompatible_files_dialog(incompatible_files):
    dialog_text = (
        "The following files cannot be added here due to their file extension:\n\n" +
        "\n".join(incompatible_files) +
        "\n\nHowever, if any of them are audio files you can still add them directly in the Tools Tab."
        "\n\nClick 'Ok' to add the compatible documents only (remembering to add audio files separately) or 'Cancel' to back out completely."
    )
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
    return incompatible_dialog.exec() == QDialog.Accepted

def load_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def select_embedding_model_directory():
    initial_dir = Path('Models') if Path('Models').exists() else Path.home()
    chosen_directory = QFileDialog.getExistingDirectory(None, "Select Embedding Model Directory", str(initial_dir))

    if chosen_directory:
        config_file_path = Path(CONFIG_FILE)
        config_data = yaml.safe_load(config_file_path.read_text(encoding='utf-8')) if config_file_path.exists() else {}
        config_data["EMBEDDING_MODEL_NAME"] = chosen_directory
        config_file_path.write_text(yaml.dump(config_data), encoding='utf-8')
        print(f"Selected directory: {chosen_directory}")