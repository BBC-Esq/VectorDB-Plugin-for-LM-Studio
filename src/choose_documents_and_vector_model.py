from pathlib import Path
import yaml
from PySide6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

# Constants
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.epub', '.txt', '.enex', '.eml', '.msg', '.csv', '.xls', '.xlsx', 
                     '.rtf', '.odt', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.html', 
                     '.htm', '.md', '.doc'}
DOCS_FOLDER = "Docs_for_DB"
CONFIG_FILE = "config.yaml"

def choose_documents_directory():
    current_dir = Path(__file__).parent.resolve()
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_paths = file_dialog.getOpenFileNames(None, "Choose Documents and Images for Database", str(current_dir))[0]
    if file_paths:
        compatible_files = [file for file in file_paths if Path(file).suffix.lower() in ALLOWED_EXTENSIONS]
        incompatible_files = [Path(file).name for file in file_paths if Path(file).suffix.lower() not in ALLOWED_EXTENSIONS]
        if incompatible_files:
            if not show_incompatible_files_dialog(incompatible_files):
                return
        target_folder = current_dir / DOCS_FOLDER
        target_folder.mkdir(parents=True, exist_ok=True)
        for file_path in compatible_files:
            symlink_target = target_folder / Path(file_path).name
            symlink_target.unlink(missing_ok=True)
            symlink_target.symlink_to(file_path)

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