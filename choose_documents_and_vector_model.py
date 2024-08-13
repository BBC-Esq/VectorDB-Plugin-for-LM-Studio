import subprocess
from pathlib import Path
import yaml
from PySide6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QMessageBox
import torch

def check_cuda_for_images(files):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'}
    if any(Path(file).suffix.lower() in image_extensions for file in files):
        if not torch.cuda.is_available():
            QMessageBox.warning(None, "CUDA Support Required", 
                "Processing images currently only available with GPU acceleration. Please remove any images and try again.")
            return False
    return True

def choose_documents_directory():
    allowed_extensions = {'.pdf', '.docx', '.epub', '.txt', '.enex', '.eml', '.msg', '.csv', '.xls', '.xlsx', 
                          '.rtf', '.odt', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.html', 
                          '.htm', '.md', '.doc'}
    current_dir = Path(__file__).parent.resolve()
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_paths, _ = file_dialog.getOpenFileNames(None, "Choose Documents and Images for Database", str(current_dir))

    if file_paths:
        if not check_cuda_for_images(file_paths):
            return

        compatible_files = [file for file in file_paths if Path(file).suffix.lower() in allowed_extensions]
        incompatible_files = [Path(file).name for file in file_paths if Path(file).suffix.lower() not in allowed_extensions]

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
        target_folder.mkdir(parents=True, exist_ok=True)
        for file_path in compatible_files:
            symlink_target = target_folder / Path(file_path).name
            symlink_target.unlink(missing_ok=True)
            symlink_target.symlink_to(file_path)

def load_config():
    with open("config.yaml", 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def select_embedding_model_directory():
    initial_dir = Path('Models') if Path('Models').exists() else Path.home()
    chosen_directory = QFileDialog.getExistingDirectory(None, "Select Embedding Model Directory", str(initial_dir))
    
    if chosen_directory:
        config_file_path = Path("config.yaml")
        config_data = yaml.safe_load(config_file_path.read_text(encoding='utf-8')) if config_file_path.exists() else {}
        config_data["EMBEDDING_MODEL_NAME"] = chosen_directory
        config_file_path.write_text(yaml.dump(config_data), encoding='utf-8')
        print(f"Selected directory: {chosen_directory}")