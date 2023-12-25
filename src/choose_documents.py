import subprocess
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
import sys

def choose_documents_directory():
    current_dir = Path(__file__).parent.resolve()
    docs_folder = current_dir / "Docs_for_DB"
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_paths, _ = file_dialog.getOpenFileNames(None, "Choose Documents for Database", str(current_dir))

    if file_paths:
        docs_folder.mkdir(parents=True, exist_ok=True)

        for file_path in file_paths:
            symlink_target = docs_folder / Path(file_path).name
            symlink_target.symlink_to(file_path)

def see_documents_directory():
    current_dir = Path(__file__).parent.resolve()
    docs_folder = current_dir / "Docs_for_DB"

    docs_folder.mkdir(parents=True, exist_ok=True)

    # Cross-platform directory opening
    if os.name == 'nt':  # Windows
        subprocess.Popen(['explorer', str(docs_folder)])
    elif sys.platform == 'darwin':  # macOS
        subprocess.Popen(['open', str(docs_folder)])
    elif sys.platform.startswith('linux'):  # Linux
        subprocess.Popen(['xdg-open', str(docs_folder)])

if __name__ == '__main__':
    app = QApplication([])
    choose_documents_directory()
    app.exec_()
