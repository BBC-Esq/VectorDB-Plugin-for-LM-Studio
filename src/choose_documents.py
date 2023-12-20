import os
import sys
import subprocess
from PySide6.QtWidgets import QApplication, QFileDialog


def choose_documents_directory():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_folder = os.path.join(current_dir, "Docs_for_DB")
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_paths, _ = file_dialog.getOpenFileNames(
        None, "Choose Documents for Database", current_dir
    )

    if file_paths:
        if not os.path.exists(docs_folder):
            os.mkdir(docs_folder)

        for file_path in file_paths:
            symlink_target = os.path.join(docs_folder, os.path.basename(file_path))
            os.symlink(file_path, symlink_target)


def see_documents_directory():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_folder = os.path.join(current_dir, "Docs_for_DB")

    # Ensure the directory exists
    if not os.path.exists(docs_folder):
        os.mkdir(docs_folder)

    if sys.platform == "win32":
        # Open the directory in Windows File Explorer
        subprocess.Popen(f'explorer "{docs_folder}"')
    elif sys.platform == "darwin":
        # Open the directory in MacOS Finder
        subprocess.Popen(f'open "{docs_folder}"', shell=True)


if __name__ == "__main__":
    app = QApplication([])
    choose_documents_directory()
    app.exec()
