import os
import subprocess
from PySide6.QtWidgets import QApplication


def see_documents_directory():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_folder = os.path.join(current_dir, "Docs_for_DB")

    # Ensure the directory exists
    if not os.path.exists(docs_folder):
        os.mkdir(docs_folder)

    # Open the directory in Windows File Explorer
    subprocess.Popen(f'explorer "{docs_folder}"')


if __name__ == '__main__':
    app = QApplication([])
    see_documents_directory()
    app.exec_()
