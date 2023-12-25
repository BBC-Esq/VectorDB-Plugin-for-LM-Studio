import os
import subprocess
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog


def choose_documents():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_folder = os.path.join(current_dir, "Docs_for_DB")
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_paths, _ = file_dialog.getOpenFileNames(
        None, "Choose Documents for Database", str(current_dir)
    )

    if file_paths:
        Path(docs_folder).mkdir(exist_ok=True)

        for file_path in file_paths:
            symlink_target = os.path.join(docs_folder, os.path.basename(file_path))
            Path(symlink_target).unlink(missing_ok=True)
            Path(symlink_target).symlink_to(Path(file_path))


def choose_directory():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_folder = os.path.join(current_dir, "Docs_for_DB")

    file_dialog = QFileDialog()
    file_dialog.setOption(QFileDialog.ShowDirsOnly)
    if file_dialog.exec():
        directories = file_dialog.selectedFiles()

        for selected_directory in directories:
            Path(docs_folder).mkdir(exist_ok=True)

            for root, dirs, files in os.walk(selected_directory):
                for file_name in files:
                    symlink_target = os.path.join(
                        docs_folder, os.path.basename(file_name)
                    )

                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, "r") as fp:
                            if isinstance(fp.read(), str):
                                Path(symlink_target).unlink(missing_ok=True)
                                Path(symlink_target).symlink_to(Path(file_path))
                    except UnicodeDecodeError:
                        pass


def see_documents_directory():
    current_dir = Path(__file__).parent.resolve()
    docs_folder = current_dir / "Docs_for_DB"

    docs_folder.mkdir(parents=True, exist_ok=True)

    # Cross-platform directory opening
    if os.name == "nt":  # Windows
        subprocess.Popen(f'explorer "{str(docs_folder)}"')
    elif os.name == "posix":  # MacOS and Linux
        subprocess.Popen(f'xdg-open "{str(docs_folder)}"')


if __name__ == "__main__":
    app = QApplication([])
    choose_documents()
    app.exec()
