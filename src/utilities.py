import os
from PySide6.QtWidgets import QApplication

def validate_symbolic_links(source_directory):
    symbolic_links = [entry for entry in os.listdir(source_directory) if os.path.islink(os.path.join(source_directory, entry))]

    for symlink in symbolic_links:
        symlink_path = os.path.join(source_directory, symlink)
        target_path = os.readlink(symlink_path)

        if not os.path.exists(target_path):
            print(f"Warning: Symbolic link {symlink} points to a missing file. It will be skipped.")
            os.remove(symlink_path)

def load_stylesheet(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stylesheet_path = os.path.join(script_dir, 'CSS', filename)
    with open(stylesheet_path, 'r') as file:
        stylesheet = file.read()
    return stylesheet

def list_theme_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    theme_dir = os.path.join(script_dir, 'CSS')
    return [f for f in os.listdir(theme_dir) if f.endswith('.css')]

def make_theme_changer(theme_name):
    def change_theme():
        stylesheet = load_stylesheet(theme_name)
        QApplication.instance().setStyleSheet(stylesheet)
    return change_theme

if __name__ == '__main__':
    source_directory = "Docs_for_DB"
    validate_symbolic_links(source_directory)