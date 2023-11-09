from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog
from PySide6.QtCore import Qt
import ast
import os
import sys
import subprocess

def find_imports(filename):
    directory = os.path.dirname(filename)
    all_py_files = [f[:-3] for f in os.listdir(directory) if f.endswith('.py')]

    with open(filename, 'r') as f:
        tree = ast.parse(f.read())

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name in all_py_files:
                    imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module in all_py_files:
                for n in node.names:
                    imports.append(f"{module}.{n.name}")

    with open('tree.txt', 'w') as f:
        for imp in imports:
            f.write(f"{imp}\n")

    # Automatically open the text file
    if sys.platform.startswith('darwin'):  # macOS
        subprocess.run(['open', 'tree.txt'])
    elif sys.platform.startswith('win'):  # Windows
        subprocess.run(['notepad', 'tree.txt'])
    else:  # Linux and other OS
        subprocess.run(['xdg-open', 'tree.txt'])

def select_file():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(None, "Select Python file", "", "Python Files (*.py);;All Files (*)", options=options)
    if file_name:
        find_imports(file_name)

app = QApplication([])

window = QWidget()
window.setWindowTitle('Import Tree Analyzer')
window.setGeometry(200, 200, 400, 200)

layout = QVBoxLayout()

button = QPushButton('Select Python File')
button.clicked.connect(select_file)
layout.addWidget(button)

window.setLayout(layout)
window.show()

sys.exit(app.exec())
