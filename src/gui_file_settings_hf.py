from pathlib import Path
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
import yaml
import logging
import traceback

from utilities import my_cprint

def set_hf_access_token(parent_widget):
    try:
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle("Hugging Face Access Token")

        layout = QVBoxLayout(dialog)

        label = QLabel("Enter a new Hugging Face access token or clear the current one:", dialog)
        layout.addWidget(label)

        token_input = QLineEdit(dialog)
        layout.addWidget(token_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        clear_button = QPushButton("Clear Token")
        button_box.addButton(clear_button, QDialogButtonBox.ActionRole)

        layout.addWidget(button_box)

        config_file_path = Path(__file__).parent / 'config.yaml'
        if config_file_path.exists():
            with open(config_file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        else:
            config = {}

        def save_token():
            token = token_input.text()
            if token:
                config['hf_access_token'] = token
                with open(config_file_path, 'w', encoding='utf-8') as file:
                    yaml.safe_dump(config, file, allow_unicode=True)
                QMessageBox.information(parent_widget, "Success", "Hugging Face access token saved successfully.")
                my_cprint("Hugging Face access token updated successfully.", "green")
            dialog.accept()

        def clear_token():
            config['hf_access_token'] = None
            with open(config_file_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(config, file, allow_unicode=True)
            QMessageBox.information(parent_widget, "Success", "Hugging Face access token cleared successfully.")
            my_cprint("Hugging Face access token cleared.", "green")
            dialog.accept()

        button_box.accepted.connect(save_token)
        button_box.rejected.connect(dialog.reject)
        clear_button.clicked.connect(clear_token)

        dialog.exec_()

    except Exception as e:
        logging.error(f"Error updating Hugging Face access token: {str(e)}")
        logging.debug(traceback.format_exc())
        QMessageBox.critical(parent_widget, "Error", f"Failed to save or clear Hugging Face access token: {str(e)}")
