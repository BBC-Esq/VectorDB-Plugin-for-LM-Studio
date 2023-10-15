from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QMessageBox
from PySide6.QtGui import QIntValidator
import yaml

class ServerSettingsTab(QWidget):
    def __init__(self):
        super(ServerSettingsTab, self).__init__()

        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)
            self.connection_str = config_data.get('server', {}).get('connection_str', '')
            self.current_port = self.connection_str.split(":")[-1].split("/")[0]
            self.current_max_tokens = config_data.get('server', {}).get('model_max_tokens', '')

        # Server port setting widgets
        self.port_label = QLabel(f"Current Port: {self.current_port}")
        self.new_port_edit = QLineEdit()
        self.new_port_edit.setPlaceholderText("Enter new port...")
        self.new_port_edit.setValidator(QIntValidator())

        # Max token setting widgets
        self.max_tokens_label = QLabel(f"Max Tokens: {self.current_max_tokens}")
        self.new_max_tokens_edit = QLineEdit()
        self.new_max_tokens_edit.setPlaceholderText("Enter new max tokens...")
        self.new_max_tokens_edit.setValidator(QIntValidator())

        # Update config button
        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_config)

        layout = QGridLayout()
        layout.addWidget(self.port_label, 0, 0)
        layout.addWidget(self.new_port_edit, 0, 1)
        layout.addWidget(self.max_tokens_label, 1, 0)
        layout.addWidget(self.new_max_tokens_edit, 1, 1)
        layout.addWidget(self.update_button, 2, 1)

        self.setLayout(layout)

    def update_config(self):
        new_port = self.new_port_edit.text()
        new_max_tokens = self.new_max_tokens_edit.text()

        # Update config.yaml if at least one setting was changed - otherwise display message
        if not new_port and not new_max_tokens:
            QMessageBox.warning(self, 'No Updates', 'No new values were entered.')
            return

        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)

        if new_port:
            config_data['server']['connection_str'] = self.connection_str.replace(self.current_port, new_port)
            self.port_label.setText(f"Current Port: {new_port}")
        
        if new_max_tokens:
            config_data['server']['model_max_tokens'] = int(new_max_tokens)
            self.max_tokens_label.setText(f"Max Tokens: {new_max_tokens}")

        with open('config.yaml', 'w') as file:
            yaml.safe_dump(config_data, file)
