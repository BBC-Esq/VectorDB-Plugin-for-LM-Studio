from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

class ModelsSettingsTab(QWidget):
    def __init__(self):
        super(ModelsSettingsTab, self).__init__()

        self.label = QLabel("Embedding model settings, which are coming soon.")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
