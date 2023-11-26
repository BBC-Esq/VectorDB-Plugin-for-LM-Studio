from PySide6.QtWidgets import QVBoxLayout, QGroupBox, QHBoxLayout, QWidget
from gui_tabs_tools_transcribe import TranscriberToolSettingsTab

def adjust_stretch(groups, layout):
    for group, factor in groups.items():
        layout.setStretchFactor(group, factor if group.isChecked() else 0)

class GuiSettingsTab(QWidget):
    def __init__(self):
        super(GuiSettingsTab, self).__init__()
        self.layout = QVBoxLayout()

        # Dictionary that maps the title of the group box to the respective widget class and its stretch factor
        classes = {
            "Transcribe File Settings": (TranscriberToolSettingsTab, 1),
        }

        self.groups = {}

        # Create settings tabs and group boxes for each class
        for title, (TabClass, stretch) in classes.items():
            settings = TabClass()
            group = QGroupBox(title)
            layout = QVBoxLayout()
            layout.addWidget(settings)
            group.setLayout(layout)
            group.setCheckable(True)
            group.setChecked(True)
            self.groups[group] = stretch

            self.layout.addWidget(group, stretch)
            group.toggled.connect(lambda checked, group=group: (settings.setVisible(checked), adjust_stretch(self.groups, self.layout)))

        # Set the layout for the widget
        self.setLayout(self.layout)
        adjust_stretch(self.groups, self.layout)

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = GuiSettingsTab()
    window.show()
    sys.exit(app.exec())
