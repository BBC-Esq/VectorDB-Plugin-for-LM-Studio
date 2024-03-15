from PySide6.QtWidgets import QVBoxLayout, QGroupBox, QHBoxLayout, QWidget
from gui_tabs_tools_transcribe import TranscriberToolSettingsTab
from gui_tabs_tools_vision import VisionToolSettingsTab

def adjust_stretch(groups, layout):
    for group, factor in groups.items():
        layout.setStretchFactor(group, factor if group.isChecked() else 0)

class GuiSettingsTab(QWidget):
    def __init__(self):
        super(GuiSettingsTab, self).__init__()
        self.layout = QVBoxLayout()

        classes = {
            "TRANSCRIBE FILE": (TranscriberToolSettingsTab, 1),
            "TEST PROCESS IMAGE": (VisionToolSettingsTab, 4),
        }

        self.groups = {}

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

        self.setLayout(self.layout)
        adjust_stretch(self.groups, self.layout)
