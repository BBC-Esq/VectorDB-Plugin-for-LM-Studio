from PySide6.QtWidgets import QPushButton, QLabel, QHBoxLayout, QWidget, QCheckBox
from PySide6.QtCore import Qt
import voice_recorder_module

def create_button_row(submit_handler, style):

    voice_recorder = voice_recorder_module.VoiceRecorder()

    def start_recording():
        voice_recorder.start_recording()

    def stop_recording():
        voice_recorder.stop_recording()

    # submit_button = QPushButton("Submit Question")
    # submit_button.setStyleSheet(style)

    start_button = QPushButton("Start Recording")
    start_button.setStyleSheet(style)
    start_button.clicked.connect(start_recording)

    stop_button = QPushButton("Stop Recording")
    stop_button.setStyleSheet(style)
    stop_button.clicked.connect(stop_recording)

    # disabled_label = QLabel("Disabled")
    # disabled_checkbox = QCheckBox()

    # def toggle_checkbox():
    #     disabled_checkbox.setChecked(not disabled_checkbox.isChecked())

    # disabled_label.mousePressEvent = lambda event: toggle_checkbox()

    hbox = QHBoxLayout()
    # hbox.addWidget(submit_button)
    hbox.addWidget(start_button)
    hbox.addWidget(stop_button)
    # hbox.addWidget(disabled_label)
    # hbox.addWidget(disabled_checkbox)

    # hbox.setStretchFactor(submit_button, 7)
    hbox.setStretchFactor(start_button, 3)
    hbox.setStretchFactor(stop_button, 3)
    # hbox.setStretchFactor(disabled_label, 1)
    # hbox.setStretchFactor(disabled_checkbox, 1)

    row_widget = QWidget()
    row_widget.setLayout(hbox)

    return row_widget
