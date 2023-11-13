from PySide6.QtWidgets import QPushButton, QLabel, QHBoxLayout, QWidget, QCheckBox
from PySide6.QtCore import Qt
import voice_recorder_module

def create_button_row(submit_handler):

    voice_recorder = voice_recorder_module.VoiceRecorder()

    def start_recording():
        voice_recorder.start_recording()

    def stop_recording():
        voice_recorder.stop_recording()

    start_button = QPushButton("Start Recording")
    start_button.clicked.connect(start_recording)

    stop_button = QPushButton("Stop Recording")
    stop_button.clicked.connect(stop_recording)

    hbox = QHBoxLayout()
    hbox.addWidget(start_button)
    hbox.addWidget(stop_button)

    hbox.setStretchFactor(start_button, 3)
    hbox.setStretchFactor(stop_button, 3)

    row_widget = QWidget()
    row_widget.setLayout(hbox)

    return row_widget
