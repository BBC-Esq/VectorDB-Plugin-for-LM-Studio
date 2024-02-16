from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout, QMessageBox, QApplication
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QThread, Signal, QTimer
import base64
from constants import CHUNKS_ONLY_TOOLTIP, SPEAK_RESPONSE_TOOLTIP, IMAGE_STOP_SIGN
import server_connector
from voice_recorder_module import VoiceRecorder
from bark_module import BarkAudio
import threading
from pathlib import Path
from utilities import check_preconditions_for_submit_question

class SubmitButtonThread(QThread):
    responseSignal = Signal(str)
    errorSignal = Signal(str)
    finishedSignal = Signal()
    stop_requested = False

    def __init__(self, user_question, chunks_only, parent=None):
        super(SubmitButtonThread, self).__init__(parent)
        self.user_question = user_question
        self.chunks_only = chunks_only

    def run(self):
        try:
            response = server_connector.ask_local_chatgpt(self.user_question, self.chunks_only)
            for response_chunk in response:
                if SubmitButtonThread.stop_requested:
                    break
                self.responseSignal.emit(response_chunk)
            self.finishedSignal.emit()
        except Exception as err:
            self.errorSignal.emit("Connection to server failed. Please ensure the external server is running.")
            print(err)

    @classmethod
    def request_stop(cls):
        cls.stop_requested = True

class DatabaseQueryTab(QWidget):
    def __init__(self):
        super(DatabaseQueryTab, self).__init__()
        self.initWidgets()
        
    def initWidgets(self):
        layout = QVBoxLayout(self)

        self.read_only_text = QTextEdit()
        self.read_only_text.setReadOnly(True)
        layout.addWidget(self.read_only_text, 4)

        self.text_input = QTextEdit()
        layout.addWidget(self.text_input, 1)

        buttons_layout = QHBoxLayout()
        self.record_button = QPushButton("Record Question (click to record)")
        self.record_button.clicked.connect(self.toggle_recording)
        buttons_layout.addWidget(self.record_button)

        self.submit_button = QPushButton("Submit Question")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        buttons_layout.addWidget(self.submit_button)

        self.stop_button = QPushButton()
        stop_icon_pixmap = QPixmap()
        stop_icon_pixmap.loadFromData(base64.b64decode(IMAGE_STOP_SIGN))
        self.stop_button.setIcon(QIcon(stop_icon_pixmap))
        self.stop_button.clicked.connect(self.on_stop_button_clicked)
        self.stop_button.setDisabled(True)
        buttons_layout.addWidget(self.stop_button)
        layout.addLayout(buttons_layout)

        row_two_layout = QHBoxLayout()
        self.chunks_only_checkbox = QCheckBox("Chunks Only")
        self.chunks_only_checkbox.setToolTip(CHUNKS_ONLY_TOOLTIP)
        self.copy_response_button = QPushButton("Copy Response")
        self.copy_response_button.clicked.connect(self.on_copy_response_clicked)
        self.bark_button = QPushButton("Bark Response")
        self.bark_button.clicked.connect(self.on_bark_button_clicked)
        self.bark_button.setToolTip(SPEAK_RESPONSE_TOOLTIP)
        row_two_layout.addWidget(self.chunks_only_checkbox)
        row_two_layout.addWidget(self.copy_response_button)
        row_two_layout.addWidget(self.bark_button)
        layout.addLayout(row_two_layout)

        self.is_recording = False
        self.voice_recorder = VoiceRecorder(self)

    def on_submit_button_clicked(self):
        script_dir = Path(__file__).resolve().parent
        is_valid, error_message = check_preconditions_for_submit_question(script_dir)
        if not is_valid:
            QMessageBox.warning(self, "Error", error_message)
            self.submit_button.setDisabled(False)
            self.stop_button.setDisabled(True)
            return
        
        self.cumulative_response = ""
        self.submit_button.setDisabled(True)
        self.stop_button.setDisabled(False)
        user_question = self.text_input.toPlainText()
        chunks_only = self.chunks_only_checkbox.isChecked()
        self.submit_thread = SubmitButtonThread(user_question, chunks_only, self)
        self.submit_thread.responseSignal.connect(self.update_response)
        self.submit_thread.errorSignal.connect(self.show_error_message)
        self.submit_thread.finishedSignal.connect(self.on_submission_finished)
        self.submit_thread.start()

    def on_stop_button_clicked(self):
        SubmitButtonThread.request_stop()
        self.stop_button.setDisabled(True)

    def on_copy_response_clicked(self):
        clipboard = QApplication.clipboard()
        response_text = self.read_only_text.toPlainText()
        if response_text:
            clipboard.setText(response_text)
            QMessageBox.information(self, "Information", "Response copied to clipboard.")
        else:
            QMessageBox.warning(self, "Warning", "No response to copy.")

    def on_bark_button_clicked(self):
        script_dir = Path(__file__).resolve().parent
        chat_history_path = script_dir / 'chat_history.txt'
        if not chat_history_path.exists():
            QMessageBox.warning(self, "Error", "No response to play.")
            return
        bark_thread = threading.Thread(target=self.run_bark_module)
        bark_thread.daemon = True
        bark_thread.start()

    def run_bark_module(self):
        bark_audio = BarkAudio()
        bark_audio.run()

    def toggle_recording(self):
        if self.is_recording:
            self.voice_recorder.stop_recording()
            self.record_button.setText("Record Question (click to record)")
        else:
            self.voice_recorder.start_recording()
            self.record_button.setText("Recording... (click to stop)")
        self.is_recording = not self.is_recording

    def update_response(self, response_chunk):
        self.cumulative_response += response_chunk
        self.read_only_text.setPlainText(self.cumulative_response)
        self.submit_button.setDisabled(False)

    def show_error_message(self, error_message):
        QMessageBox.warning(self, "Error", error_message)
        self.submit_button.setDisabled(False)
        self.stop_button.setDisabled(True)

    def on_submission_finished(self):
        self.submit_button.setDisabled(False)
        self.stop_button.setDisabled(True)

    def update_transcription(self, transcription_text):
        self.text_input.setPlainText(transcription_text)
