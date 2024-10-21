import json
import requests
import sseclient
import subprocess
import sys
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QCommandLinkButton, QMessageBox,
    QLabel, QApplication, QProgressDialog
)
from PySide6.QtCore import QThread, Signal, QObject, Qt
from PySide6.QtGui import QTextCursor, QPixmap

import platform
import signal
import os
import subprocess
import time

from constants import kobold_config, jeeves_system_message, rag_string
from database_interactions import QueryVectorDB

from pathlib import Path
import atexit

from huggingface_hub import snapshot_download

def has_discrete_vulkan_gpu():
    try:
        output = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True, check=True).stdout
        devicetypes = [line.split("=")[1].strip() for line in output.splitlines() if "deviceType" in line]
        return any(dtype == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" for dtype in devicetypes)
    except Exception as e:
        print(f"Error getting Vulkan information: {e}")
        return False

def create_kcppt_file(file_path, usecpu, model_param, **kwargs):
    config = kobold_config.copy()

    invalid_keys = []

    for key, value in kwargs.items():
        if key in config:
            config[key] = value
        else:
            invalid_keys.append(key)

    config['usecpu'] = usecpu

    config['model_param'] = Path(model_param).as_posix()

    try:
        with open(file_path, 'w') as file:
            json.dump(config, file, indent=2)

        if invalid_keys:
            print("Warning: The following keys are not valid and were not updated:")
            for key in invalid_keys:
                print(f"- {key}")
    except Exception as e:
        print(f"Failed to write KCPPT file: {e}")
        raise

def format_path(path: Path) -> str:
    return path.as_posix()

class ChatSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()

class ChatAPIWorker(QThread):
    def __init__(self, url, payload):
        super().__init__()
        self.url = url
        self.payload = payload
        self.signals = ChatSignals()
        self._is_running = True

    def run(self):
        try:
            with requests.Session() as session:
                with session.post(self.url, json=self.payload, stream=True, timeout=5) as response:
                    response.raise_for_status()
                    client = sseclient.SSEClient(response)

                    for event in client.events():
                        if not self._is_running:
                            break
                        if event.event == "message":
                            try:
                                data = json.loads(event.data)
                                if 'token' in data:
                                    token = data['token']
                                    self.signals.response_signal.emit(token)
                                else:
                                    self.signals.error_signal.emit(f"Unexpected data format: {data}")
                            except json.JSONDecodeError:
                                self.signals.error_signal.emit(f"Failed to parse: {event.data}")
                        time.sleep(0.01)
        except Exception as e:
            if self._is_running:
                self.signals.error_signal.emit(str(e))
        finally:
            self.signals.finished_signal.emit()

    def stop(self):
        self._is_running = False

class DownloadWorker(QThread):
    download_finished = Signal(bool, str)

    def __init__(self, repo_id, allow_patterns, local_dir):
        super().__init__()
        self.repo_id = repo_id
        self.allow_patterns = allow_patterns
        self.local_dir = local_dir

    def run(self):
        try:
            snapshot_download(repo_id=self.repo_id,
                             allow_patterns=self.allow_patterns,
                             local_dir=self.local_dir)
            self.download_finished.emit(True, "Download completed successfully.")
        except Exception as e:
            self.download_finished.emit(False, str(e))

class ChatWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ask Jeeves (Welcome back Jeeves!)")
        self.setGeometry(100, 100, 750, 850)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.chat_signals = ChatSignals()
        self.chat_signals.response_signal.connect(self.update_response)
        self.chat_signals.error_signal.connect(self.show_error_message)
        self.chat_signals.finished_signal.connect(self.on_submission_finished)

        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)

        image_path = Path(__file__).parent / "Assets" / "ask_jeeves_transparent.jpg"
        pixmap = self.load_image(image_path)
        if pixmap:
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(image_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True

        )

        self.chat_display.setPlainText("Hello, my name is Jeeves. Thank you for the job opportunity!  Ask me how to use this program.")

        self.layout.addWidget(self.chat_display, 4)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        self.layout.addWidget(self.input_field)

        self.setCentralWidget(central_widget)

        self.worker = None
        self.llm_is_active = False
        self.api_url = "http://localhost:5001/api/extra/generate/stream"

        self.vector_db = QueryVectorDB(selected_database="user_manual")

        self.initialize()

    def initialize(self):
        current_dir = Path(__file__).parent.resolve()
        assets_dir = current_dir / "Assets"
        assets_dir.mkdir(exist_ok=True)
        self.model_filename = "Llama-3.2-3B-Instruct-Q8_0.gguf"
        self.model_path = assets_dir / self.model_filename

        if not self.model_path.exists():
            image_path = current_dir / "Assets" / "ask_jeeves_transparent.jpg"
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                print(f"Failed to load image: {image_path}")
            else:
                pixmap = pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            msg_box = QMessageBox(self)
            msg_box.setIconPixmap(pixmap)
            msg_box.setWindowTitle("Jeeves is Unemployed")
            msg_box.setText("You have not yet hired Jeeves!")
            msg_box.setInformativeText("You must first download Jeeves to hire him.\n\nClick OK to download or Cancel to exit.")
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Ok)
            ret = msg_box.exec()

            if ret == QMessageBox.Ok:
                self.download_model(assets_dir)
            else:
                self.close()
        else:
            self.setup_subprocess(current_dir)


    def download_model(self, download_dir):
        self.progress_dialog = QProgressDialog("Downloading model...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Downloading")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        repo_id = "lmstudio-community/Llama-3.2-3B-Instruct-GGUF"
        allow_patterns = ["Llama-3.2-3B-Instruct-Q8_0.gguf"]

        self.download_worker = DownloadWorker(repo_id, allow_patterns, download_dir)
        self.download_worker.download_finished.connect(self.on_download_finished)
        self.download_worker.start()

    def on_download_finished(self, success, message):
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Download Successful", message)
            self.setup_subprocess(self.model_path.parent.parent)
        else:
            QMessageBox.critical(self, "Download Failed", f"Failed to download the model: {message}")
            self.close()

    def setup_subprocess(self, current_dir):
        try:
            creationflags = 0
            preexec_fn = None
            if platform.system() == 'Windows':
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            config_filename = "kobold_config.kcppt"
            assets_dir = current_dir / "Assets"
            assets_dir.mkdir(exist_ok=True)
            config_path = assets_dir / config_filename

            has_discrete_gpu = has_discrete_vulkan_gpu()

            usecpu = True if not has_discrete_gpu else False
            try:
                create_kcppt_file(
                    config_path,
                    usecpu=usecpu,
                    model_param=format_path(self.model_path),
                    # Add any other default settings you want to enforce here
                    # threads=8,  # Example additional parameter
                )

            except Exception as e:
                QMessageBox.critical(self, "Configuration Error", f"Failed to create KCPPT file: {e}")
                self.close()
                return

            exe_path = assets_dir / "koboldcpp_nocuda.exe"
            if not exe_path.exists():
                QMessageBox.critical(
                    self, 
                    "Subprocess Error", 
                    f"The executable '{exe_path}' was not found in the Assets directory '{assets_dir}'. Please ensure it is present."
                )
                self.close()
                return

            subprocess_command = [
                str(exe_path),
                "--config",
                format_path(config_path)
            ]

            self.process = subprocess.Popen(
                subprocess_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=creationflags,
                preexec_fn=preexec_fn
            )

            atexit.register(self.terminate_subprocess)

        except FileNotFoundError:
            QMessageBox.critical(self, "Subprocess Error", "koboldcpp_nocuda.exe not found.")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Subprocess Error", f"Failed to start subprocess: {e}")
            self.close()

    def terminate_subprocess(self):
        """Terminate the subprocess if it is still running."""
        if hasattr(self, 'process') and self.process.poll() is None:
            try:
                if platform.system() == 'Windows':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if platform.system() == 'Windows':
                        self.process.kill()
                    else:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception as e:
                QMessageBox.warning(self, "Subprocess Termination Error", f"Error terminating subprocess: {e}")

    def load_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Image Load Error", f"Failed to load image: {image_path}")
            return None
        return pixmap

    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message:
            QMessageBox.warning(self, "Input Error", "Please enter a message.")
            return

        self.chat_display.clear()
        self.input_field.clear()
        self.input_field.setDisabled(True)

        try:
            contexts, metadata = self.vector_db.search(user_message, k=8, score_threshold=0.9)
            if not contexts:
                QMessageBox.warning(self, "No Contexts Found", "No relevant contexts were found for your query.")
                self.input_field.setDisabled(False)
                return
        except Exception as e:
            QMessageBox.warning(self, "Database Query Error", f"An error occurred while querying the database: {e}")
            self.input_field.setDisabled(False)
            return

        contexts_text = "\n\n".join(contexts)
        full_context = f"{rag_string}\n\n{contexts_text}"

        new_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

{jeeves_system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

        combined_prompt = f"{full_context}\n\n{new_prompt}"

        payload = {
            "prompt": combined_prompt,
            "max_length": 512,
            "temperature": 0.0,
            # "top_p": 0.9,
            "stream": True,
            # "max_context_length": 4096,
            # "rep_pen": 1.0,
            # "rep_pen_range": 2048,
            # "rep_pen_slope": 0.7,
            # "tfs": 0.97,
            # "top_a": 0.8,
            # "top_k": 0,
            # "typical": 0.19,
        }

        self.worker = ChatAPIWorker(self.api_url, payload)
        self.worker.signals.response_signal.connect(self.update_response)
        self.worker.signals.error_signal.connect(self.show_error_message)
        self.worker.signals.finished_signal.connect(self.on_submission_finished)
        self.worker.start()

        self.chat_display.append("Jeeves:\n")
        self.llm_is_active = True

    def update_response(self, response_chunk):
        if self.llm_is_active:
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.insertPlainText(response_chunk)
        else:
            self.chat_display.append(f"Jeeves: {response_chunk}")
        self.chat_display.ensureCursorVisible()

    def show_error_message(self, error_message):
        QMessageBox.warning(self, "Error", f"An error occurred: {error_message}")
        self.input_field.setDisabled(False)
        self.llm_is_active = False

    def on_submission_finished(self):
        self.input_field.setDisabled(False)
        self.input_field.setFocus()
        self.llm_is_active = False
        if self.worker:
            self.worker.signals.response_signal.disconnect(self.update_response)
            self.worker.signals.error_signal.disconnect(self.show_error_message)
            self.worker.signals.finished_signal.disconnect(self.on_submission_finished)
            self.worker = None

    def closeEvent(self, event):
        self.terminate_subprocess()

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        if hasattr(self, 'vector_db'):
            self.vector_db.cleanup()

        super().closeEvent(event)