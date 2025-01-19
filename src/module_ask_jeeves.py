import json
import subprocess
import platform
import signal
import os
import time
import atexit
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import partial
import socket

from ctypes import windll, byref, sizeof, c_void_p, c_int
from ctypes.wintypes import BOOL, HWND, DWORD

import requests
import sseclient
from huggingface_hub import snapshot_download
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSizePolicy,
    QTextEdit, QLineEdit, QMessageBox, QVBoxLayout, QPushButton,
    QLabel, QCommandLinkButton, QHBoxLayout, QApplication, QProgressDialog
)
from PySide6.QtCore import QThread, Signal, QObject, Qt, QTimer
from PySide6.QtGui import QTextCursor, QPixmap

from database_interactions import QueryVectorDB
from constants import (
    kobold_config, 
    jeeves_system_message, 
    rag_string,
    JEEVES_MODELS,
    master_questions,
    CustomButtonStyles,
)


def cleanup_existing_processes():
    try:
        if platform.system() == 'Windows':
            output = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq koboldcpp_nocuda.exe']).decode()
            if 'koboldcpp_nocuda.exe' in output:
                subprocess.run(['taskkill', '/F', '/IM', 'koboldcpp_nocuda.exe'], capture_output=True)
                time.sleep(2)
                print("Cleaned up existing KoboldCPP processes")
        else:
            subprocess.run(['pkill', '-f', 'koboldcpp_nocuda'], capture_output=True)
    except Exception as e:
        print(f"Error cleaning up processes: {e}")

def has_discrete_vulkan_gpu():
    try:
        output = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True, check=True).stdout
        devicetypes = [line.split("=")[1].strip() for line in output.splitlines() if "deviceType" in line]
        return any(dtype == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" for dtype in devicetypes)
    except Exception as e:
        print(f"Error getting Vulkan information: {e}")
        return False

def wait_for_server(timeout=30, process=None):
    start_time = time.time()
    attempt_count = 0
    
    while time.time() - start_time < timeout:
        try:
            if process and process.poll() is not None:
                stdout, stderr = process.communicate()
                print("\nProcess terminated unexpectedly!")
                print("STDOUT:", stdout.decode() if stdout else "None")
                print("STDERR:", stderr.decode() if stderr else "None")
                cleanup_existing_processes()
                return False
                
            attempt_count += 1
            print(f"Attempt {attempt_count}: Checking if server is running...")
            
            response = requests.get("http://localhost:5001/api/v1/model")
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                print(f"\nServer ready! Startup took {elapsed_time:.2f} seconds")
                return True
        except requests.exceptions.RequestException:
            if process:
                if process.stdout.peek():
                    process.stdout.readline()
                if process.stderr.peek():
                    process.stderr.readline()
            time.sleep(1)
            
    print(f"\nServer startup timed out after {timeout} seconds")
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
        self.response = None

    def run(self):
        try:
            with requests.Session() as session:
                self.response = session.post(
                    self.url,
                    json=self.payload,
                    stream=True,
                    timeout=30
                )
                self.response.raise_for_status()
                client = sseclient.SSEClient(self.response)

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
        except Exception as e:
            print(f"Stream error: {str(e)}")
            if self._is_running:
                self.signals.error_signal.emit(str(e))
        finally:
            if self.response:
                self.response.close()
            self.signals.finished_signal.emit()

    def stop(self):
        self._is_running = False
        if self.response:
            self.response.close()

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

class ServerStartupWorker(QThread):
    server_started = Signal(object)
    server_failed = Signal(str)

    def __init__(self, assets_dir, model_path):
        super().__init__()
        self.assets_dir = assets_dir
        self.model_path = model_path
        self.process = None

    def run(self):
        try:
            creationflags = 0
            preexec_fn = None
            if platform.system() == 'Windows':
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            config_filename = "kobold_config.kcppt"
            config_path = self.assets_dir / config_filename

            has_discrete_gpu = has_discrete_vulkan_gpu()
            usecpu = not has_discrete_gpu

            try:
                create_kcppt_file(
                    config_path,
                    usecpu=usecpu,
                    model_param=format_path(self.model_path),
                )
            except Exception as e:
                self.server_failed.emit(f"Failed to create KCPPT file: {e}")
                return

            exe_path = self.assets_dir / "koboldcpp_nocuda.exe"
            if not exe_path.exists():
                self.server_failed.emit(f"The executable '{exe_path}' was not found in the Assets directory '{self.assets_dir}'. Please ensure it is present.")
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

            print("Waiting for server to be ready...")
            if not wait_for_server(process=self.process):
                self.server_failed.emit("Kobold server failed to start within timeout period")
                self.terminate_subprocess()
                return

            atexit.register(self.terminate_subprocess)
            self.server_started.emit(self.process)

        except FileNotFoundError:
            self.server_failed.emit("koboldcpp_nocuda.exe not found.")
        except Exception as e:
            self.server_failed.emit(f"Failed to start subprocess: {e}")

    def terminate_subprocess(self):
        if self.process and self.process.poll() is None:
            print("Firing Jeeves...poor guy...")
            try:
                if platform.system() == 'Windows':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    stdout, stderr = self.process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    if platform.system() == 'Windows':
                        self.process.kill()
                    else:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    stdout, stderr = self.process.communicate()
                    print("Subprocess terminated")
            except Exception as e:
                print(f"Error firing Jeeves...looks like he'll be around for awhile...: {e}")

class ChatWindow(QMainWindow):
    def __init__(self, parent=None, selected_model=None):
        super().__init__(parent)
        self.selected_model = selected_model or "EXAONE 3.5"
        self.setWindowTitle("Ask Jeeves (Welcome back Jeeves!)")
        self.setGeometry(100, 100, 850, 950)

        self.model = SentenceTransformer('ibm-granite/granite-embedding-30m-english')
        self.question_embeddings = self.model.encode(master_questions)
        self.suggestion_cache = {}
        
        self.chat_signals = ChatSignals()
        self.chat_signals.response_signal.connect(self.update_response)
        self.chat_signals.error_signal.connect(self.show_error_message)
        self.chat_signals.finished_signal.connect(self.on_submission_finished)

        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(1)

        image_path = Path(__file__).parent / "Assets" / "ask_jeeves_transparent.jpg"
        pixmap = self.load_image(image_path)
        if pixmap:
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(image_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlainText("Hello, my name is Jeeves. Thank you for the job opportunity! Ask me how to use this program.")
        self.layout.addWidget(self.chat_display, 4)

        # Setup input field
        self.input_field = QLineEdit()
        self.input_field.setFixedHeight(40)
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.textChanged.connect(self.debounce_update)
        self.layout.addWidget(self.input_field)

        # Replace the suggestion buttons section with this:
        self.suggestion_widget = QWidget()
        self.suggestion_widget.setMinimumHeight(100)
        self.suggestion_layout = QVBoxLayout(self.suggestion_widget)
        self.suggestion_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestion_layout.setSpacing(1)

        self.suggestion_buttons = []
        for _ in range(3):
            btn = QPushButton()
            btn.setVisible(True)
            btn.setStyleSheet(CustomButtonStyles.TEAL_BUTTON_STYLE)

            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumSize(200, 35)  # Set minimum size explicitly

            btn.clicked.connect(self.on_suggestion_clicked)
            btn.setStyleSheet("text-align: left; padding: 1px 14px;")

            self.suggestion_buttons.append(btn)
            self.suggestion_layout.addWidget(btn)

        self.suggestion_layout.addStretch()
        self.layout.addWidget(self.suggestion_widget)

        # Timer setup for debounce
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._delayed_update)

        self.setCentralWidget(central_widget)

        self.worker = None
        self.llm_is_active = False
        self.api_url = "http://localhost:5001/api/extra/generate/stream"
        self.vector_db = QueryVectorDB(selected_database="user_manual")
        self.initialize()

    def showEvent(self, event):
        super().showEvent(event)
        self.apply_dark_mode_settings()

    def apply_dark_mode_settings(self):
        DWMWA_USE_IMMERSIVE_DARK_MODE = DWORD(20)
        set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
        hwnd = HWND(int(self.winId()))
        true_bool = BOOL(True)
        set_window_attribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            byref(true_bool),
            sizeof(true_bool)
        )

        # Make window border black (Windows 11+)
        DWMWA_BORDER_COLOR = DWORD(34)
        black_color = c_int(0xFF000000)  # ABGR = 0xAARRGGBB => FF 00 00 00 = fully opaque black
        set_window_attribute(
            hwnd,
            DWMWA_BORDER_COLOR,
            byref(black_color),
            sizeof(black_color)
        )

    def find_top_similar(self, input_text, top_k=3):
        if not input_text.strip() or len(input_text) < 3:
            return []
            
        input_embedding = self.model.encode([input_text])[0]
        similarities = np.dot(self.question_embeddings, input_embedding) / (
            np.linalg.norm(self.question_embeddings, axis=1) * np.linalg.norm(input_embedding)
        )
        
        # Get indices of top_k similar questions
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter out questions with low similarity
        threshold = 0.6
        top_questions = [
            master_questions[idx] for idx, sim in zip(top_indices, top_similarities) if sim > threshold
        ]
        
        return top_questions

    def debounce_update(self, text):
        self.current_text = text
        self.timer.start(500)

    def _delayed_update(self):
        text = self.current_text
        if len(text) >= 3:
            suggestions = self.find_top_similar(text, top_k=3)
            self.update_suggestions(suggestions)
        else:
            self.clear_suggestions()

    def update_suggestions(self, suggestions):
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                btn.setText(suggestions[i])
                btn.setEnabled(True)
            else:
                btn.setText("")
                btn.setEnabled(False)

    def clear_suggestions(self):
        for btn in self.suggestion_buttons:
            btn.setText("")
            btn.setEnabled(False)

    def on_suggestion_clicked(self):
        sender = self.sender()
        if sender and isinstance(sender, QPushButton):
            suggestion = sender.text()
            self.input_field.setText(suggestion)
            self.send_message()

    def initialize(self):
        current_dir = Path(__file__).parent.resolve()
        assets_dir = current_dir / "Assets"
        assets_dir.mkdir(exist_ok=True)

        cleanup_existing_processes()

        model_config = JEEVES_MODELS[self.selected_model]
        self.model_filename = model_config["filename"]
        self.model_path = assets_dir / self.model_filename

        self.input_field.setDisabled(True)

        if not self.model_path.exists():
            image_path = current_dir / "Assets" / "ask_jeeves_transparent.jpg"
            pixmap = self.load_image(image_path)
            if pixmap:
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
            self.start_server_in_background(assets_dir, self.model_path)

    def is_port_available(self, port, kill_existing=True):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', port))
            if kill_existing:
                cleanup_existing_processes()
                time.sleep(2)
                return self.is_port_available(port, kill_existing=False)  # Also add self here
            return False
        except ConnectionRefusedError:
            return True

    def start_server_in_background(self, assets_dir, model_path):
        if not self.is_port_available(5001):
            QMessageBox.warning(self, "Server Error", 
                "Failed to clean up existing processes. Please restart your computer.")
            self.close()
            return
            
        self.server_worker = ServerStartupWorker(assets_dir, model_path)
        self.server_worker.server_started.connect(self.on_server_started)
        self.server_worker.server_failed.connect(self.on_server_failed)
        self.server_worker.start()

    def on_server_started(self, process):
        self.process = process
        self.input_field.setDisabled(False)

    def on_server_failed(self, error_message):
        QMessageBox.critical(self, "Server Error", error_message)
        self.close()

    def download_model(self, download_dir):
        self.progress_dialog = QProgressDialog("Downloading model...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Downloading")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        model_config = JEEVES_MODELS[self.selected_model]
        repo_id = model_config["repo_id"]
        allow_patterns = model_config["allow_patterns"]

        self.download_worker = DownloadWorker(repo_id, allow_patterns, download_dir)
        self.download_worker.download_finished.connect(self.on_download_finished)
        self.download_worker.start()

    def on_download_finished(self, success, message):
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Download Successful", message)
            self.start_server_in_background(self.model_path.parent.parent, self.model_path)
        else:
            QMessageBox.critical(self, "Download Failed", f"Failed to download the model: {message}")
            self.close()

    def load_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Image Load Error", f"Failed to load image: {image_path}")
            return None
        return pixmap

    def send_message(self):
        if not hasattr(self, 'process') or self.process.poll() is not None:
            QMessageBox.warning(self, "Server Not Ready", "The server is not ready yet. Please wait a moment.")
            return

        user_message = self.input_field.text().strip()
        if not user_message:
            QMessageBox.warning(self, "Input Error", "Please enter a message.")
            return

        self.chat_display.clear()
        self.input_field.clear()
        self.input_field.setDisabled(True)

        try:
            contexts, metadata = self.vector_db.search(user_message, k=7, score_threshold=0.9)
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

        model_config = JEEVES_MODELS[self.selected_model]
        new_prompt = model_config["prompt_template"].format(
            jeeves_system_message=jeeves_system_message,
            user_message=user_message
        )

        combined_prompt = f"{full_context}\n\n{new_prompt}"

        payload = {
            "prompt": combined_prompt,
            "max_length":1024,
            "temperature": 0.0,
            "stream": True,
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

    def terminate_subprocess(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            print("Terminating Jeeves subprocess...")
            try:
                if platform.system() == 'Windows':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    stdout, stderr = self.process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    if platform.system() == 'Windows':
                        self.process.kill()
                    else:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    stdout, stderr = self.process.communicate()
                    print("Subprocess terminated")
            except Exception as e:
                print(f"Error terminating subprocess: {e}")
                QMessageBox.warning(self, "Subprocess Termination Error", f"Error terminating subprocess: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'server_worker') and self.server_worker.isRunning():
            self.server_worker.wait()

        self.terminate_subprocess()

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        if hasattr(self, 'vector_db'):
            self.vector_db.cleanup()

        super().closeEvent(event)

"""
IPC via HTTP/REST
[KoboldCPP Subprocess] <--------------> [Main Process]
        |                                     |
        |                                     |
        | SSE Stream        Worker Thread     |     Main Thread (Event Loop)
        |                   -------------     |     ----------------------
        |                        |            |           |
        |                  [ChatAPIWorker.run()]          |
        |                        |                        |
        |                  [Get SSE Token]                |
        |                        |                        |
        |                  json.loads(event.data)         |
        |                        |                        |
        |                  extract 'token'                |
        |                        |                        |
        |                  signals.response_signal.emit() |
        |                        |--------------------> [Qt Event Queue]
        |                        |                        |
        |                  [Get Next Token]       process signals/events
        |                        |                        |
        |                        |               [update_response()]
        |                        |                        |
        |                        |            QTextEdit.insertPlainText()
        |                        |                        |
        |                        |              [Chat Display Window]
"""