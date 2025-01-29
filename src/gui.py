import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
import sys
import os
from pathlib import Path

def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path_runtime = nvidia_base_path / 'cuda_runtime' / 'bin'
    cuda_path_runtime_lib = nvidia_base_path / 'cuda_runtime' / 'bin' / 'lib' / 'x64'
    cuda_path_runtime_include = nvidia_base_path / 'cuda_runtime' / 'include'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base_path / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base_path / 'cuda_nvcc' / 'bin'
    
    paths_to_add = [
        str(cuda_path_runtime),
        str(cuda_path_runtime_lib),
        str(cuda_path_runtime_include),
        str(cublas_path),
        str(cudnn_path),
        str(nvrtc_path),
        str(nvcc_path),
    ]
    
    current_value = os.environ.get('PATH', '')
    new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
    os.environ['PATH'] = new_value
    
    # i blame triton
    triton_cuda_path = nvidia_base_path / 'cuda_runtime'
    os.environ['CUDA_PATH'] = str(triton_cuda_path)

set_cuda_paths()

from ctypes import windll, byref, sizeof, c_void_p, c_int
from ctypes.wintypes import BOOL, HWND, DWORD

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QMenuBar, QHBoxLayout, QMessageBox, QInputDialog
)
from initialize import main as initialize_system
from metrics_bar import MetricsWidget as MetricsBar
from gui_tabs import create_tabs
from utilities import list_theme_files, make_theme_changer, load_stylesheet, download_kobold_executable, download_kokoro_tts, download_with_threadpool
from gui_file_settings_hf import set_hf_access_token
from module_ask_jeeves import ChatWindow
from constants import JEEVES_MODELS

script_dir = Path(__file__).parent.resolve()

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()
        initialize_system()
        self.metrics_bar = MetricsBar()
        self.tab_widget = create_tabs()
        self.init_ui()
        self.init_menu()
        self.chat_window = None
        self.set_dark_titlebar()

    def set_dark_titlebar(self):
        DWMWA_USE_IMMERSIVE_DARK_MODE = DWORD(20)
        set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
        hwnd = HWND(int(self.winId()))
        rendering_policy = BOOL(True)
        set_window_attribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            byref(rendering_policy), 
            sizeof(rendering_policy)
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

    def init_ui(self):
        self.setWindowTitle('VectorDB Plugin (Kobold and LM Studio Edition)')
        self.setGeometry(300, 300, 820, 1000)
        self.setMinimumSize(350, 410)
        
        main_layout = QVBoxLayout(self)
        
        main_layout.addWidget(self.tab_widget)
        
        metrics_layout = QHBoxLayout()
        metrics_layout.addWidget(self.metrics_bar)
        
        # max height for the MetricsBar
        self.metrics_bar.setMaximumHeight(80)
        
        main_layout.addLayout(metrics_layout)

    def init_menu(self):
        self.menu_bar = QMenuBar(self)
        
        self.file_menu = self.menu_bar.addMenu('File')
        
        self.theme_menu = self.file_menu.addMenu('Themes')
        for theme in list_theme_files():
            self.theme_menu.addAction(theme).triggered.connect(make_theme_changer(theme))
        
        self.hf_token_menu = self.file_menu.addAction('Hugging Face Access Token')
        self.hf_token_menu.triggered.connect(lambda: set_hf_access_token(self))

        self.jeeves_action = self.menu_bar.addAction('Jeeves')
        self.jeeves_action.triggered.connect(self.open_chat_window)

    def open_chat_window(self):
        # First check - embedding model
        required_folder = script_dir / 'Models' / 'vector' / 'ibm-granite--granite-embedding-30m-english'
        if not required_folder.exists() or not required_folder.is_dir():
            QMessageBox.warning(
                self,
                "Ask Jeeves",
                "Before using Jeeves you must download the granite-30m embedding model, which you can do from the Models tab. Jeeves is waiting."
            )
            return

        # Second check - Kokoro TTS model
        tts_path = script_dir / "Models" / "tts" / "ctranslate2-4you--Kokoro-82M-light"
        if not tts_path.exists() or not tts_path.is_dir():
            ret = QMessageBox.question(
                self,
                "Kokoro TTS Model Not Found",
                "The Kokoro TTS model is missing!\n\nWould you like to download it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if ret == QMessageBox.Yes:
                def on_kokoro_download_complete(success, message):
                    if success:
                        QMessageBox.information(
                            self,
                            "Download Complete",
                            "Kokoro TTS model has been downloaded successfully.\n\n"
                            "Text-to-speech functionality will be available when you restart Jeeves."
                        )
                    else:
                        QMessageBox.critical(
                            self,
                            "Download Error",
                            f"Failed to download Kokoro TTS model: {message}"
                        )
                download_with_threadpool(download_kokoro_tts, callback=on_kokoro_download_complete)
                return
            else:
                QMessageBox.information(
                    self,
                    "TTS Unavailable",
                    "You can continue without text-to-speech functionality.\n\n"
                    "You can always download the TTS model later by closing and reopening Jeeves."
                )

        # Third check - Kobold executable
        exe_path = script_dir / "Assets" / "koboldcpp_nocuda.exe"
        if not exe_path.exists():
            ret = QMessageBox.question(
                self,
                "KoboldCPP Not Found",
                "KoboldCPP executable is missing!\n\nWould you like to download KoboldCPP now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if ret == QMessageBox.Yes:
                def on_download_complete(success, message):
                    if success:
                        QMessageBox.information(
                            self,
                            "Action Required",
                            "KoboldCPP has been downloaded, but requires Windows security permissions before first use.\n\n"
                            "1. Please close this program\n"
                            "2. Follow the instructions at https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio for setting up Windows permissions\n"
                            "3. Restart this program and try Jeeves again"
                        )
                    else:
                        QMessageBox.critical(
                            self,
                            "Download Error", 
                            f"Failed to download KoboldCPP: {message}")

                download_with_threadpool(download_kobold_executable, callback=on_download_complete)
                return

        model_choice, ok = QInputDialog.getItem(
            self,
            "Select Chat Model",
            "Choose Jeeves' Brain:",
            list(JEEVES_MODELS.keys()),
            0,
            False
        )
        
        if not ok or not model_choice:
            return

        if self.chat_window:
            self.chat_window.close()
            self.chat_window = None

        self.chat_window = ChatWindow(parent=self, selected_model=model_choice)
        self.chat_window.destroyed.connect(self.on_chat_window_closed)
        self.chat_window.show()

    def on_chat_window_closed(self):
        self.chat_window = None

    def cleanup_tabs(self):
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, 'cleanup') and callable(tab.cleanup):
                tab.cleanup()

    def closeEvent(self, event):
        docs_dir = Path(__file__).parent / 'Docs_for_DB'
        for item in docs_dir.glob('*'):
            if item.is_file():
                item.unlink()
        self.metrics_bar.stop_metrics_collector()
        
        self.cleanup_tabs()
        
        super().closeEvent(event)

def main():
    from PySide6.QtCore import Qt

    # Enable High DPI scaling and high-resolution pixmaps
    if hasattr(QApplication, 'setHighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Optionally, set the application font size based on DPI (recommended)
    # font = app.font()
    # font.setPointSize(10)  # Adjust as necessary
    # app.setFont(font)

    app.setStyleSheet(load_stylesheet('custom_stylesheet_default.css'))

    ex = DocQA_GUI()
    ex.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()