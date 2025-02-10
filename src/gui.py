import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
import sys
import os
from pathlib import Path

from utilities import set_cuda_paths
set_cuda_paths()

from ctypes import windll, byref, sizeof, c_void_p, c_int
from ctypes.wintypes import BOOL, HWND, DWORD

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QMenuBar, QHBoxLayout, QMessageBox, QInputDialog
)
import yaml
from initialize import main as initialize_system
from metrics_bar import MetricsWidget as MetricsBar
from gui_tabs import create_tabs
from utilities import (
    list_theme_files,
    load_stylesheet,
    ensure_theme_config,
    update_theme_in_config,
    make_theme_changer,
    download_kobold_executable,
    download_kokoro_tts,
    download_with_threadpool,
)
from gui_file_credentials import manage_credentials
from module_ask_jeeves import ChatWindow, launch_jeeves_process
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
        self.setWindowTitle('VectorDB Plugin')
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
        self.hf_token_menu.triggered.connect(lambda: manage_credentials(self, 'hf'))

        self.openai_key_menu = self.file_menu.addAction('OpenAI API Key')
        self.openai_key_menu.triggered.connect(lambda: manage_credentials(self, 'openai'))

        self.jeeves_action = self.menu_bar.addAction('Jeeves')
        self.jeeves_action.triggered.connect(self.open_chat_window)

    def open_chat_window(self):
        required_folder = script_dir / 'Models' / 'vector' / 'BAAI--bge-small-en-v1.5'
        if not required_folder.exists() or not required_folder.is_dir():
            QMessageBox.warning(
                self,
                "Ask Jeeves",
                "Before using Jeeves you must download the bge-small-en-v1.5 embedding model, which you can do from the Models tab. Jeeves is waiting."
            )
            return

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
                            "Kokoro TTS model has been downloaded successfully."
                        )
                    else:
                        QMessageBox.critical(
                            self,
                            "Download Error",
                            f"Failed to download Kokoro TTS model: {message}"
                        )
                download_with_threadpool(download_kokoro_tts, callback=on_kokoro_download_complete)
                return

        if sys.platform == 'win32':
            import multiprocessing
            multiprocessing.freeze_support()
            jeeves_process = multiprocessing.Process(target=launch_jeeves_process)
            jeeves_process.start()
        else:
            jeeves_process = multiprocessing.Process(target=launch_jeeves_process)
            jeeves_process.start()

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

    if hasattr(QApplication, 'setHighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    # Optionally, set the application font size based on DPI (recommended)
    # font = app.font()
    # font.setPointSize(10)  # Adjust as necessary
    # app.setFont(font)

    theme = ensure_theme_config()
    app.setStyleSheet(load_stylesheet(theme))

    ex = DocQA_GUI()
    ex.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()


