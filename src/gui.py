import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
import sys
import os
from pathlib import Path

def set_cuda_paths():
    """
    Specifically, this function:
    - Adds the following paths to the `PATH` environment variable:
      - `cuda_runtime/bin` (CUDA runtime binaries)
      - `cuda_runtime/bin/lib/x64` (runtime libraries)
      - `cuda_runtime/include` (runtime headers)
      - `cublas/bin` (cuBLAS binaries)
      - `cudnn/bin` (cuDNN binaries)
      - `cuda_nvrtc/bin` (NVRTC binaries)
      - `cuda_nvcc/bin` (NVCC binaries)
    - Sets the `CUDA_PATH` environment variable to point to the base `cuda_runtime` directory,
      ensuring compatibility with tools and frameworks that depend on this specific path.

    Existing values in the `PATH` variable are preserved, and new paths are appended as needed.
    """
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
    
    # because of triton
    triton_cuda_path = nvidia_base_path / 'cuda_runtime'
    os.environ['CUDA_PATH'] = str(triton_cuda_path)

set_cuda_paths()

"""
# DEBUG VERSION
def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path_runtime = nvidia_base_path / 'cuda_runtime'
    
    # Set CUDA_PATH first
    os.environ['CUDA_PATH'] = str(cuda_path_runtime)
    
    # Debug prints to check if files exist in expected locations
    expected_files = [
        cuda_path_runtime / 'bin' / 'cudart64_12.dll',
        cuda_path_runtime / 'bin' / 'ptxas.exe',
        cuda_path_runtime / 'include' / 'cuda.h',
        cuda_path_runtime / 'lib' / 'x64' / 'cuda.lib'
    ]
    
    print(f"\nCUDA_PATH is set to: {os.environ['CUDA_PATH']}")
    print("\nChecking for required files:")
    for file in expected_files:
        print(f"File {file} exists: {file.exists()}")
    
    # Rest of your path setup
    cuda_path_runtime_lib = cuda_path_runtime / 'bin' / 'lib' / 'x64'
    cuda_path_runtime_include = cuda_path_runtime / 'include'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base_path / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base_path / 'cuda_nvcc' / 'bin'
    
    paths_to_add = [
        str(cuda_path_runtime / 'bin'),
        str(cuda_path_runtime_lib),
        str(cuda_path_runtime_include),
        str(cublas_path),
        str(cudnn_path),
        str(nvrtc_path),
        str(nvcc_path),
    ]
    
    current_path = os.environ.get('PATH', '')
    new_path = os.pathsep.join(paths_to_add + [current_path] if current_path else paths_to_add)
    os.environ['PATH'] = new_path

set_cuda_paths()
"""

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QMenuBar, QHBoxLayout, QMessageBox, QInputDialog
)
from initialize import main as initialize_system
from metrics_bar import MetricsWidget as MetricsBar
from gui_tabs import create_tabs
from utilities import list_theme_files, make_theme_changer, load_stylesheet
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
        required_folder = script_dir / 'Models' / 'vector' / 'thenlper--gte-base'
        
        if not required_folder.exists() or not required_folder.is_dir():
            QMessageBox.warning(
                self,
                "Ask Jeeves",
                "Before using Jeeves you must download the gte-base embedding model, which you can do from the Models tab. Jeeves is waiting."
            )
            return
                
        model_choice, ok = QInputDialog.getItem(
            self,
            "Select Chat Model",
            "Choose which model you'd like Jeeves to use:",
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
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet('custom_stylesheet_default.css'))
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()