import sys
import os
from pathlib import Path

def set_cuda_paths():
    script_dir = Path(__file__).parent.resolve()
    cuda_path = script_dir / 'Lib' / 'site-packages' / 'nvidia'
    cublas_path = cuda_path / 'cublas' / 'bin'
    cudnn_path = cuda_path / 'cudnn' / 'bin'

    paths_to_add = [str(cuda_path), str(cublas_path), str(cudnn_path)]

    env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_2', 'PATH']

    for env_var in env_vars:
        current_value = os.environ.get(env_var, '')
        new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value

    print("CUDA paths have been set or updated in the environment variables.")

# Execute the function immediately
set_cuda_paths()

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QStyleFactory, QMenuBar
)
import multiprocessing
from initialize import main as initialize_system
from metrics_bar import MetricsBar
from gui_tabs import create_tabs
from utilities import list_theme_files, make_theme_changer, load_stylesheet

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()
        initialize_system()
        self.metrics_bar = MetricsBar()
        self.init_ui()
        self.init_menu()

    def init_ui(self):
        self.setWindowTitle('LM Studio VectorDB Plugin - www.chintellalaw.com')
        self.setGeometry(300, 300, 775, 1000)
        self.setMinimumSize(350, 410)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(create_tabs())
        main_layout.addWidget(self.metrics_bar)

    def init_menu(self):
        self.menu_bar = QMenuBar(self)
        self.theme_menu = self.menu_bar.addMenu('Themes')
        for theme in list_theme_files():
            self.theme_menu.addAction(theme).triggered.connect(make_theme_changer(theme))

    def closeEvent(self, event):
        docs_dir = Path(__file__).parent / 'Docs_for_DB'
        for item in docs_dir.glob('*'):
            if item.is_file():
                item.unlink()
        self.metrics_bar.stop_metrics_collector()
        super().closeEvent(event)

def main():
    multiprocessing.set_start_method('spawn')
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet('custom_stylesheet_steel_ocean.css'))
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()