import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QStyleFactory, QMenuBar
)
import multiprocessing
from initialize import main as initialize_system
from metrics_bar import MetricsBar
from gui_tabs import create_tabs
from utilities import list_theme_files, make_theme_changer, load_stylesheet

def set_cuda_paths():
    venv_base = Path(sys.executable).parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    for env_var in ['CUDA_PATH', 'CUDA_PATH_V12_2']:
        current_path = os.environ.get(env_var, '')
        os.environ[env_var] = os.pathsep.join(filter(None, [str(nvidia_base_path), current_path]))

set_cuda_paths()

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