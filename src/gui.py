import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import sys
import os
from pathlib import Path

def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path = nvidia_base_path / 'cuda_runtime' / 'bin'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add = [str(cuda_path), str(cublas_path), str(cudnn_path)]
    env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_4', 'PATH']
    
    # print(f"Virtual environment base: {venv_base}")
    # print(f"NVIDIA base path: {nvidia_base_path}")
    # print(f"CUDA path: {cuda_path}")
    # print(f"cuBLAS path: {cublas_path}")
    # print(f"cuDNN path: {cudnn_path}")
    
    for env_var in env_vars:
        current_value = os.environ.get(env_var, '')
        new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value
        # print(f"\n{env_var} updated:")
        # print(f"  Old value: {current_value}")
        # print(f"  New value: {new_value}")
    
    # print("\nCUDA paths have been set or updated in the environment variables.")

set_cuda_paths()

import logging
import traceback
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QStyleFactory, QMenuBar
)
from initialize import main as initialize_system
from metrics_bar import MetricsBar
from gui_tabs import create_tabs
from utilities import list_theme_files, make_theme_changer, load_stylesheet

script_dir = Path(__file__).parent.resolve()

log_file_path = script_dir / 'gui_log.txt'

logging.basicConfig(filename='gui_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
# print(f"Log file should be created at: {log_file_path}")


class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()
        try:
            initialize_system()
            self.metrics_bar = MetricsBar()
            self.tab_widget = create_tabs()
            self.init_ui()
            self.init_menu()
            logging.info("GUI initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing GUI: {str(e)}")
            logging.debug(traceback.format_exc())
            raise

    def init_ui(self):
        try:
            self.setWindowTitle('VectorDB Plugin (LM Studio Edition)')
            self.setGeometry(300, 300, 820, 1000)
            self.setMinimumSize(350, 410)
            
            main_layout = QVBoxLayout(self)
            main_layout.addWidget(self.tab_widget)
            main_layout.addWidget(self.metrics_bar)
            logging.info("UI initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing UI: {str(e)}")
            logging.debug(traceback.format_exc())
            raise

    def init_menu(self):
        try:
            self.menu_bar = QMenuBar(self)
            self.theme_menu = self.menu_bar.addMenu('Themes')
            for theme in list_theme_files():
                self.theme_menu.addAction(theme).triggered.connect(make_theme_changer(theme))
            logging.info("Menu initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing menu: {str(e)}")
            logging.debug(traceback.format_exc())
            raise

    def cleanup_tabs(self):
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, 'cleanup') and callable(tab.cleanup):
                tab.cleanup()
                logging.info(f"Cleaned up tab: {self.tab_widget.tabText(i)}")

    def closeEvent(self, event):
        try:
            docs_dir = Path(__file__).parent / 'Docs_for_DB'
            for item in docs_dir.glob('*'):
                if item.is_file():
                    item.unlink()
            self.metrics_bar.stop_metrics_collector()
            
            # call each tab's cleanup function
            self.cleanup_tabs()
            
            logging.info("Application closed successfully")
            super().closeEvent(event)
        except Exception as e:
            logging.error(f"Error during application close: {str(e)}")
            logging.debug(traceback.format_exc())

def main():
    try:
        logging.info("Starting application")
        app = QApplication(sys.argv)
        app.setStyleSheet(load_stylesheet('custom_stylesheet_steel_ocean.css'))
        ex = DocQA_GUI()
        ex.show()
        logging.info("Application main window shown")
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"Critical error in main function: {str(e)}")
        logging.debug(traceback.format_exc())
        print(f"A critical error occurred. Please check the log file 'gui_log.txt' for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()