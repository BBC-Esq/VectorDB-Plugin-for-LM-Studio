import sys
import os
from pathlib import Path
import logging
import traceback
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QStyleFactory, QMenuBar
)
import multiprocessing
from initialize import main as initialize_system
from metrics_bar import MetricsBar
from gui_tabs import create_tabs
from utilities import list_theme_files, make_theme_changer, load_stylesheet

# Set up logging
logging.basicConfig(filename='gui_log.txt', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def set_cuda_paths():
    try:
        venv_base = Path(sys.executable).parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
        for env_var in ['CUDA_PATH', 'CUDA_PATH_V12_1']:
            current_path = os.environ.get(env_var, '')
            os.environ[env_var] = os.pathsep.join(filter(None, [str(nvidia_base_path), current_path]))
        logging.info("CUDA paths set successfully")
    except Exception as e:
        logging.error(f"Error setting CUDA paths: {str(e)}")
        logging.debug(traceback.format_exc())

set_cuda_paths()

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()
        try:
            initialize_system()
            self.metrics_bar = MetricsBar()
            self.init_ui()
            self.init_menu()
            logging.info("GUI initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing GUI: {str(e)}")
            logging.debug(traceback.format_exc())
            raise

    def init_ui(self):
        try:
            self.setWindowTitle('LM Studio VectorDB Plugin - www.chintellalaw.com')
            self.setGeometry(300, 300, 775, 1000)
            self.setMinimumSize(350, 410)
            
            main_layout = QVBoxLayout(self)
            main_layout.addWidget(create_tabs())
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

    def closeEvent(self, event):
        try:
            docs_dir = Path(__file__).parent / 'Docs_for_DB'
            for item in docs_dir.glob('*'):
                if item.is_file():
                    item.unlink()
            self.metrics_bar.stop_metrics_collector()
            logging.info("Application closed successfully")
            super().closeEvent(event)
        except Exception as e:
            logging.error(f"Error during application close: {str(e)}")
            logging.debug(traceback.format_exc())

def main():
    try:
        logging.info("Starting application")
        multiprocessing.set_start_method('spawn')
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