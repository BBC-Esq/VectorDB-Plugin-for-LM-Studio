import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QStyleFactory, QMenuBar
)
import multiprocessing
from initialize import main as initialize_system
from metrics_bar import MetricsBar
from gui_tabs import create_tabs
from utilities import list_theme_files, make_theme_changer, load_stylesheet
from pathlib import Path

class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()

        initialize_system()
        self.metrics_bar = MetricsBar()
        self.init_ui()
        self.init_menu()

    def init_ui(self):
        self.setWindowTitle('LM Studio ChromaDB Plugin - www.chintellalaw.com')
        self.setGeometry(300, 300, 850, 1040)
        self.setMinimumSize(350, 410)

        # Main Layout
        main_layout = QVBoxLayout(self)

        # Tab Widget
        tab_widget = create_tabs()
        main_layout.addWidget(tab_widget)

        # Metrics Bar
        main_layout.addWidget(self.metrics_bar)

    def init_menu(self):
        self.menu_bar = QMenuBar(self)
        self.theme_menu = self.menu_bar.addMenu('Themes')

        self.theme_files = list_theme_files()
        for theme in self.theme_files:
            action = self.theme_menu.addAction(theme)
            action.triggered.connect(make_theme_changer(theme))

    def closeEvent(self, event):
        docs_dir = Path(__file__).parent / 'Docs_for_DB'
        
        for item in docs_dir.iterdir():
            if item.is_file():
                item.unlink()
                
        self.metrics_bar.stop_metrics_collector()
        super().closeEvent(event)

def main():
    multiprocessing.set_start_method('spawn')
    app = QApplication(sys.argv)
    stylesheet = load_stylesheet('custom_stylesheet_steel_ocean.css')
    app.setStyleSheet(stylesheet)
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
