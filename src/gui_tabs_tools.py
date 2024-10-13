from PySide6.QtWidgets import QVBoxLayout, QGroupBox, QWidget, QPushButton, QMessageBox, QHBoxLayout
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QColor
from gui_tabs_tools_transcribe import TranscriberToolSettingsTab
from gui_tabs_tools_vision import VisionToolSettingsTab
from gui_tabs_tools_scrape import ScrapeDocumentationTab
# from gui_tabs_tools_ocr import OcrToolSettingsTab
# from gui_tabs_tools_keybert import KeywordExtractorTab
from gui_tabs_tools_misc import MiscTab
from initialize import restore_vector_db_backup
from utilities import backup_database
from pathlib import Path
import shutil

class RestoreBackupThread(QThread):
    finished = Signal(bool)
    def run(self):
        try:
            restore_vector_db_backup()
            self.finished.emit(True)
        except Exception as e:
            print(f"Error during backup restoration: {e}")
            self.finished.emit(False)

class BackupDatabaseThread(QThread):
    finished = Signal(bool)
    def run(self):
        try:
            backup_database()
            self.finished.emit(True)
        except Exception as e:
            print(f"Error during database backup: {e}")
            self.finished.emit(False)

class GuiSettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.groups = {}
        classes = {
            "TRANSCRIBE FILE": (TranscriberToolSettingsTab, 1.5),
            "SCRAPE DOCUMENTATION": (ScrapeDocumentationTab, 1.5),
            "TEST VISION MODEL": (VisionToolSettingsTab, 6),
            # "PERFORM OCR": (OcrToolSettingsTab, 2),
            # "KEYWORD EXTRACTOR": (KeywordExtractorTab, 3),
            "MISC": (MiscTab, 1),
        }
        for title, (TabClass, stretch) in classes.items():
            settings = TabClass()
            group = QGroupBox(title, checkable=True, checked=True)
            group.setLayout(QVBoxLayout())
            group.layout().addWidget(settings)
            
            self.groups[group] = stretch
            self.layout.addWidget(group, stretch)
            
            group.toggled.connect(lambda checked, g=group, s=settings: 
                                  (s.setVisible(checked), self.adjust_stretch()))

    def adjust_stretch(self):
        for group, factor in self.groups.items():
            self.layout.setStretchFactor(group, factor if group.isChecked() else 0)