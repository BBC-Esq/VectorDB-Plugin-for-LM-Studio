from PySide6.QtWidgets import QVBoxLayout, QGroupBox, QWidget, QPushButton, QMessageBox, QHBoxLayout
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QColor
from gui_tabs_tools_transcribe import TranscriberToolSettingsTab
from gui_tabs_tools_vision import VisionToolSettingsTab
from initialize import restore_vector_db_backup

class RestoreBackupThread(QThread):
    finished = Signal(bool)
    def run(self):
        try:
            restore_vector_db_backup()
            self.finished.emit(True)
        except Exception as e:
            print(f"Error during backup restoration: {e}")
            self.finished.emit(False)

class GuiSettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.groups = {}
        classes = {
            "TRANSCRIBE FILE": (TranscriberToolSettingsTab, 1),
            "TEST VISION MODEL": (VisionToolSettingsTab, 4),
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
        
        self.backup_group = QGroupBox("RESTORE DATABASE BACKUPS", checkable=True, checked=True)
        backup_layout = QVBoxLayout()
        
        self.restore_backup_button = QPushButton("Restore Backups for All Databases")
        self.restore_backup_button.clicked.connect(self.restore_backup)
        
        subdued_red = QColor(50, 10, 10)
        light_grey = QColor(200, 200, 200)
        self.restore_backup_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {subdued_red.name()};
                color: {light_grey.name()};
                padding: 5px;
                border: none;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {QColor(subdued_red.lighter(120)).name()};
            }}
            QPushButton:pressed {{
                background-color: {QColor(subdued_red.darker(110)).name()};
            }}
            QPushButton:disabled {{
                background-color: {QColor(subdued_red.lighter(150)).name()};
                color: {QColor(light_grey.darker(150)).name()};
            }}
        """)
        
        center_button_layout = QHBoxLayout()
        center_button_layout.addStretch(1)
        center_button_layout.addWidget(self.restore_backup_button)
        center_button_layout.addStretch(1)
        
        backup_layout.addLayout(center_button_layout)
        self.backup_group.setLayout(backup_layout)
        
        self.layout.addWidget(self.backup_group)
        
        self.backup_group.toggled.connect(lambda checked: self.restore_backup_button.setVisible(checked))
        
        self.adjust_stretch()
        
        self.backup_thread = None

    def adjust_stretch(self):
        for group, factor in self.groups.items():
            self.layout.setStretchFactor(group, factor if group.isChecked() else 0)

    def restore_backup(self):
        confirm = QMessageBox.question(
            self,
            "Confirm Restoration",
            "This action will erase all existing databases and restore any backups.\n\nAre you sure that you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            self.restore_backup_button.setEnabled(False)
            self.restore_backup_button.setText("Restoring...")
            self.backup_thread = RestoreBackupThread()
            self.backup_thread.finished.connect(self.on_restore_finished)
            self.backup_thread.start()
        else:
            pass

    def on_restore_finished(self, success):
        self.restore_backup_button.setEnabled(True)
        self.restore_backup_button.setText("Restore Backups for All Databases")
        if success:
            QMessageBox.information(self, "Backup Restored", "The database backup has been successfully restored.")
        else:
            QMessageBox.critical(self, "Restoration Failed", "Failed to restore the database backup. Check the console for error details.")