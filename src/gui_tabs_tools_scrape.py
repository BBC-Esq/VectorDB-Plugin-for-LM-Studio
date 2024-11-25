import os
import subprocess
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel, QColor
import platform
import shutil
from module_scraper import ScraperWorker, WorkerThread, ScraperRegistry
from constants import scrape_documentation

class ScrapeDocumentationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setToolTip("Tab for scraping documentation from the selected source.")
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        label = QLabel("Select Documentation:")
        label.setToolTip("Choose the documentation set you want to scrape from the combo box.")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(label)

        hbox = QHBoxLayout()

        self.doc_combo = QComboBox()
        self.doc_combo.setToolTip("Select the documentation source from the dropdown.")
        self.populate_combo_box()
        hbox.addWidget(self.doc_combo)

        self.scrape_button = QPushButton("Scrape")
        self.scrape_button.setToolTip("Start the scraping process for the selected documentation.")
        self.scrape_button.clicked.connect(self.start_scraping)
        hbox.addWidget(self.scrape_button)

        hbox.setStretch(0, 1)
        hbox.setStretch(1, 1)

        main_layout.addLayout(hbox)

        self.status_label = QLabel("Pages scraped: 0")
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setOpenExternalLinks(False)
        self.status_label.setToolTip("Click the links to open the folder containing scraped data.")
        self.status_label.linkActivated.connect(self.open_folder)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(self.status_label)

        main_layout.addStretch()

    def populate_combo_box(self):
        doc_options = list(scrape_documentation.keys())
        doc_options.sort(key=str.lower)

        model = QStandardItemModel()

        scraped_dir = os.path.join(os.path.dirname(__file__), "Scraped_Documentation")

        for doc in doc_options:
            folder = scrape_documentation[doc]["folder"]
            folder_path = os.path.join(scraped_dir, folder)

            item = QStandardItem(doc)

            if os.path.exists(folder_path):
                item.setForeground(QColor('#4B0F0F'))

            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            model.appendRow(item)

        self.doc_combo.setModel(model)

    def start_scraping(self):
        selected_doc = self.doc_combo.currentText()
        url = scrape_documentation[selected_doc]["URL"]
        folder = scrape_documentation[selected_doc]["folder"]
        scraper_name = scrape_documentation[selected_doc].get("scraper_class", "BaseScraper")
        
        scraper_class = ScraperRegistry.get_scraper(scraper_name)
        
        self.current_folder = os.path.join(os.path.dirname(__file__), "Scraped_Documentation", folder)

        if os.path.exists(self.current_folder):
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Folder already exists for {selected_doc}")
            msg_box.setInformativeText("Proceeding will delete the existing contents and start a new scraping session.")
            msg_box.setWindowTitle("Existing Folder Warning")
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)

            if msg_box.exec() == QMessageBox.Cancel:
                return

            for filename in os.listdir(self.current_folder):
                file_path = os.path.join(self.current_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

        self.status_label.setText("Pages scraped: 0")
        self.scrape_button.setEnabled(False)

        self.worker = ScraperWorker(url, folder, scraper_class)
        self.worker.status_updated.connect(self.update_status)
        self.worker.scraping_finished.connect(self.scraping_finished)

        self.thread = WorkerThread(self.worker)
        self.thread.start()

    def update_status(self, status):
        self.status_label.setText(f'Pages scraped: {status} <a href="open_folder">Open Folder</a>')

    def scraping_finished(self):
        self.scrape_button.setEnabled(True)
        selected_doc = self.doc_combo.currentText()
        final_count = len([f for f in os.listdir(self.current_folder) if f.endswith('.html')])
        self.status_label.setText(f'Scraping {selected_doc} completed. Pages scraped: {final_count} <a href="open_folder">Open Folder</a>')
        self.populate_combo_box()

    def open_folder(self, link):
        if link == "open_folder":
            if platform.system() == "Windows":
                os.startfile(self.current_folder)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", self.current_folder])
            else:
                subprocess.Popen(["xdg-open", self.current_folder])