import json
import textwrap
import multiprocessing
from pathlib import Path
import logging
import yaml
import os
import traceback
from PySide6.QtCore import Qt, QUrl, QThread, Signal as pyqtSignal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QMessageBox

from module_process_images import choose_image_loader

CONFIG_FILE = 'config.yaml'

logging.basicConfig(level=logging.DEBUG) # doublecheck how to suppress cuda version info messsage

class CustomWebEnginePage(QWebEnginePage):
    def acceptNavigationRequest(self, url, _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            logging.debug(f"Opening URL in system browser: {url.toString()}")
            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, _type, isMainFrame)

    def createWindow(self, _type):
        logging.debug(f"createWindow called with type: {_type}")
        if _type == QWebEnginePage.WebBrowserTab or _type == QWebEnginePage.WebBrowserBackgroundTab:
            return self
        return None

class ImageProcessorThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        try:
            documents = choose_image_loader()
            self.finished.emit(documents)
        except Exception as e:
            error_msg = f"Error in image processing: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class VisionToolSettingsTab(QWidget):
    HTML_FILE = 'vision_model_table.html'

    def __init__(self):
        super().__init__()

        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)

        hBoxLayout = QHBoxLayout()
        mainVLayout.addLayout(hBoxLayout)

        processButton = QPushButton("Process")
        hBoxLayout.addWidget(processButton)
        processButton.clicked.connect(self.confirmationBeforeProcessing)

        self.webView = QWebEngineView()
        custom_page = CustomWebEnginePage(self.webView)
        self.webView.setPage(custom_page)
        script_dir = Path(__file__).resolve().parent
        html_file_path = script_dir / self.HTML_FILE
        self.webView.setUrl(QUrl.fromLocalFile(str(html_file_path)))
        mainVLayout.addWidget(self.webView)

        self.thread = None

    def confirmationBeforeProcessing(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(
            "1. Create Database Tab:\n"
            "Select files you theoretically want in the vector database.\n\n"
            "2. Settings Tab:\n"
            "Select the vision model you want to test.\n\n"
            "3. Click the 'Process' button.\n\n"
            "This will test the selected vison model before actually entering the images into the vector database.\n\n"
            "Do you want to proceed?"
        )
        msgBox.setWindowTitle("Confirm Processing")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            self.startProcessing()
    
    def startProcessing(self):
        if self.thread is None:
            self.thread = ImageProcessorThread()
            self.thread.finished.connect(self.onProcessingFinished)
            self.thread.error.connect(self.onProcessingError)
            self.thread.start()

    def onProcessingFinished(self, documents):
        self.thread = None
        logging.info(f"Processed {len(documents)} documents")
        contents = self.extract_page_content(documents)
        self.save_page_contents(contents)

    def onProcessingError(self, error_msg):
        self.thread = None
        logging.error(f"Processing error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during image processing:\n\n{error_msg}")

    def extract_page_content(self, documents):
        contents = []
        for doc in documents:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                content = doc.page_content
                filename = doc.metadata.get('file_name', 'Unknown filename')
            elif isinstance(doc, dict):
                content = doc.get("page_content", "Document is missing 'page_content'.")
                filename = doc.get("metadata", {}).get('file_name', 'Unknown filename')
            else:
                content = "Document is missing 'page_content'."
                filename = 'Unknown filename'
            
            # Wrap the content to 100 characters
            wrapped_content = textwrap.fill(content, width=100)
            contents.append((filename, wrapped_content))
        
        return contents

    def save_page_contents(self, contents):
        """
        Takes a list of (filename, page_content) tuples, writes them to a file named 'sample_vision_summaries.txt'
        in the same directory as the script, and then opens the file.
        """
        script_dir = Path(__file__).resolve().parent
        output_file = script_dir / "sample_vision_summaries.txt"
        
        with open(output_file, 'w', encoding='utf-8') as file:
            for filename, content in contents:
                file.write(f"{filename}\n\n")
                file.write(f"{content}\n\n")

        logging.info(f"Saved vision summaries to {output_file}")

        self.open_file(output_file)

    def open_file(self, file_path):
        try:
            if os.name == 'nt':
                os.startfile(file_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', file_path])
            elif sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', file_path])
            else:
                raise NotImplementedError("Unsupported operating system")
            logging.info(f"Opened file: {file_path}")
        except Exception as e:
            error_msg = f"Error opening file: {e}"
            logging.error(error_msg)
            QMessageBox.warning(self, "Error", error_msg)

if __name__ == "__main__":
    pass