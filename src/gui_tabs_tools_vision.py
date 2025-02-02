import sys
import textwrap
import subprocess
from pathlib import Path
import logging
import yaml
import tempfile
import os
import traceback
import gc
import time
from PIL import Image
import torch
from PySide6.QtCore import QUrl, QThread, Signal as pyqtSignal, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox, QFileDialog, QProgressDialog

import module_process_images
from module_process_images import choose_image_loader
from constants import VISION_MODELS

CONFIG_FILE = 'config.yaml'

class CustomWebEnginePage(QWebEnginePage):
    def acceptNavigationRequest(self, url, _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            print(f"Opening URL in system browser: {url.toString()}")
            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, _type, isMainFrame)

    def createWindow(self, _type):
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

class MultiModelProcessorThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            results = []
            with Image.open(self.image_path) as raw_image:
                for i, model_name in enumerate(VISION_MODELS.keys()):
                    if self.is_cancelled:
                        print("\nProcessing cancelled by user")
                        torch.cuda.empty_cache()
                        gc.collect()
                        break

                    try:
                        print(f"\nProcessing with {model_name}...")
                        model_config = {"vision": {"chosen_model": model_name}}

                        loader_name = VISION_MODELS[model_name]['loader']
                        loader_class = getattr(module_process_images, loader_name)
                        loader = loader_class(model_config)

                        loader.model, loader.tokenizer, loader.processor = loader.initialize_model_and_tokenizer()
                        start_time = time.time()
                        description = loader.process_single_image(raw_image)
                        process_time = time.time() - start_time
                        description = textwrap.fill(description, width=10)
                        results.append((model_name, description, process_time))

                        if hasattr(loader, 'model') and loader.model is not None:
                            loader.model.cpu()
                            del loader.model
                        if hasattr(loader, 'tokenizer') and loader.tokenizer is not None:
                            del loader.tokenizer
                        if hasattr(loader, 'processor') and loader.processor is not None:
                            del loader.processor

                        torch.cuda.empty_cache()
                        gc.collect()

                        print(f"Completed {model_name}")
                        self.progress.emit(i + 1)

                    except Exception as e:
                        error_msg = f"Error processing with {model_name}: {str(e)}\n{traceback.format_exc()}"
                        results.append((model_name, error_msg, 0.0))
                        print(error_msg)
                        torch.cuda.empty_cache()
                        gc.collect()

            torch.cuda.empty_cache()
            gc.collect()
            self.finished.emit(results)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            self.error.emit(str(e))

class VisionToolSettingsTab(QWidget):
    HTML_FILE = 'vision_model_table.html'

    def __init__(self):
        super().__init__()

        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)

        hBoxLayout = QHBoxLayout()
        mainVLayout.addLayout(hBoxLayout)

        processButton = QPushButton("Multiple Files + One Vision Model")
        hBoxLayout.addWidget(processButton)
        processButton.clicked.connect(self.confirmationBeforeProcessing)

        newButton = QPushButton("Single Image + All Vision Models")
        hBoxLayout.addWidget(newButton)
        newButton.clicked.connect(self.selectSingleImage)

        self.webView = QWebEngineView()
        custom_page = CustomWebEnginePage(self.webView)
        self.webView.setPage(custom_page)
        script_dir = Path(__file__).resolve().parent
        html_file_path = script_dir / "Assets" / self.HTML_FILE
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
        print(f"Processed {len(documents)} documents")
        contents = self.extract_page_content(documents)
        self.save_page_contents(contents)

    def onProcessingError(self, error_msg):
        self.thread = None
        logging.error(f"Processing error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during image processing:\n\n{error_msg}")

    def extract_page_content(self, documents):
        contents = []
        total_length = 0

        for doc in documents:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                content = doc.page_content
                filepath = doc.metadata.get('source', doc.metadata.get('file_path', doc.metadata.get('file_name', 'Unknown filepath')))
            elif isinstance(doc, dict):
                content = doc.get("page_content", "Document is missing 'page_content'.")
                filepath = doc.get("metadata", {}).get('source', 
                         doc.get("metadata", {}).get('file_path',
                         doc.get("metadata", {}).get('file_name', 'Unknown filepath')))
            else:
                content = "Document is missing 'page_content'."
                filepath = 'Unknown filepath'

            content_length = len(content)
            total_length += content_length
            wrapped_content = textwrap.fill(content, width=100)
            contents.append((filepath, wrapped_content, content_length))

        avg_length = total_length / len(documents) if documents else 0
        return contents, avg_length

    def save_page_contents(self, contents):
        contents, avg_length = contents

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as temp_file:
            temp_file.write(f"Average Summary Length: {avg_length:.2f} characters\n")
            temp_file.write("="*50 + "\n\n")

            for filepath, content, length in contents:
                temp_file.write(f"File Path: {filepath}\n")
                temp_file.write(f"Summary Length: {length} characters\n")
                temp_file.write("-"*50 + "\n")
                temp_file.write(f"{content}\n\n")
                temp_file.write("="*50 + "\n\n")

            temp_name = temp_file.name

        self.open_file(temp_name)

    def save_comparison_results(self, image_path, results):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as temp_file:
            model_col_width = 23
            count_col_width = 12
            time_col_width = 12

            temp_file.write(f"Image Path: {image_path}\n")
            temp_file.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            chunk_advice = "Remember to adjust your 'chunk size' setting to exceed the longest image summary that you expect. For large bodies of text (e.g. from a .pdf) splitting/overlapping chunks of text is fine, but for image summaries you want any/all summaries to fit within a single chunk that will be put into the vector database."
            temp_file.write(textwrap.fill(chunk_advice, width=100) + "\n\n")

            temp_file.write("Model Performance Comparison Table:\n")
            temp_file.write("+" + "-"*model_col_width + "+" + "-"*count_col_width + "+" + "-"*time_col_width + "+\n")
            temp_file.write("|" + "Model Name".center(model_col_width) + "|" + "Char Count".center(count_col_width) + "|" + "Time (sec)".center(time_col_width) + "|\n")
            temp_file.write("+" + "-"*model_col_width + "+" + "-"*count_col_width + "+" + "-"*time_col_width + "+\n")

            for model_name, description, process_time in results:
                temp_file.write("|" + model_name.ljust(model_col_width) + "|" + 
                              str(len(description)).center(count_col_width) + "|" + 
                              f"{process_time:.2f}".center(time_col_width) + "|\n")

            temp_file.write("+" + "-"*model_col_width + "+" + "-"*count_col_width + "+" + "-"*time_col_width + "+\n\n")

            for model_name, description, process_time in results:
                temp_file.write(f"Model: {model_name}\n")
                temp_file.write(f"Summary Length: {len(description)}\n")
                temp_file.write(f"Processing Time: {process_time:.2f} seconds\n")
                temp_file.write("="*50 + "\n")
                if description.strip():
                    temp_file.write(textwrap.fill(description, width=100) + "\n\n")
                else:
                    temp_file.write("[No output generated]\n\n")
                temp_file.write("-"*50 + "\n\n")

            temp_name = temp_file.name
        
        self.open_file(temp_name)
        return temp_name

    def selectSingleImage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.bmp *.tif *.tiff)"
        )
        if file_path:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText(
                "Process this image with all available vision models?\n\n"
                "This will test each model sequentially and may take several minutes.\n"
                "Models will be loaded and unloaded to manage memory usage.\n\n"
                "Do you want to proceed?"
            )
            msgBox.setWindowTitle("Confirm Processing")
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            returnValue = msgBox.exec()

            if returnValue == QMessageBox.Ok:
                self.progress = QProgressDialog("Processing image with multiple models...", "Cancel", 0, len(VISION_MODELS), self)
                self.progress.setWindowModality(Qt.WindowModal)
                self.progress.setWindowTitle("Processing")
                self.progress.canceled.connect(self.cancelProcessing)
                
                self.thread = MultiModelProcessorThread(file_path)
                self.thread.finished.connect(self.onMultiModelProcessingFinished)
                self.thread.error.connect(self.onMultiModelProcessingError)
                self.thread.progress.connect(self.progress.setValue)
                self.thread.start()

    def cancelProcessing(self):
        if self.thread is not None:
            self.thread.cancel()

    def onMultiModelProcessingFinished(self, results):
        self.progress.close()
        try:
            output_file = self.save_comparison_results(self.thread.image_path, results)
            self.open_file(output_file)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving results:\n\n{str(e)}")
        self.thread = None

    def onMultiModelProcessingError(self, error_msg):
        self.progress.close()
        QMessageBox.critical(self, "Error", f"An error occurred during processing:\n\n{error_msg}")
        self.thread = None

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
        except Exception as e:
            error_msg = f"Error opening file: {e}"
            logging.error(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
