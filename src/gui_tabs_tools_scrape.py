import os
import asyncio
import aiohttp
import aiofiles
import subprocess
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PySide6.QtWidgets import (
   QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QStandardItem, QStandardItemModel, QColor
import platform
import shutil
from constants import scrape_documentation

class AsyncWorker(QObject):
   status_updated = Signal(str)
   scraping_finished = Signal()

   def __init__(self, url, folder):
       super().__init__()
       self.url = url
       self.folder = folder
       self.stats = {'scraped': 0}
       self.save_dir = os.path.join(os.path.dirname(__file__), "Scraped_Documentation", self.folder)

   def run(self):
       asyncio.run(self.crawl_domain())

   def count_saved_files(self):
       count = len([f for f in os.listdir(self.save_dir) if f.endswith('.html')])
       return count

   async def crawl_domain(self, max_concurrent_requests=10, batch_size=50):
       parsed_url = urlparse(self.url)
       acceptable_domain = parsed_url.netloc
       acceptable_domain_extension = parsed_url.path.rstrip('/')

       save_dir = os.path.join(os.path.dirname(__file__), "Scraped_Documentation", self.folder)
       if not os.path.exists(save_dir):
           os.makedirs(save_dir)

       log_file = os.path.join(save_dir, "failed_urls.log")

       semaphore = asyncio.BoundedSemaphore(max_concurrent_requests)
       to_visit = [self.url]
       visited = set()

       async def process_batch(batch_urls):
           tasks = []
           for url in batch_urls:
               if url not in visited:
                   visited.add(url)
                   task = self.fetch(session, url, acceptable_domain, semaphore, save_dir, log_file, acceptable_domain_extension)
                   tasks.append(task)
           return await asyncio.gather(*tasks)

       async with aiohttp.ClientSession() as session:
           while to_visit:
               current_batch = to_visit[:batch_size]
               to_visit = to_visit[batch_size:]
               
               results = await process_batch(current_batch)
               
               for new_links in results:
                   if new_links:
                       new_to_visit = new_links - visited
                       to_visit.extend(new_to_visit)
               
               await asyncio.sleep(0.1)
               self.status_updated.emit(f"{self.stats['scraped']}")

       self.scraping_finished.emit()
       return visited

   async def fetch(self, session, url, base_domain, semaphore, save_dir, log_file, acceptable_domain_extension, retries=3):
       filename = os.path.join(save_dir, self.sanitize_filename(url) + ".html")
       
       if os.path.exists(filename):
           return set()
           
       async with semaphore:
           for attempt in range(1, retries + 1):
               try:
                   async with session.get(url) as response:
                       if response.status == 200:
                           content_type = response.headers.get('content-type', '').lower()
                           if 'text/html' in content_type:
                               html = await response.text()
                               await self.save_html(html, url, save_dir)
                               actual_count = self.count_saved_files()
                               self.stats['scraped'] = actual_count
                               return self.extract_links(html, url, base_domain, acceptable_domain_extension)
                           else:
                               self.stats['scraped'] = self.count_saved_files()
                               return set()
                       else:
                           self.stats['scraped'] = self.count_saved_files()
                           return set()
               except Exception:
                   if attempt == retries:
                       await self.log_failed_url(url, log_file)
                       self.stats['scraped'] = self.count_saved_files()
                   await asyncio.sleep(1)
           return set()

   async def save_html(self, content, url, save_dir):
       filename = os.path.join(save_dir, self.sanitize_filename(url) + ".html")
       soup = BeautifulSoup(content, 'lxml')
       
       source_link = soup.new_tag("a", href=url)
       source_link.string = "Original Source"
       
       if soup.body:
           soup.body.insert(0, source_link)
       elif soup.html:
           new_body = soup.new_tag("body")
           new_body.insert(0, source_link)
           soup.html.insert(0, new_body)
       else:
           new_html = soup.new_tag("html")
           new_body = soup.new_tag("body")
           new_body.insert(0, source_link)
           new_html.insert(0, new_body)
           soup.insert(0, new_html)
       
       async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
           await f.write(str(soup))

   def sanitize_filename(self, url):
       url = url.split('#')[0]
       filename = url.replace("https://", "").replace("http://", "").replace("/", "_").replace("?", "_")
       if filename.endswith('.html'):
           filename = filename[:-5]
       return filename

   async def log_failed_url(self, url, log_file):
       async with aiofiles.open(log_file, 'a') as f:
           await f.write(url + "\n")

   def extract_links(self, html, base_url, base_domain, acceptable_domain_extension):
       soup = BeautifulSoup(html, 'lxml')
       links = set()

       for a_tag in soup.find_all('a', href=True):
           href = a_tag['href']
           href = href.replace('&amp;num;', '#')
           if href.startswith('/'):
               url = urljoin(f"https://{base_domain}", href)
           else:
               url = urljoin(base_url, href)
           
           url = url.split('#')[0]

           if self.is_valid_url(url, base_domain, acceptable_domain_extension):
               links.add(url)

       return links

   def is_valid_url(self, url, base_domain, acceptable_domain_extension):
       parsed_url = urlparse(url)
       return parsed_url.netloc == base_domain and acceptable_domain_extension in parsed_url.path

class WorkerThread(QThread):
   def __init__(self, worker):
       super().__init__()
       self.worker = worker

   def run(self):
       self.worker.run()

class ScrapeDocumentationTab(QWidget):
   def __init__(self):
       super().__init__()
       self.setToolTip("Tab for scraping documentation from the selected source.")
       self.init_ui()

   def init_ui(self):
       main_layout = QVBoxLayout(self)

       label = QLabel("Select Documentation:")
       label.setToolTip("Choose the documentation set you want to scrape from the comobox.")
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

       self.worker = AsyncWorker(url, folder)
       self.worker.status_updated.connect(self.update_status)
       self.worker.scraping_finished.connect(self.scraping_finished)

       self.thread = WorkerThread(self.worker)
       self.thread.start()

   def update_status(self, status):
       self.status_label.setText(f'Pages scraped: {status} <a href="open_folder">Open Folder</a>')

   def scraping_finished(self):
       self.scrape_button.setEnabled(True)
       final_count = len([f for f in os.listdir(self.current_folder) if f.endswith('.html')])
       self.status_label.setText(f'Scraping completed. Pages scraped: {final_count} <a href="open_folder">Open Folder</a>')
       self.populate_combo_box()

   def open_folder(self, link):
       if link == "open_folder":
           if platform.system() == "Windows":
               os.startfile(self.current_folder)
           elif platform.system() == "Darwin":
               subprocess.Popen(["open", self.current_folder])
           else:
               subprocess.Popen(["xdg-open", self.current_folder])