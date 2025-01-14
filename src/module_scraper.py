import os
import time
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtWidgets import QMessageBox
from charset_normalizer import detect
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class BaseScraper:
    def __init__(self, url, folder):
        self.url = url
        self.folder = folder
        self.save_dir = os.path.join(os.path.dirname(__file__), "Scraped_Documentation", folder)

    def process_html(self, soup):
        main_content = self.extract_main_content(soup)
        if main_content:
            new_soup = BeautifulSoup('<html><body></body></html>', 'lxml')
            new_soup.body.append(main_content)
            return new_soup
        return soup

    def extract_main_content(self, soup):
        """
        By default, each scraped .html file is saved in its entirety unless overridden by a subclass.  If implemented, only the
        specified portion from each .html file will be saved.  Otherwise, "None" is returned and the entire .html file is saved.
        """
        return None

class HuggingfaceScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find('div', class_='prose-doc prose relative mx-auto max-w-4xl break-words')

class ReadthedocsScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find('div', class_='rst-content')

class LangchainScraper(BaseScraper):
    def extract_main_content(self, soup):
        main_content = None

        article_content = soup.find('article', class_='bd-article')
        if article_content:
            main_content = article_content.find('section')
            if main_content:
                return main_content

        main_element = soup.find('main', class_='bd-main')
        if main_element:
            content_div = main_element.find('div', class_='highlight')
            if content_div:
                return content_div
        return soup

class QtForPythonScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find('article', attrs={'role': 'main', 'id': 'furo-main-content'})

class PyTorchScraper(BaseScraper):
    def extract_main_content(self, soup):
        article = soup.find('article', {'class': 'pytorch-article'})
        if article:
            return article

        main_content = soup.find('div', {'class': 'main-content'})
        if main_content:
            return main_content

        main = soup.find(['article', 'div'], {
            'role': 'main',
            'itemtype': 'http://schema.org/Article'
        })
        return main

class TileDBScraper(BaseScraper):
    def extract_main_content(self, soup):
        content = soup.find('div', class_='[&>*+*]:mt-5 grid whitespace-pre-wrap')

        if content:
            new_content = soup.new_tag('article')

            header = soup.find('header', class_='max-w-3xl mx-auto mb-6 space-y-3')
            if header:
                new_content.append(header)

            new_content.append(content)

            return new_content

        return None

class TileDBVectorSearchScraper(BaseScraper):
    def extract_main_content(self, soup):
        header = soup.find('header', id='title-block-header')
        content = soup.find('section', id='benchmarks')

        if content:
            new_content = soup.new_tag('article')

            if header:
                new_content.append(header)

            new_content.append(content)

            return new_content

        return None

class ScraperRegistry:
    _scrapers = {
        "BaseScraper": BaseScraper,
        "HuggingfaceScraper": HuggingfaceScraper, 
        "ReadthedocsScraper": ReadthedocsScraper,
        "LangchainScraper": LangchainScraper,
        "QtForPythonScraper": QtForPythonScraper,
        "PyTorchScraper": PyTorchScraper,
        "TileDBScraper": TileDBScraper,
        "TileDBVectorSearchScraper": TileDBVectorSearchScraper,
    }

    @classmethod
    def get_scraper(cls, scraper_name):
        return cls._scrapers.get(scraper_name, BaseScraper)

class ScraperWorker(QObject):
    status_updated = Signal(str)
    scraping_finished = Signal()

    def __init__(self, url, folder, scraper_class=BaseScraper):
        super().__init__()
        self.scraper = scraper_class(url, folder)
        self.url = url
        self.folder = folder
        self.stats = {'scraped': 0}
        self.save_dir = os.path.join(os.path.dirname(__file__), "Scraped_Documentation", self.folder)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.observer = Observer()
        handler = FileHandler(self)
        self.observer.schedule(handler, self.save_dir, recursive=False)
        self.observer.start()

    def run(self):
        asyncio.run(self.crawl_domain())

    def count_saved_files(self):
        if not os.path.exists(self.save_dir):
            return 0
        count = len([f for f in os.listdir(self.save_dir) if f.endswith('.html')])
        return count

    async def crawl_domain(self, max_concurrent_requests=100, batch_size=100):
        parsed_url = urlparse(self.url)
        acceptable_domain = parsed_url.netloc
        acceptable_domain_extension = parsed_url.path.rstrip('/')

        save_dir = self.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        log_file = os.path.join(save_dir, "failed_urls.log")

        semaphore = asyncio.BoundedSemaphore(max_concurrent_requests)
        to_visit = [self.url]
        visited = set()

        async def process_batch(batch_urls, session):
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

                results = await process_batch(current_batch, session)

                for new_links in results:
                    if new_links:
                        new_to_visit = new_links - visited
                        to_visit.extend(new_to_visit)

                await asyncio.sleep(0.2)

        self.scraping_finished.emit()
        return visited

    async def fetch(self, session, url, base_domain, semaphore, save_dir, log_file, acceptable_domain_extension, retries=3):
        filename = os.path.join(save_dir, self.sanitize_filename(url) + ".html")

        if os.path.exists(filename):
            return set()

        fallback_encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'windows-1252', 'ascii']
        headers = {
            'Accept-Charset': 'utf-8, iso-8859-1;q=0.8, *;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        async with semaphore:
            for attempt in range(1, retries + 1):
                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            content_type = response.headers.get('content-type', '').lower()
                            if 'text/html' in content_type:
                                try:
                                    html = await response.text(encoding='utf-8')
                                except UnicodeDecodeError:
                                    raw = await response.read()
                                    try:
                                        detected = detect(raw)
                                        detected_encoding = detected.get('encoding')
                                        if not detected_encoding or detected_encoding.lower() in ['unknown', 'utf-8']:
                                            detected_encoding = 'latin-1'
                                    except Exception as e:
                                        detected_encoding = 'latin-1'

                                    try:
                                        html = raw.decode(detected_encoding)
                                    except UnicodeDecodeError:
                                        for encoding in fallback_encodings:
                                            try:
                                                html = raw.decode(encoding)
                                                break
                                            except UnicodeDecodeError:
                                                continue
                                        else:
                                            html = raw.decode('utf-8', errors='ignore')

                                await self.save_html(html, url, save_dir)
                                self.stats['scraped'] = self.count_saved_files()
                                return self.extract_links(html, url, base_domain, acceptable_domain_extension)
                            else:
                                self.stats['scraped'] = self.count_saved_files()
                                return set()
                        else:
                            self.stats['scraped'] = self.count_saved_files()
                            return set()
                except UnicodeDecodeError:
                    if attempt == retries:
                        await self.log_failed_url(url, log_file)
                        self.stats['scraped'] = self.count_saved_files()
                    await asyncio.sleep(2)  # Longer delay for encoding issues
                except Exception as e:
                    if attempt == retries:
                        await self.log_failed_url(url, log_file)
                        self.stats['scraped'] = self.count_saved_files()
                        QMessageBox.critical(None, "Error", f"Failed to fetch URL: {url}\nError: {str(e)}")
                    await asyncio.sleep(1)
            return set()

    async def save_html(self, content, url, save_dir):
        filename = os.path.join(save_dir, self.sanitize_filename(url) + ".html")
        soup = BeautifulSoup(content, 'lxml')
        processed_soup = self.scraper.process_html(soup)
        source_link = processed_soup.new_tag("a", href=url)
        source_link.string = "Original Source"

        if processed_soup.body:
            processed_soup.body.insert(0, source_link)
        elif processed_soup.html:
            new_body = processed_soup.new_tag("body")
            new_body.insert(0, source_link)
            processed_soup.html.insert(0, new_body)
        else:
            new_html = processed_soup.new_tag("html")
            new_body = processed_soup.new_tag("body")
            new_body.insert(0, source_link)
            new_html.insert(0, new_body)
            processed_soup.insert(0, new_html)

        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(str(processed_soup))

    def sanitize_filename(self, url):
        url = url.split('#')[0]

        while '[' in url and ']' in url:
            start = url.find('[')
            end = url.find(']')
            if start < end:
                url = url[:start] + url[end+1:]

        while '(' in url and ')' in url:
            start = url.find('(')
            end = url.find(')')
            if start < end:
                url = url[:start] + url[end+1:]

        filename = url.replace("https://", "").replace("http://", "")

        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')

        if filename.lower().endswith('.html'):
            filename = filename[:-5]

        if len(filename) > 200:
            filename = filename[:200]

        filename = filename.rstrip('. ')

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
        return parsed_url.netloc == base_domain and parsed_url.path.startswith(acceptable_domain_extension)

    def cleanup(self):
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()

class FileHandler(FileSystemEventHandler):
    def __init__(self, worker, throttle_interval=0.5):
        self.worker = worker
        self.throttle_interval = throttle_interval
        self.last_execution_time = 0

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.html'):
            current_time = time.time()
            if current_time - self.last_execution_time >= self.throttle_interval:
                self.last_execution_time = current_time
                self.execute_handler()

    def execute_handler(self):
        count = len([f for f in os.listdir(self.worker.save_dir) if f.endswith('.html')])
        self.worker.status_updated.emit(str(count))

class WorkerThread(QThread):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.run()