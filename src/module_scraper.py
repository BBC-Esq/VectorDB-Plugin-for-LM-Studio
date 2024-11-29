import os
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtWidgets import QMessageBox
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
        By default, each scraped .html file is saved in its entirety.  Subclasses can override and specify only a certain portion.
        If implemented, only the specified portion from each .html file will be saved.
        If NOT implemented, "None" is returned and the default behavior occurs (i.e. the entire .html file is saved).
        """
        return None

class HuggingfaceScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find('div', class_='prose-doc prose relative mx-auto max-w-4xl break-words')

class ReadthedocsScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find('div', class_='rst-content')

class ScraperRegistry:
    _scrapers = {
        "BaseScraper": BaseScraper,
        "HuggingfaceScraper": HuggingfaceScraper,
        "ReadthedocsScraper": ReadthedocsScraper
    }

    @classmethod
    def get_scraper(cls, scraper_name):
        return cls._scrapers.get(scraper_name, BaseScraper)

class ScraperWorker(QObject):
    status_updated = Signal(str)
    scraping_finished = Signal()

    # In ScraperWorker class
    def __init__(self, url, folder, scraper_class=BaseScraper):
        super().__init__()
        self.scraper = scraper_class(url, folder)
        self.url = url
        self.folder = folder
        self.stats = {'scraped': 0}
        self.save_dir = os.path.join(os.path.dirname(__file__), "Scraped_Documentation", self.folder)
        
        # Initialize watchdog observer
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
                # self.status_updated.emit(f"{self.stats['scraped']}")

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
                                # actual_count = self.count_saved_files()
                                # self.stats['scraped'] = actual_count
                                return self.extract_links(html, url, base_domain, acceptable_domain_extension)
                            else:
                                self.stats['scraped'] = self.count_saved_files()
                                return set()
                        else:
                            self.stats['scraped'] = self.count_saved_files()
                            return set()
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
        processed_soup = self.scraper.process_html(soup) # calls process_html from base class
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

    def cleanup(self):
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()

class FileHandler(FileSystemEventHandler):
    def __init__(self, worker):
        self.worker = worker
        
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.html'):
            count = len([f for f in os.listdir(self.worker.save_dir) if f.endswith('.html')])
            self.worker.status_updated.emit(str(count))

class WorkerThread(QThread):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.run()