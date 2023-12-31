import subprocess
import os
import shutil
import hashlib
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

package_name = "langchain==0.0.341"

install_command = ["pip", "install", package_name]

created_directories = []

def track_directory_creation(event_type, src_path):
    if event_type == "created" and os.path.isdir(src_path):
        created_directories.append(src_path)

class DirectoryCreationHandler(FileSystemEventHandler):
    def on_created(self, event):
        track_directory_creation("created", event.src_path)

observer = Observer()
observer.schedule(DirectoryCreationHandler(), path='.', recursive=True)
observer.start()

try:
    subprocess.check_call(install_command)
except subprocess.CalledProcessError:
    print(f"Failed to install {package_name}")
    observer.stop()
    observer.join()
    sys.exit()

observer.stop()
observer.join()

filtered_directories = []

for directory in created_directories:
    if "site-packages/langchain" in directory.replace("\\", "/"):
        base_path = directory.split("site-packages\\langchain")[0]
        filtered_directories.append(base_path)

base_path = filtered_directories[0].replace("\\", "/") if filtered_directories else ""
final_path = f"{base_path}site-packages/langchain/document_loaders/parsers/pdf.py"

print(f"Target path: {final_path}")

def calculate_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

source_path = "User_Manual/pdf.py"
print(f"Source path: {source_path}")

try:
    source_hash = calculate_hash(source_path)
    try:
        target_hash = calculate_hash(final_path)
    except FileNotFoundError:
        target_hash = None

    if source_hash != target_hash:
        shutil.copy(source_path, final_path)
        print("File copied as the hashes are different.")
    else:
        print("Files are identical. No action taken.")
except FileNotFoundError:
    print("Warning: pdf.py not found in User_Manual folder.")
except Exception as e:
    print(f"An error occurred: {e}")

print(f"Installation and file copying completed successfully.")
