import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the name of the package you want to install
package_name = "langchain==0.0.341"

# Run the pip install command with subprocess
install_command = ["pip", "install", package_name]

# Create a list to store the created directories
created_directories = []

# Function to track directory creation during installation
def track_directory_creation(event_type, src_path):
    if event_type == "created" and os.path.isdir(src_path):
        created_directories.append(src_path)

# Start tracking directory creation using Watchdog
class DirectoryCreationHandler(FileSystemEventHandler):
    def on_created(self, event):
        track_directory_creation("created", event.src_path)

observer = Observer()
observer.schedule(DirectoryCreationHandler(), path='.', recursive=True)
observer.start()

# Run the pip install command
try:
    subprocess.check_call(install_command)
except subprocess.CalledProcessError:
    print(f"Failed to install {package_name}")

# Stop the directory creation tracking
observer.stop()
observer.join()

# Filter and save only the paths that include "site-packages\langchain"
filtered_directories = []

for directory in created_directories:
    components = directory.split(os.path.sep)
    langchain_index = components.index("site-packages") + 1
    
    if langchain_index < len(components) and components[langchain_index] == "langchain":
        filtered_path = os.path.join(*components[:langchain_index+1])
        filtered_directories.append(filtered_path)

# Save the filtered directory paths to a text file
with open("filtered_directories.txt", "w") as file:
    for directory in filtered_directories:
        file.write(f"{directory}\n")

print(f"Filtered directories saved to 'filtered_directories.txt'")
