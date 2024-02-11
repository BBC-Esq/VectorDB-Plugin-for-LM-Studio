from pathlib import Path
from PySide6.QtWidgets import QMessageBox, QApplication
import shutil
import platform
import os
import yaml
import gc
import sys
from termcolor import cprint

def is_nvidia_gpu_available():
    return torch.cuda.is_available() and "nvidia" in torch.cuda.get_device_name(0).lower()

if is_nvidia_gpu_available():
    import pynvml

def validate_symbolic_links(source_directory):
    source_path = Path(source_directory)
    symbolic_links = [entry for entry in source_path.iterdir() if entry.is_symlink()]

    for symlink in symbolic_links:
        target_path = symlink.resolve(strict=False)

        if not target_path.exists():
            print(f"Warning: Symbolic link {symlink.name} points to a missing file. It will be skipped.")
            symlink.unlink()

def load_stylesheet(filename):
    script_dir = Path(__file__).parent
    stylesheet_path = script_dir / 'CSS' / filename
    with stylesheet_path.open('r') as file:
        stylesheet = file.read()
    return stylesheet

def list_theme_files():
    script_dir = Path(__file__).parent
    theme_dir = script_dir / 'CSS'
    return [f.name for f in theme_dir.iterdir() if f.suffix == '.css']

def make_theme_changer(theme_name):
    def change_theme():
        stylesheet = load_stylesheet(theme_name)
        QApplication.instance().setStyleSheet(stylesheet)
    return change_theme

def backup_database():
    source_directory = Path('Vector_DB')
    backup_directory = Path('Vector_DB_Backup')

    if backup_directory.exists():
        for item in backup_directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        backup_directory.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source_directory, backup_directory, dirs_exist_ok=True)

# gui_tabs_databases.py
def open_file(file_path):
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", file_path])
        else:
            subprocess.Popen(["xdg-open", file_path])
    except OSError:
        QMessageBox.warning(None, "Error", "No default viewer detected.")

# gui_tabs_databases.py
def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        QMessageBox.warning(None, "Unable to delete file(s), please delete manually.")

# database_interactions.py
def check_preconditions_for_db_creation(script_dir):
    config_path = script_dir / 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if platform.system() == "Darwin" and any((script_dir / "Images_for_DB").iterdir()):
        return False, "Image processing has been disabled for MacOS for the time being until a fix can be implemented. Please remove all files from the 'Images_for_DB' folder and try again."

    embedding_model_name = config.get('EMBEDDING_MODEL_NAME')
    if not embedding_model_name:
        return False, "You must first download an embedding model, select it, and choose documents first before proceeding."

    documents_dir = script_dir / "Docs_for_DB"
    images_dir = script_dir / "Images_for_DB"
    if not any(documents_dir.iterdir()) and not any(images_dir.iterdir()):
        return False, "No documents found to process. Please select files to add to the vector database and try again."

    compute_device = config.get('Compute_Device', {}).get('available', [])
    database_creation = config.get('Compute_Device', {}).get('database_creation')
    if ("cuda" in compute_device or "mps" in compute_device) and database_creation == "cpu":
        reply = QMessageBox.question(None, 'Warning', 
                                     "GPU-acceleration is available and highly recommended for creating a vector database. Click OK to proceed or Cancel to go back and change the device.", 
                                     QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return False, ""

    confirmation_reply = QMessageBox.question(None, 'Confirmation', 
                                             "Creating a vector database can take a significant amount of time and cannot be cancelled mid-processing. Click OK to proceed or Cancel to back out.",
                                             QMessageBox.Ok | QMessageBox.Cancel)
    if confirmation_reply == QMessageBox.Cancel:
        return False, "Database creation cancelled by user."

    return True, ""

# gui.py
def check_preconditions_for_submit_question(script_dir):
    config_path = script_dir / 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    embedding_model_name = config.get('EMBEDDING_MODEL_NAME')
    if not embedding_model_name:
        return False, "You must first download an embedding model, select it, and choose documents first before proceeding."

    documents_dir = script_dir / "Docs_for_DB"
    images_dir = script_dir / "Images_for_DB"
    if not any(documents_dir.iterdir()) and not any(images_dir.iterdir()):
        return False, "No documents found to process. Please select files to add to the vector database and try again."

    vector_db_dir = script_dir / "Vector_DB"
    if not any(f.suffix == '.parquet' for f in vector_db_dir.iterdir()):
        return False, "You must first create a vector database before clicking this button."

    return True, ""

def my_cprint(*args, **kwargs):
    filename = os.path.basename(sys._getframe(1).f_code.co_filename)
    modified_message = f"{filename}: {args[0]}"
    kwargs['flush'] = True
    cprint(modified_message, *args[1:], **kwargs)
    
def print_cuda_memory_usage():
    '''
    from utilities import print_cuda_memory_usage
    print_cuda_memory_usage()
    '''
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Memory Total: {memory_info.total / 1024**2} MB") 
        print(f"Memory Used: {memory_info.used / 1024**2} MB")
        print(f"Memory Free: {memory_info.free / 1024**2} MB")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        pynvml.nvmlShutdown()

def check_for_object_references(obj):
    '''
    from utilities import check_for_object_references
    my_list = [1, 2, 3, 4, 5]
    check_for_object_references(my_list)
    '''
    script_dir = os.path.dirname(__file__)
    referrers_file_path = os.path.join(script_dir, "references.txt")

    with open(referrers_file_path, "w", encoding='utf-8') as file:
        referrers = gc.get_referrers(obj)
        file.write(f"Number of references found: {len(referrers)}\n")
        for ref in referrers:
            file.write(str(ref) + "\n")

def get_cuda_compute_capabilities():
    ccs = []
    for i in range(torch.cuda.device_count()):
        cc_major, cc_minor = torch.cuda.get_device_capability(torch.cuda.device(i))
        ccs.append(f"{cc_major}.{cc_minor}")

    return ccs

def get_cuda_version():
    major, minor = map(int, torch.version.cuda.split("."))

    return f'{major}{minor}'