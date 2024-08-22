import gc
import importlib
import importlib.metadata
import importlib.util
import os
import logging
import platform
import shutil
import sys
from pathlib import Path
import pickle
import re

import torch
import yaml
from packaging import version
from PySide6.QtWidgets import QApplication, QMessageBox
from termcolor import cprint

# IMPLEMENT THIS IF/WHEN A USER TRIES TO CREATE A DB WITH A CPU WITH AN INCOMPATIBLE VISION MODEL
def cpu_db_creation_vision_model_compatibility(directory, image_extensions, config_path):
    has_images = False
    for root, _, files in os.walk(directory):
        if any(file.lower().endswith(ext) for file in files for ext in image_extensions):
            has_images = True
            break
    
    if not has_images:
        return False, None

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    compute_device = config.get('Compute_Device', {}).get('database_creation', 'cpu')
    
    if compute_device.lower() == 'cpu':
        return True, None

    embedding_model = config.get('EMBEDDING_MODEL_NAME', '').lower()
    if not (embedding_model.endswith('florence-2-base') or embedding_model.endswith('florence-2-large')):
        message = ("You've selected one or more images to process but have selected an incompatible vision model "
                   "when creating the database with a CPU. Please select either 'Florence-2-base' or 'Florence-2-large'.")
        return True, message

    return True, None


def get_pkl_file_path(pkl_file_path):
    # used in gui_tabs_databases.py when opening a file
    try:
        with open(pkl_file_path, 'rb') as file:
            document = pickle.load(file)
        internal_file_path = document.metadata.get('file_path')
        if internal_file_path and Path(internal_file_path).exists():
            return internal_file_path
        else:
            return None
    except Exception as e:
        raise ValueError(f"Could not process pickle file: {e}")

def print_first_citation_metadata(metadata_list):
    """
    DEBUG: Print the metadata attributes/fields for the first citation in the list.
    """
    if metadata_list:
        print("Metadata attributes/fields for the first citation:")
        for key, value in metadata_list[0].items():
            print(f"{key}: {value}")

def format_citations(metadata_list):
    """
    Format metadata into an HTML list after removing redundant ones.
    """
    unique_citations = set()
    for metadata in metadata_list:
        file_path = metadata['file_path']
        file_name = Path(file_path).name
        unique_citations.add(f'<a href="file:{file_path}" style="color:#DAA520;">{file_name}</a>')
    
    list_items = "".join(f"<li>{citation}</li>" for citation in sorted(unique_citations))
    
    return f"<ol>{list_items}</ol>"


def get_physical_core_count():
    return psutil.cpu_count(logical=False)


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def is_nvidia_gpu_available():
    return torch.cuda.is_available() and torch.version.cuda is not None

config = load_config('config.yaml')


# Import pynvml only if CUDA is available and an NVIDIA GPU is detected
if is_nvidia_gpu_available():
    import pynvml

    def print_cuda_memory_usage():
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
else:
    def print_cuda_memory_usage():
        print("CUDA is not available, or no NVIDIA GPU detected.")


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


def open_file(file_path):
    # open a file with the system's default program
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", file_path])
        else:
            subprocess.Popen(["xdg-open", file_path])
    except OSError:
        QMessageBox.warning(None, "Error", "No default viewer detected.")


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        QMessageBox.warning(None, "Unable to delete file(s), please delete manually.")


def check_preconditions_for_db_creation(script_dir, database_name):
    # is db name valid
    if not database_name or len(database_name) < 3 or database_name.lower() in ["null", "none"]:
        QMessageBox.warning(None, "Invalid Name", "Name must be at least 3 characters long and not be 'null' or 'none.'")
        return False, "Invalid database name."

    # is the db name already used
    database_folder_path = script_dir / "Docs_for_DB" / database_name
    if database_folder_path.exists():
        QMessageBox.warning(None, "Database Exists", "A database with this name already exists. Please choose a different database name.")
        return False, "Database already exists."

    # does config.yaml exist
    config_path = script_dir / 'config.yaml'
    if not config_path.exists():
        QMessageBox.warning(None, "Configuration Missing", "The configuration file (config.yaml) is missing.")
        return False, "Configuration file missing."
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # can't process images on mac
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']
    documents_dir = script_dir / "Docs_for_DB"
    if platform.system() == "Darwin" and any(file.suffix in image_extensions for file in documents_dir.iterdir() if file.is_file()):
        QMessageBox.warning(None, "MacOS Limitation", "Image processing has been disabled for MacOS until a fix can be implemented. Please remove all image files and try again.")
        return False, "Image files present on MacOS."

    # is vector model selected
    embedding_model_name = config.get('EMBEDDING_MODEL_NAME')
    if not embedding_model_name:
        QMessageBox.warning(None, "Model Missing", "You must first download an embedding model, select it, and choose documents first before proceeding.")
        return False, "Embedding model not selected."

    # are documents selected
    if not any(file.is_file() for file in documents_dir.iterdir()):
        QMessageBox.warning(None, "No Documents", "No documents are yet added to be processed.")
        return False, "No documents in Docs_for_DB."

    # is gpu-acceleration selected
    compute_device = config.get('Compute_Device', {}).get('available', [])
    database_creation = config.get('Compute_Device', {}).get('database_creation')
    if ("cuda" in compute_device or "mps" in compute_device) and database_creation == "cpu":
        reply = QMessageBox.question(None, 'Warning', 
                                     "GPU-acceleration is available and highly recommended. Click OK to proceed or Cancel to go back and change the device.", 
                                     QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return False, "User cancelled operation based on device check."

    # if no cuda and half selected, inform user and exit early
    if not torch.cuda.is_available():
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if config.get('database', {}).get('half', False):
            message = ("CUDA is not available on your system, but half-precision (FP16) "
                       "is selected for database creation. Half-precision requires CUDA. "
                       "Please disable half-precision in the configuration or use a CUDA-enabled GPU.")
            QMessageBox.warning(None, "CUDA Unavailable for Half-Precision", message)
            return False, "CUDA unavailable for half-precision operation."

    # final confirmation
    confirmation_reply = QMessageBox.question(None, 'Confirmation', 
                                             "Creating a vector database can take a significant amount of time and cannot be cancelled. Click OK to proceed.",
                                             QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
    if confirmation_reply == QMessageBox.Cancel:
        return False, "Database creation cancelled by user."

    return True, ""


# gui.py
def check_preconditions_for_submit_question(script_dir):
    config_path = script_dir / 'config.yaml'

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    database_to_search = config.get('database', {}).get('database_to_search')

    vector_db_subdir = script_dir / "Vector_DB" / str(database_to_search) if database_to_search else None

    return True, ""


def my_cprint(*args, **kwargs):
    filename = os.path.basename(sys._getframe(1).f_code.co_filename)
    modified_message = f"{args[0]}"
    # modified_message = f"{filename}: {args[0]}" # uncomment to print script name as well
    kwargs['flush'] = True
    cprint(modified_message, *args[1:], **kwargs)
    
    
def print_cuda_memory_usage():
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


def get_cuda_compute_capabilities():
    ccs = []
    for i in range(torch.cuda.device_count()):
        cc_major, cc_minor = torch.cuda.get_device_capability(torch.cuda.device(i))
        ccs.append(f"{cc_major}.{cc_minor}")

    return ccs


def get_cuda_version():
    major, minor = map(int, torch.version.cuda.split("."))

    return f'{major}{minor}'


# returns True if cuda exists and supports compute 8.6 of higher
def has_bfloat16_support():
    if not torch.cuda.is_available():
        return False
    
    capability = torch.cuda.get_device_capability()
    return capability >= (8, 6)


def get_precision():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a CUDA-enabled GPU.")

    capability = torch.cuda.get_device_capability()
    if capability >= (8, 6):
        precision = torch.bfloat16
    else:
        precision = torch.float16
    
    return precision


def get_device_and_precision():
    if torch.cuda.is_available():
        device = "cuda"
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 6):
            precision = "bfloat16"
        else:
            precision = "float16"
    else:
        device = "cpu"
        precision = "float32"
    return device, precision


class FlashAttentionUtils:
    @staticmethod
    def check_package_availability():
        # Check if flash_attn is installed
        return importlib.util.find_spec("flash_attn") is not None

    @staticmethod
    def check_version_compatibility():
        # Check flash_attn version
        if not FlashAttentionUtils.check_package_availability():
            return False
        flash_attention_version = version.parse(importlib.metadata.version("flash_attn"))
        if torch.version.cuda:
            return flash_attention_version >= version.parse("2.1.0")
        elif torch.version.hip:
            return flash_attention_version >= version.parse("2.0.4")
        return False

    @staticmethod
    def check_dtype_compatibility(dtype):
        # Check if dtype is compatible
        return dtype in [torch.float16, torch.bfloat16]

    @staticmethod
    def check_gpu_initialization():
        # Check if CUDA is available and default device is CUDA
        return torch.cuda.is_available() and torch.cuda.current_device() >= 0

    @staticmethod
    def check_device_map(device_map):
        # Check if device_map is compatible
        if device_map is None:
            return True
        if isinstance(device_map, dict):
            return "cpu" not in device_map.values() and "disk" not in device_map.values()
        return True

    @classmethod
    def is_flash_attention_compatible(cls, dtype=None, device_map=None):
        # Run all checks
        checks = [
            cls.check_package_availability(),
            cls.check_version_compatibility(),
            cls.check_dtype_compatibility(dtype) if dtype else True,
            cls.check_gpu_initialization(),
            cls.check_device_map(device_map)
        ]
        return all(checks)

    @staticmethod
    def enable_flash_attention(config):
        # Enable Flash Attention in the config
        config._attn_implementation = "flash_attention_2"
        return config



'''
open a text file on multiple platforms
    try:
        if os.name == 'nt':
            os.startfile(output_file_path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', output_file_path])
        elif sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', output_file_path])
        else:
            raise NotImplementedError("Unsupported operating system")
    except Exception as e:
        print(f"Error opening file: {e}")
'''

def set_logging_level():
    """
    CRITICAL displays only CRITICAL.
    ERROR displays ERROR and CRITICAL.
    WARNING displays WARNING, ERROR, and CRITICAL.
    INFO displays INFO, WARNING, ERROR, and CRITICAL.
    DEBUG displays DEBUG, INFO, WARNING, ERROR, and CRITICAL.
    """
    library_levels = {
        "accelerate": logging.WARNING,
        "bitsandbytes": logging.WARNING,
        "ctranslate2": logging.WARNING,
        "datasets": logging.WARNING,
        "einops": logging.WARNING,
        "einx": logging.WARNING,
        "flash_attn": logging.WARNING,
        "huggingface-hub": logging.WARNING,
        "langchain": logging.WARNING,
        "langchain-community": logging.WARNING,
        "langchain-core": logging.WARNING,
        "langchain-huggingface": logging.WARNING,
        "langchain-text-splitters": logging.WARNING,
        "numpy": logging.WARNING,
        "openai": logging.WARNING,
        "openai-whisper": logging.WARNING,
        "optimum": logging.WARNING,
        "pillow": logging.WARNING,
        "requests": logging.WARNING,
        "sentence-transformers": logging.WARNING,
        "sounddevice": logging.WARNING,
        "speechbrain": logging.WARNING,
        "sympy": logging.WARNING,
        "tiledb": logging.WARNING,
        "tiledb-cloud": logging.WARNING,
        "tiledb-vector-search": logging.WARNING,
        "timm": logging.WARNING,
        "tokenizers": logging.WARNING,
        "torch": logging.WARNING,
        "torchaudio": logging.WARNING,
        "torchvision": logging.WARNING,
        "transformers": logging.WARNING,
        "unstructured": logging.WARNING,
        "unstructured-client": logging.WARNING,
        "vector-quantize-pytorch": logging.WARNING,
        "vocos": logging.WARNING,
        "xformers": logging.WARNING
    }

    for lib, level in library_levels.items():
        logging.getLogger(lib).setLevel(level)

