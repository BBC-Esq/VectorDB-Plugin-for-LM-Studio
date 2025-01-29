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
import psutil
import subprocess
import re

import torch
import yaml
from packaging import version
from PySide6.QtCore import QRunnable, QObject, Signal, QThreadPool
from PySide6.QtWidgets import QApplication, QMessageBox
from termcolor import cprint

class DownloadSignals(QObject):
    finished = Signal(bool, str)
    progress = Signal(str)

class DownloadRunnable(QRunnable):
    def __init__(self, download_func, *args):
        super().__init__()
        self.download_func = download_func
        self.args = args
        self.signals = DownloadSignals()

    def run(self):
        try:
            result = self.download_func(*self.args)
            self.signals.finished.emit(result, "Download completed successfully")
        except Exception as e:
            self.signals.finished.emit(False, str(e))

def download_with_threadpool(download_func, *args, callback=None):
    runnable = DownloadRunnable(download_func, *args)
    if callback:
        runnable.signals.finished.connect(callback)
    QThreadPool.globalInstance().start(runnable)

def download_kokoro_tts():
    from pathlib import Path
    from huggingface_hub import snapshot_download
    import shutil

    repo_id = "ctranslate2-4you/Kokoro-82M-light"
    tts_path = Path(__file__).parent / "Models" / "tts" / "ctranslate2-4you--Kokoro-82M-light"
    
    try:
        # Create parent directories if they don't exist
        tts_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading Kokoro TTS model from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(tts_path),
            max_workers=4
        )
        print("Kokoro TTS model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"Failed to download Kokoro TTS model: {e}")
        if tts_path.exists():
            shutil.rmtree(tts_path)
        return False

def download_kobold_executable():
    import requests
    from pathlib import Path

    file_name = "koboldcpp_nocuda.exe"
    url = f"https://github.com/LostRuins/koboldcpp/releases/download/v1.80.3/{file_name}"

    script_dir = Path(__file__).parent
    assets_dir = script_dir / "Assets"
    assets_dir.mkdir(exist_ok=True)

    kobold_path = assets_dir / file_name

    try:
        print(f"Downloading KoboldCPP from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(kobold_path, 'wb') as file:
            file.write(response.content)
        print(f"KoboldCPP downloaded successfully to {kobold_path}")
        return True
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while downloading KoboldCPP: {http_err}")
        return False
    except Exception as e:
        print(f"Failed to download KoboldCPP: {e}")
        return False

def normalize_chat_text(text):
    """
    Normalizes chat text by processing numbers, currency, and various text patterns.
    
    Args:
        text (str): The input text to normalize
        
    Returns:
        str: The normalized text
    """
    def split_num(num):
        num = num.group()
        if '.' in num:
            return num
        elif ':' in num:
            h, m = [int(n) for n in num.split(':')]
            if m == 0:
                return f"{h} o'clock"
            elif m < 10:
                return f'{h} oh {m}'
            return f'{h} {m}'
        year = int(num[:4])
        if year < 1100 or year % 1000 < 10:
            return num
        left, right = num[:2], int(num[2:4])
        s = 's' if num.endswith('s') else ''
        if 100 <= year % 1000 <= 999:
            if right == 0:
                return f'{left} hundred{s}'
            elif right < 10:
                return f'{left} oh {right}{s}'
        return f'{left} {right}{s}'

    def flip_money(m):
        m = m.group()
        bill = 'dollar' if m[0] == '$' else 'pound'
        if m[-1].isalpha():
            return f'{m[1:]} {bill}s'
        elif '.' not in m:
            s = '' if m[1:] == '1' else 's'
            return f'{m[1:]} {bill}{s}'
        b, c = m[1:].split('.')
        s = '' if b == '1' else 's'
        c = int(c.ljust(2, '0'))
        coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
        return f'{b} {bill}{s} and {c} {coins}'

    def point_num(num):
        a, b = num.group().split('.')
        return ' point '.join([a, ' '.join(b)])

    # Replace section symbol
    text = text.replace('§', ' section ')
    
    # Replace smart quotes and other special characters
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    
    # Normalize titles
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    
    # Process numbers and currency
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    
    # Clean up spacing and format
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    
    return text.strip()

def test_triton_installation():
    """
    Tests if Triton is properly installed and working by comparing a simple addition operation between PyTorch's
    native implementation and a custom Triton kernel. Returns True or False.

    Example:

    from triton_test import test_triton_installation

    is_triton_working = test_triton_installation()

    if is_triton_working:
       print("Proceeding with Triton functionality...")
    else:
       print("Cannot proceed - Triton is not working properly")
"""
    try:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor):
            output = torch.empty_like(x)
            assert x.is_cuda and y.is_cuda and output.is_cuda
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
            return output

        print("Testing Triton installation...")
        a = torch.rand(3, device="cuda")
        b = a + a
        b_compiled = add(a, a)
        difference = b_compiled - b

        works = torch.all(difference == 0).item()
        if works:
            print("✓ Triton is working correctly")
        else:
           print("⨯ Triton test failed - results do not match PyTorch implementation.  Please visit the Windows Triton repository")
           print("here and review the instructions for installing any necessary dependencies: https://github.com/woct0rdho/triton-windows")
        return works

    except Exception as e:
        print(f"⨯ Triton test failed - error occurred: {str(e)}")
        return False

def supports_flash_attention():
    """Check if the current CUDA device supports flash attention (compute capability >= 8.0)."""
    if not torch.cuda.is_available():
        return False
    # Get the compute capability of the current CUDA device
    major, minor = torch.cuda.get_device_capability()
    return major >= 8

def check_cuda_re_triton():
    """
    Checks whether the files required by Triton 3.1.0 are present in the relative paths.
    This mirrors where the windows_utils.py script within the Triton library will look for them.
    """
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_runtime = nvidia_base_path / 'cuda_runtime'

    files_to_check = [
        cuda_runtime / "bin" / "cudart64_12.dll",
        cuda_runtime / "bin" / "ptxas.exe",
        cuda_runtime / "include" / "cuda.h",
        cuda_runtime / "lib" / "x64" / "cuda.lib"
    ]

    print("Checking CUDA files:")
    for file_path in files_to_check:
        exists = file_path.exists()
        status = "✓ Found" if exists else "✗ Missing"
        print(f"{status}: {file_path}")
    print()

def get_model_native_precision(embedding_model_name, vector_models):
    for group_models in vector_models.values():
        for model in group_models:
            if model['repo_id'] == embedding_model_name or model['name'] in embedding_model_name:
                return model['precision']
    return 'float32'

def get_appropriate_dtype(compute_device, use_half, model_native_precision):
    if compute_device.lower() == 'cpu':
        return torch.float32

    if model_native_precision == 'float16':
        return torch.float16
    elif model_native_precision == 'float32':
        if not use_half:
            return torch.float32
        else:
            if torch.cuda.is_available() and torch.version.cuda:
                cuda_capability = torch.cuda.get_device_capability()
                if cuda_capability[0] >= 8 and cuda_capability[1] >= 0:
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32
    else:
        return torch.float32

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
    Create citations with relevance scores and, for .pdf files, page numbers.
    """
    def group_metadata(metadata_list):
        grouped = {}
        for metadata in metadata_list:
            file_path = metadata['file_path']
            grouped.setdefault(file_path, {
                'name': Path(file_path).name,
                'scores': [],
                'pages': set(),
                'file_type': metadata.get('file_type', '')
            })
            grouped[file_path]['scores'].append(metadata['similarity_score'])
            if grouped[file_path]['file_type'] == '.pdf':
                page_number = metadata.get('page_number')
                if page_number is not None:
                    grouped[file_path]['pages'].add(page_number)
        return grouped

    def format_pages(pages):
        if not pages:
            return ''
        sorted_pages = sorted(pages)
        ranges = []
        start = prev = sorted_pages[0]
        for page in sorted_pages[1:]:
            if page == prev + 1:
                prev = page
            else:
                ranges.append((start, prev))
                start = prev = page
        ranges.append((start, prev))
        page_str = ', '.join(f"{s}-{e}" if s != e else f"{s}" for s, e in ranges)
        return f'<span style="color:#666;"> p.{page_str}</span>'

    def create_citation(data, file_path):
        min_score = min(data['scores'])
        max_score = max(data['scores'])
        score_range = f"{min_score:.4f}" if min_score == max_score else f"{min_score:.4f}-{max_score:.4f}"
        pages_html = format_pages(data['pages']) if data['file_type'] == '.pdf' else ''
        citation = (
            f'<a href="file:{file_path}" style="color:#DAA520;text-decoration:none;">{data["name"]}</a>'
            f'<span style="color:#808080;font-size:0.9em;"> ['
            f'<span style="color:#4CAF50;">{score_range}</span>]'
            f'{pages_html}'
            f'</span>'
        )
        return min_score, citation

    grouped_citations = group_metadata(metadata_list)
    citations_with_scores = [create_citation(data, file_path) for file_path, data in grouped_citations.items()]
    sorted_citations = [citation for _, citation in sorted(citations_with_scores)]
    list_items = "".join(f"<li>{citation}</li>" for citation in sorted_citations)

    return f"<ol>{list_items}</ol>"

def count_physical_cores():
    return psutil.cpu_count(logical=False)

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

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

def backup_database_incremental(new_database_name):
    source_directory = Path('Vector_DB')
    backup_directory = Path('Vector_DB_Backup')

    backup_directory.mkdir(parents=True, exist_ok=True)

    source_db_path = source_directory / new_database_name
    backup_db_path = backup_directory / new_database_name

    if backup_db_path.exists():
        try:
            shutil.rmtree(backup_db_path)
        except Exception as e:
            print(f"Warning: Could not remove existing backup of {new_database_name}: {e}")

    try:
        shutil.copytree(source_db_path, backup_db_path)
        # my_cprint(f"Creating backup of {new_database_name} database.", "green")
    except Exception as e:
        print(f"Error backing up {new_database_name}: {e}")

    # log of the latest backup info
    # with open(backup_directory / "backup_manifest.txt", "a") as manifest:
        # manifest.write(f"{new_database_name} backed up at {datetime.now()}\n")

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

# not currently used
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
    return capability >= (8, 0)

def get_precision():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a CUDA-enabled GPU.")

    capability = torch.cuda.get_device_capability()
    if capability >= (8, 0):
        precision = torch.bfloat16
    else:
        precision = torch.float16
    
    return precision

def get_device_and_precision():
    if torch.cuda.is_available():
        device = "cuda"
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            precision = "bfloat16"
        else:
            precision = "float16"
    else:
        device = "cpu"
        precision = "float32"
    return device, precision

class FlashAttentionUtils:
    """
    Flash Attention 2 is only supported on Ampere and newer GPUs
    https://github.com/Dao-AILab/flash-attention/blob/0dfb28174333d9eefb7c1dd4292690a8458d1e89/csrc/flash_attn/flash_api.cpp#L370
    """
    @staticmethod
    def check_package_availability():
        # check if flash_attn is installed
        return importlib.util.find_spec("flash_attn") is not None

    @staticmethod
    def check_version_compatibility():
        # check flash_attn version
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
        # check if dtype is compatible
        return dtype in [torch.float16, torch.bfloat16]

    @staticmethod
    def check_gpu_initialization():
        # check if CUDA is available and default device is CUDA
        return torch.cuda.is_available() and torch.cuda.current_device() >= 0

    @staticmethod
    def check_device_map(device_map):
        # check if device_map is compatible
        if device_map is None:
            return True
        if isinstance(device_map, dict):
            return "cpu" not in device_map.values() and "disk" not in device_map.values()
        return True

    @classmethod
    def is_flash_attention_compatible(cls, dtype=None, device_map=None):
        # run all checks
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

def prepare_long_path(base_path: str, filename: str) -> str:
    """
    Prepares a path for long filenames, especially for Windows systems.
    
    Args:
    base_path (str): The base directory path.
    filename (str): The original filename.
    
    Returns:
    str: Prepared full path, using extended-length path syntax if necessary.
    """
    base_path = os.path.normpath(base_path)
    full_path = os.path.join(base_path, filename)
    
    if os.name == 'nt' and len(full_path) > 255:
        full_path = "\\\\?\\" + os.path.abspath(full_path)
    
    return full_path