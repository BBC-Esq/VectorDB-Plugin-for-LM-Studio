import sys
import subprocess
import time
import os
import tkinter as tk
from tkinter import messagebox
from replace_sourcecode import (
    replace_pdf_file,
    replace_instructor_file,
    replace_sentence_transformer_file,
    replace_chattts_file,
    add_cuda_files,
)



start_time = time.time()

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tk.Tk()
    root.withdraw()
    if yes_no:
        result = messagebox.askyesno(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
        result = False
    else:
        messagebox.showinfo(title, message)
        result = True
    root.destroy()
    return result

def check_python_version_and_confirm():
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if major == 3 and minor == 12:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.", type="yesno", yes_no=True)
    else:
        tkinter_message_box("Python Version Error", "This program requires Python 3.11 or 3.12\n\nPython versions prior to 3.11 are not supported.  Python 3.11 is supported but you must use the installer named setup_windows_cp311 instead of this one.\n\nExiting the installer...", type="error")
        return False

def is_nvidia_gpu_installed():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def manual_installation_confirmation():
    if not tkinter_message_box("Confirmation", "Have you installed Git?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Git Large File Storage?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Pandoc?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Microsoft Build Tools and/or Visual Studio with the necessary libraries to compile code?\n\nClick YES to confirm or NO to cancel installation.", type="yesno", yes_no=True):
        return False
    return True

if not check_python_version_and_confirm():
    sys.exit(1)

nvidia_gpu_detected = is_nvidia_gpu_installed()
if nvidia_gpu_detected:
    message = "An NVIDIA GPU has been detected.\n\nDo you want to proceed with the installation?"
else:
    message = "No NVIDIA GPU has been detected. An NVIDIA GPU is required for this script to function properly.\n\nDo you still want to proceed with the installation?"

if not tkinter_message_box("GPU Detection", message, type="yesno", yes_no=True):
    sys.exit(1)

if not manual_installation_confirmation():
    sys.exit(1)

def upgrade_pip_setuptools_wheel(max_retries=5, delay=3):
    upgrade_commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "wheel", "--no-cache-dir"]
    ]
    
    for command in upgrade_commands:
        package = command[5]
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Upgrading {package}...")
                process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully upgraded {package}\033[0m")
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to upgrade {package} after {max_retries} attempts.")
            except Exception as e:
                print(f"An unexpected error occurred while upgrading {package}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to upgrade {package} after {max_retries} attempts due to unexpected errors.")

def pip_install_with_retry(library, max_retries=5, delay=3):
    if library.startswith("torch=="):
        pip_args_list = [
            # python 3.11
            # ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-win_amd64.whl#sha256=6c8a7003ef1327479ede284b6e5ab3527d3900c2b2d401af15bcc50f2245a59f"],
            # ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torchaudio-2.5.1%2Bcu124-cp311-cp311-win_amd64.whl#sha256=b3d75f4e6efc5412fe78c7f2787ee4f39cea1317652e1a47785879cde109f5c4"],
            # ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp311-cp311-win_amd64.whl#sha256=15796b453a99ed0f0cbc249d129685ddc88157310135fb3addaf738a15db5306"]
            # python 3.12
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl#sha256=3c3f705fb125edbd77f9579fa11a138c56af8968a10fc95834cdd9fdf4f1f1a6"],
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torchaudio-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl#sha256=cca2de94f232611b20d379edf28befa7a1aa482ae9ed41c3b958b08ed1bf4983"],
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp312-cp312-win_amd64.whl#sha256=0f6c7b3b0e13663fb3359e64f3604c0ab74c2b4809ae6949ace5635a5240f0e5"]
        ]
    elif "@" in library or "git+" in library:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    else:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    
    for pip_args in pip_args_list:
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {pip_args[3]}")
                subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully installed {pip_args[3]}\033[0m")
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to install {pip_args[3]} after {max_retries} attempts.")
                    return 0
    return 1

def install_libraries(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install_with_retry(library)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

# Libraries to install first
priority_libraries = [
    # "flash_attn @ https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl",
    flash_attn @ "https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl",
    "torch==2.5.1",
    # "triton @ https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp311-cp311-win_amd64.whl",
    "triton @ https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp312-cp312-win_amd64.whl",
    "whisper_s2t @ git+https://github.com/shashikg/WhisperS2T.git@e7f7e6dbfdc7f3a39454feb9dd262fd3653add8c",
    "WhisperSpeech @ git+https://github.com/BBC-Esq/WhisperSpeech.git@41c9accb7d9ac1e4e5f5c110a4a973c566c56fd8",
]

other_libraries = [
    "accelerate==1.2.1",
    "aiofiles==24.1.0",
    "aiohappyeyeballs==2.4.4",
    "aiohttp==3.11.11", # langchain libraries require <4
    "aiosignal==1.3.2", # only required by aiohttp
    "anndata==0.11.1",
    "annotated-types==0.7.0",
    "anyio==4.7.0",
    "array_api_compat==1.10.0", # only anndata requires
    "async-timeout==5.0.1",
    "attrs==24.3.0",
    "av==14.0.1",
    "backoff==2.2.1",
    "beautifulsoup4==4.12.3",
    "bitsandbytes==0.45.0",
    "braceexpand==0.1.7",
    "certifi==2024.12.14",
    "cffi==1.17.1",
    "chardet==5.2.0",
    "charset-normalizer==3.4.1", # requests requires <4
    "chattts==0.2.1",
    "click==8.1.8",
    "cloudpickle==3.1.0", # only required by tiledb-cloud
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "contourpy==1.3.1", # onlyk required by matplotlib
    "ctranslate2==4.5.0",
    "cycler==0.12.1",
    "dataclasses-json==0.6.7",
    "datasets==3.2.0",
    "deepdiff==8.1.1", # required by unstructured
    "dill==0.3.8", # datasets 3.2.0 requires <0.3.9; multiprocess 0.70.16 requires >=0.3.8
    "distro==1.9.0",
    "docx2txt==0.8",
    "einops==0.8.0",
    "einx==0.3.0",
    "emoji==2.14.0",
    "encodec==0.1.1",
    "et-xmlfile==1.1.0", # openpyxl requires; caution...openpyxl 3.1.5 (6/28/2024) predates et-xmlfile 2.0.0 (10/25/2024)
    "fastcore==1.7.28", # only required by whisperspeech
    "fastprogress==1.0.3", # only required by whisperspeech
    "filetype==1.2.0",
    "filelock==3.16.1",
    "fonttools==4.55.3" # only required by matplotlib
    "frozendict==2.4.6",
    "frozenlist==1.5.0",
    "fsspec==2024.9.0", # datasets 3.2.0 requires <=2024.9.0
    "greenlet==3.1.1",
    "gTTS==2.5.4",
    "h11==0.14.0",
    "h5py==3.12.1",
    "httpcore==1.0.7",
    "httpx==0.28.1",
    "httpx-sse==0.4.0",
    "huggingface-hub==0.27.0", # tokenizers 0.20.3 requires >=0.16.4,<1.0
    "humanfriendly==10.0",
    "HyperPyYAML==1.2.2",
    "idna==3.10",
    "importlib_metadata==8.5.0",
    "InstructorEmbedding==1.0.1",
    "Jinja2==3.1.5",
    "jiter==0.8.2", # required by openai newer versions
    "joblib==1.4.2",
    "jsonpatch==1.33",
    "jsonpath-python==1.0.6",
    "jsonpointer==3.0.0",
    "kiwisolver==1.4.8",
    "langchain==0.3.13",
    "langchain-community==0.3.13",
    "langchain-core==0.3.28",
    "langchain-huggingface==0.1.2",
    "langchain-text-splitters==0.3.4",
    "langdetect==1.0.9",
    "langsmith==0.2.7",
    "llvmlite==0.43.0", # only required by numba
    "lxml==5.3.0",
    "Markdown==3.7",
    "markdown-it-py==3.0.0",
    "MarkupSafe==3.0.2",
    "marshmallow==3.23.2",
    "matplotlib==3.10.0", # uniquely requires pyparsing==3.1.2 cycler==0.12.1 kiwisolver==1.4.5
    "mdurl==0.1.2",
    "more-itertools==10.5.0",
    "mpmath==1.3.0", # sympy 1.12.1 requires less than 1.4
    "msg-parser==1.2.0",
    "multidict==6.1.0",
    "multiprocess==0.70.16", # datasets 3.2.0 requires <0.70.17
    "mypy-extensions==1.0.0",
    "natsort==8.4.0",
    "nest-asyncio==1.6.0",
    "networkx==3.4.2",
    "nltk==3.8.1", # not higher; gives unexplained error
    "numba==0.60.0", # only required by openai-whisper
    "numpy==1.26.4", # langchain libraries <2; numba <2.1; scipy <2.3; chattts <2.0.0
    "nvidia-cuda-runtime-cu12==12.4.127", # Torch 2.5.1 official support (based on CUDA 12.4.1)
    "nvidia-cublas-cu12==12.4.5.8", # Torch 2.5.1 official support (based on CUDA 12.4.1)
    "nvidia-cuda-nvrtc-cu12==12.4.127", # Torch 2.5.1 official support (based on CUDA 12.4.1)
    "nvidia-cuda-nvcc-cu12==12.4.131", # Torch 2.5.1 official support (based on CUDA 12.4.1) 
    "nvidia-cufft-cu12==11.2.1.3", # Torch 2.5.1 official support (based on CUDA 12.4.1)
    "nvidia-cudnn-cu12==9.1.0.70", # Torch 2.5.1 officially supported version
    "nvidia-ml-py==12.560.30",
    "olefile==0.47",
    "openai==1.58.1", # only required by chat_lm_studio.py script and whispers2t (if using openai vanilla backend)
    "openai-whisper==20240930", # only required by whisper_s2t (if using openai vanilla backend)
    "openpyxl==3.1.5",
    "optimum==1.23.3",
    "ordered-set==4.1.0",
    "orderly-set==5.2.3", # deepdiff 8.1.1 requires 5.2.3
    "orjson==3.10.13",
    "packaging==24.2",
    "pandas==2.2.3",
    "peft==0.14.0", # only required by mississippi model
    "pillow==11.0.0",
    "platformdirs==4.3.6",
    "propcache==0.2.1",
    "protobuf==5.29.2",
    "psutil==6.1.1",
    "pyarrow==18.1.0",
    "pyarrow-hotfix==0.6",
    "pybase16384==0.3.7", # only required by chattts
    "pycparser==2.22",
    "pydantic==2.10.4",
    "pydantic_core==2.27.2",
    "pydantic-settings==2.7.1",
    "Pygments==2.18.0",
    "pypandoc==1.14",
    "pyparsing==3.2.0",
    "pypdf==5.1.0",
    "pyreadline3==3.5.4",
    "python-dateutil==2.9.0.post0",
    "python-docx==1.1.2",
    "python-dotenv==1.0.1",
    "python-iso639==2024.10.22",
    "python-magic==0.4.27",
    "pytz==2024.2",
    "PyYAML==6.0.2",
    "rapidfuzz==3.11.0",
    "regex==2024.11.6",
    "requests==2.32.3",
    "requests-toolbelt==1.0.0",
    "rich==13.9.4",
    "ruamel.yaml==0.18.7",
    "ruamel.yaml.clib==0.2.12",
    "safetensors==0.4.5",
    "scikit-learn==1.6.0",
    "scipy==1.14.1",
    "sentence-transformers==3.0.1",
    "sentencepiece==0.2.0",
    "six==1.17.0",
    "sniffio==1.3.1",
    "sounddevice==0.5.1",
    "soundfile==0.12.1",
    "soupsieve==2.6",
    "speechbrain==0.5.16",
    "SQLAlchemy==2.0.36", # langchain and langchain-community require less than 3.0.0
    "sseclient-py==1.8.0",
    "sympy==1.13.1", # torch 2.5.1 requires sympy==1.13.1
    "tabulate==0.9.0",
    "tblib==1.7.0", # tiledb-cloud requires >= 1.7.0 but < 1.8.0
    "tenacity==9.0.0",
    "termcolor==2.5.0",
    "threadpoolctl==3.5.0",
    "tiktoken==0.8.0",
    "tiledb==0.33.0",
    "tiledb-cloud==0.13.0",
    "tiledb-vector-search==0.11.0",
    "timm==1.0.12",
    "tokenizers==0.21.0",
    "tqdm==4.67.1",
    "transformers==4.47.1",
    "typing-inspect==0.9.0",
    "typing_extensions==4.12.2",
    "unstructured-client==0.24.1",
    "tzdata==2024.2",
    "urllib3==2.2.3", # requests 2.32.3 requires <3
    "vector-quantize-pytorch==1.20.11",
    "vocos==0.1.0",
    "watchdog==6.0.0",
    "webdataset==0.2.100", # required by all TTS libraries
    "wrapt==1.17.0",
    # "https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp311-cp311-win_amd64.whl", # torch 2.5.1 specific
    "https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp312-cp312-win_amd64.whl", # torch 2.5.1 specific
    "xlrd==2.0.1",
    "xxhash==3.5.0",
    "yarl==1.18.3", # aiohttp requires <2
    "zipp==3.21.0",
]

full_install_libraries = [
    "PySide6==6.8.1",
    "pymupdf==1.25.1",
    "unstructured==0.13.4"
]

def pip_install_with_deps(library, max_retries=5, delay=3):
    pip_args = ["uv", "pip", "install", library]

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library} with dependencies")
            subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=600)
            print(f"\033[92mSuccessfully installed {library} with dependencies\033[0m")
            return attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to install {library} after {max_retries} attempts.")
                return 0

def install_libraries_with_deps(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install_with_deps(library)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

# 1. upgrade pip, setuptools, wheel
print("Upgrading pip, setuptools, and wheel:")
upgrade_pip_setuptools_wheel()

# 2. install uv
print("Installing uv:")
subprocess.run(["pip", "install", "uv"], check=True)

# 3. install priority_libraries
print("\nInstalling priority libraries:")
priority_failed, priority_multiple = install_libraries(priority_libraries)

# 4. install install other_libraries
print("\nInstalling other libraries:")
other_failed, other_multiple = install_libraries(other_libraries)

# 5. install full_install_libraries
print("\nInstalling libraries with dependencies:")
full_install_failed, full_install_multiple = install_libraries_with_deps(full_install_libraries)

print("\n----- Installation Summary -----")

all_failed = priority_failed + other_failed + full_install_failed
all_multiple = priority_multiple + other_multiple + full_install_multiple

if all_failed:
    print("\033[91m\nThe following libraries failed to install:\033[0m")
    for lib in all_failed:
        print(f"\033[91m- {lib}\033[0m")

if all_multiple:
    print("\033[93m\nThe following libraries required multiple attempts to install:\033[0m")
    for lib, attempts in all_multiple:
        print(f"\033[93m- {lib} (took {attempts} attempts)\033[0m")

if not all_failed and not all_multiple:
    print("\033[92mAll libraries installed successfully on the first attempt.\033[0m")
elif not all_failed:
    print("\033[92mAll libraries were eventually installed successfully.\033[0m")

if all_failed:
    sys.exit(1)

# 6. replace sourcode files
replace_pdf_file()
replace_instructor_file()
replace_sentence_transformer_file()
replace_chattts_file()
add_cuda_files()

# 7. Create directores if needed
def create_directory_structure():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "Models")
    subdirs = ["chat", "tts", "vector", "vision", "whisper"]
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created Models directory: {models_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(models_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Ensured subdirectory exists: {subdir_path}")


create_directory_structure()

# 8. download kobold
def download_kobold():
    import platform
    import requests
    import os

    file_name = "koboldcpp_nocuda.exe"
    url = f"https://github.com/LostRuins/koboldcpp/releases/download/v1.76/{file_name}"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, "Assets")
    
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    kobold_path = os.path.join(assets_dir, file_name)
    
    try:
        print(f"Downloading KoboldCPP from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(kobold_path, 'wb') as file:
            file.write(response.content)
        
        print(f"\033[92mKoboldCPP nocuda version downloaded successfully to {kobold_path}.\033[0m")
                
    except requests.exceptions.HTTPError as http_err:
        print(f"\033[91mHTTP error occurred while downloading KoboldCPP: {http_err}\033[0m")
    except Exception as e:
        print(f"\033[91mFailed to download KoboldCPP nocuda version: {e}\033[0m")

download_kobold()

# 9. manually add jeeves database to config.yaml
def update_config_yaml():
    import yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    vector_model_path = os.path.join(script_dir, 'Models', 'vector', 'ibm-granite--granite-embedding-30m-english')
    
    if 'created_databases' not in config:
        config['created_databases'] = {}
    if 'user_manual' not in config['created_databases']:
        config['created_databases']['user_manual'] = {}
    
    config['created_databases']['user_manual']['chunk_overlap'] = 349
    config['created_databases']['user_manual']['chunk_size'] = 700
    config['created_databases']['user_manual']['model'] = vector_model_path
    
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)

update_config_yaml()

end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")

"""
******************************************
* Torch Wheel Names and Version Support
******************************************

# modern torch wheels contain either "cu121" or "cu124" in their name and are prebuilt for the following versions
+---------------+---------------------------------------------------------+
| Pytorch Wheel | PyTorch Versions Supported                              |
+---------------+---------------------------------------------------------+
| cu124         | 2.5.1, 2.5.0, 2.4.1, 2.4.0                              |
| cu121         | 2.5.1, 2.5.0, 2.4.1, 2.4.0, 2.3.1, 2.3.0, 2.2.2...2.1.0 |
+---------------+---------------------------------------------------------+
* "cu121" and "cu124" wheels include libraries from CUDA releases 12.1.1 and 12.4.1, respectively (see next table)

******************************************
* Torch requirements
******************************************

# dependencies scraped from pypi for linux builds
# useful to determine windows compatibility when pip installing CUDA libraries rather than relying on a systemwide installation
+-------+-----------------+------------------+------------+-----------+--------+------------+--------+
| Torch | cuda-nvrtc-cu12 | cuda-runtime-cu12| cublas-cu12| cudnn-cu12| triton | mkl        | sympy  |
+-------+-----------------+------------------+------------+-----------+--------+------------+--------+
| 2.5.1 | 12.4.127        | 12.4.127         | 12.4.5.8   | 9.1.0.70  | 3.1.0  | -          | 1.13.1 |
| 2.5.0 | 12.4.127        | 12.4.127         | 12.4.5.8   | 9.1.0.70  | 3.1.0  | -          | 1.13.1 |
| 2.4.1 | 12.1.105        | 12.1.105         | 12.1.3.1   | 9.1.0.70  | 3.0.0  | -          | -      |
| 2.4.0 | 12.1.105        | 12.1.105         | 12.1.3.1   | 9.1.0.70  | 3.0.0  | -          | -      |
| 2.3.1 | 12.1.105        | 12.1.105         | 12.1.3.1   | 8.9.2.26  | 2.3.1  | <=2021.4.0 | -      |
| 2.3.0 | 12.1.105        | 12.1.105         | 12.1.3.1   | 8.9.2.26  | 2.3.0  | <=2021.4.0 | -      |
| 2.2.2 | 12.1.105        | 12.1.105         | 12.1.3.1   | 8.9.2.26  | 2.2.0  | -          | -      |
+-------+-----------------+------------------+------------+-----------+--------+------------+--------+
* 12.1.105 and 12.1.3.1 come from CUDA release 12.1.1
* 12.4.127 and 12.4.5.8 come from CUDA release 12.4.1
* In other words, torch is not 100% compatible with CUDA 12.1.0 or 12.4.0, for example, or any other version.

****************************************
* cuDNN and CUDA Compatibility
****************************************

# According to Nvidia, cuDNN 8.9.2.26 is only compatible up to CUDA 12.2
# For cuDNN 9+, Nvidia promises compatibility for all CUDA 12.x releases, but static linking fluctuates
+---------------+-----------------+-------------------------+
| cuDNN Version | Static Linking  | No Static Linking       |
+---------------+-----------------+-------------------------+
| 8.9.2         | 12.1, 11.8      | 12.0, ≤11.7             |
| 8.9.3         | 12.1, 11.8      | 12.0, ≤11.7             |
| 8.9.4         | 12.2, 11.8      | 12.1, 12.0, ≤11.7       |
| 8.9.5         | 12.2, 11.8      | 12.1, 12.0, ≤11.7       |
| 8.9.6         | 12.2, 11.8      | 12.1, 12.0, ≤11.7       |
| 8.9.7         | 12.2, 11.8      | 12.1, 12.0, ≤11.7       |
| 9.0.0         | 12.3, 11.8      | 12.2, 12.1, 12.0, ≤11.7 |
| 9.1.0         | 12.4-12.0, 11.8 | ≤11.7                   |
| 9.1.1         | 12.5-12.0, 11.8 | ≤11.7                   |
+---------------+-----------------+-------------------------+
* 9.2+ continues the same trend

*********************************************
* PyTorch Official Compatibility Matrix
*********************************************

# https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
+-------+----------------------------+----------------------------------------+----------------------------+
| Torch | Python                     | Stable                                 | Experimental               |
+-------+----------------------------+----------------------------------------+----------------------------+
| 2.6   | >=3.9, <=3.12, (3.13 exp.) | CUDA 11.8, 12.1, 12.4 + CUDNN 9.1.0.70 | None                       |
|       |                            | CUDA 12.6 + CUDNN 9.5.1.17             | None                       |
+-------+----------------------------+----------------------------------------+----------------------------+
| 2.5   | >=3.9, <=3.12, (3.13 exp.) | CUDA 11.8, 12.1, 12.4 + CUDNN 9.1.0.70 | None                       |
+-------+----------------------------+----------------------------------------+----------------------------+
| 2.4   | >=3.8, <=3.12              | CUDA 11.8, 12.1 + CUDNN 9.1.0.70       | CUDA 12.4 + CUDNN 9.1.0.70 |
+-------+----------------------------+----------------------------------------+----------------------------+
| 2.3   | >=3.8, <=3.11, (3.12 exp.) | CUDA 11.8 + CUDNN 8.7.0.84             | CUDA 12.1 + CUDNN 8.9.2.26 |
+-------+----------------------------+----------------------------------------+----------------------------+
| 2.2   | >=3.8, <=3.11, (3.12 exp.) | CUDA 11.8 + CUDNN 8.7.0.84             | CUDA 12.1 + CUDNN 8.9.2.26 |
+-------+----------------------------+----------------------------------------+----------------------------+

*********************************
* Xformers
*********************************

# strictly tied to a specific torch version per "metadata" in each whl file
+------------------+---------------+
| Xformers Version | Torch Version |
+------------------+---------------+
| v0.0.29.post1    | 2.5.1         |
| v0.0.29          | 2.5.1         |
| v0.0.28.post3    | 2.5.1         |
| v0.0.28.post2    | 2.5.0         |
| v0.0.28.post1    | 2.4.1         |
| v0.0.27.post2    | 2.4.0         |
| v0.0.27.post1    | 2.4.0         |
| v0.0.27          | 2.3.0         | # release notes confusingly say "some operation might require torch 2.4"
| v0.0.26.post1    | 2.3.0         |
| v0.0.25.post1    | 2.2.2         |
+------------------+---------------+
* Pypi only has windows wheels through 2.4.0.
* 2.4.0+ windows wheels only available from pytorch directly - https://download.pytorch.org/whl/cu124/xformers/
  * e.g. pip install https://download.pytorch.org/whl/cu124/xformers-0.0.28.post3-cp311-cp311-win_amd64.whl


*********************************
* Triton
*********************************

# 3.0.0 and earlier wheels are located here: https://github.com/jakaline-dev/Triton_win/releases
  * E.g., https://github.com/jakaline-dev/Triton_win/releases/download/3.0.0/triton-3.0.0-cp311-cp311-win_amd64.whl
  * only supports up to Python 3.11

# 3.1.0 and later wheels: https://github.com/woct0rdho/triton-windows/releases
 * "Triton 3.1.0 requires torch 2.4.0+
 * "The wheels are built against CUDA 12.5, and they should work with other CUDA 12.x."
 * https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp311-cp311-win_amd64.whl
 * supports Python 3.12


*****************************************
* LINUX Flash Attention 2 Compatibility
*****************************************

# According to flash-attention/.github/workflows/publish.yml
+--------------+-----------------------------------------------+----------------+
| FA2 Version  | Torch Versions Supported                      | CUDA Versions  |
+--------------+-----------------------------------------------+----------------+
| v2.7.1.post4 | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241001 | 11.8.0, 12.3.2 |
| v2.7.1.post3 | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241001 | 11.8.0, 12.3.2 |
| v2.7.1.post2 | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241001 | 11.8.0, 12.3.2 |
| v2.7.1.post1 | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241010 | 11.8.0, 12.4.1 |
| v2.7.1       | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241010 | 11.8.0, 12.4.1 |
| v2.7.0.post2 | 2.2.2, 2.3.1, 2.4.0, 2.5.1                    | 11.8.0, 12.4.1 |
| v2.7.0.post1 | 2.2.2, 2.3.1, 2.4.0, 2.5.1                    | 11.8.0, 12.4.1 |
| v2.7.0       | 2.2.2, 2.3.1, 2.4.0, 2.5.1                    | 11.8.0, 12.3.2 |
| v2.6.3*      | 2.2.2, 2.3.1, 2.4.0                           | 11.8.0, 12.3.2 |
| v2.6.2       | 2.2.2, 2.3.1, 2.4.0.dev20240527               | 11.8.0, 12.3.2 |
| v2.6.1       | 2.2.2, 2.3.1, 2.4.0.dev20240514               | 11.8.0, 12.3.2 |
| v2.6.0.post1 | 2.2.2, 2.3.1, 2.4.0.dev20240514               | 11.8.0, 12.2.2 |
| v2.6.0       | 2.2.2, 2.3.1, 2.4.0.dev20240512               | 11.8.0, 12.2.2 |
| v2.5.9.post1 | 2.2.2, 2.3.0, 2.4.0.dev20240407               | 11.8.0, 12.2.2 |
+--------------+-----------------------------------------------+----------------+
* 2.5.8 is the first to support torch 2.2.2
* no prebuilt wheels simultaneously support torch 2.2.2 and CUDA prior to 12.2.2

*****************************************
* WINDOWS Flash Attention 2 Compatibility
*****************************************

# per https://github.com/bdashore3/flash-attention/releases/
+--------------+---------------------+----------------+
| FA2 Version  | Torch Versions      | CUDA Versions |
+--------------+---------------------+---------------+
| v2.7.1.post1 | 2.3.1, 2.4.0, 2.5.1 | 12.4.1        |
| v2.7.0.post2 | 2.3.1, 2.4.0, 2.5.1 | 12.4.1        |
| v2.6.3       | 2.2.2, 2.3.1, 2.4.0 | 12.3.2        |
| v2.6.1       | 2.2.2, 2.3.1        | 12.3.2        |
| v2.5.9.post2 | 2.2.2, 2.3.1        | 12.2.2        |
| v2.5.9.post1 | 2.2.2, 2.3.0        | 12.2.2        |
| v2.5.8       | 2.2.2, 2.3.0        | 12.2.2        |
| v2.5.6       | 2.1.2, 2.2.2        | 12.2.2        |
| v2.5.2       | 2.1.2, 2.2.0        | 12.2.2        |
| v2.4.2       | 2.1.2, 2.2.0        | 12.2.2        |
+--------------+---------------------+---------------+


*********************************
* Xformers Compatibility with FA 2
*********************************
https://github.com/facebookresearch/xformers/blob/46a02df62a6192bf11456e712ae072bfd9c83e71/xformers/ops/fmha/flash.py#L66
https://github.com/facebookresearch/xformers/commit/839c4ec4b928f1f02f83d25a7d111bde819e6bce


***************************************************************
* CUDA, Torch, cuDNN, Triton, FA2, Xformers, one big mess
***************************************************************

PER CTRANSLATE2...

Starting from version 4.5.0, ctranslate2 is compatible with cuDNN 9+.

---

According to Ashraf...

either use ct2<4.5 along with torch<2.4 or ct2==4.5 along with torch>=2.4

v4.5.0 works just fine with pip installed cudnn, but if you have a torch version where the cuda binaries are
precompiled such as torch==2.5.0+cu121 or any version that ends with +cu12, this error comes up, the only
solution is downgrade to v4.4.0 at the moment which is strange because it was compiled using cudnn 8.9

+---------------+---------------------+
| Torch Version | Ctranslate2 Version |
+---------------+---------------------+
| 2.*.*+cu121   | <=4.4.0             |
| 2.*.*+cu124   | >=4.5.0             |
| >=2.4.0       | >=4.5.0             |
| <2.4.0        | <4.5.0              |
+---------------+---------------------+
* torch(CUDA or CPU) are compatible with CT2 except for torch +cu121, which requires CT2 <=4.4.0

Update: it's compatible with torch==2.*+cu124 so it's only incompatible with 12.1, I'll open a PR to solve this
  * but his fix didn't work: https://github.com/OpenNMT/CTranslate2/pull/1807


**************************************
* CTRANSLATE2 Compatibility
**************************************

Ctranslate2 3.24.0 - last to use cuDNN 8.1.1 with CUDA 11.2.2 by default
Ctranslate2 4.0.0 - first to use cuDNN 8.8.0 with CUDA 12.2 by default
Ctranslate2 4.5.0 - first to use cuDNN 9.1 with CUDA 12.2 by default

# based on /blob/master/python/tools/prepare_build_environment_windows.sh


*************************************************
* Python 3.12
*************************************************

Python 3.12.4 is incompatible with pydantic.v1 as of pydantic==2.7.3
https://github.com/langchain-ai/langchain/issues/22692
***Everything should now be fine as long as Langchain 0.3+ is used, which requires pydantic version 2+***

Other libraries can be checked at: https://pyreadiness.org/3.12/
"""