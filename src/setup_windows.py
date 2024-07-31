import sys
import subprocess
import time
import os
import tkinter as tk
from tkinter import messagebox
from replace_sourcecode import replace_pdf_file, replace_instructor_file, replace_sentence_transformer_file

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
    if major == 3 and minor == 11:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.", type="yesno", yes_no=True)
    else:
        tkinter_message_box("Python Version Error", "This program requires Python 3.11.x.\n\nThe Pytorch library does not support Python 3.12 yet and I have chosen to no longer support Python 3.10.\n\nExiting the installer...", type="error")
        return False

def is_nvidia_gpu_installed():
    try:
        output = subprocess.check_output(["nvidia-smi"])
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

start_time = time.time()

subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.2.32"], check=True)
print("\033[92mInstalled uv package manager.\033[0m")

def upgrade_pip_setuptools_wheel(max_retries=5, delay=3):
    upgrade_commands = [
        [sys.executable, "-m", "uv", "pip", "install", "--upgrade", "pip", "--no-cache-dir"],
        [sys.executable, "-m", "uv", "pip", "install", "--upgrade", "setuptools", "--no-cache-dir"],
        [sys.executable, "-m", "uv", "pip", "install", "--upgrade", "wheel", "--no-cache-dir"]
    ]
    
    for command in upgrade_commands:
        package = command[5]
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Upgrading {package}...")
                process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
                print(f"Successfully upgraded {package}")
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
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=efbcfdd4399197d06b32f7c0e1711c615188cdd65427b933648c7478fb880b3f"],
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=10ad542aab6b47dbe73c441381986d50a7ed5021cbe01d593a14477ec1f067a0"],
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu121/torchaudio-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=c7dee68cd3d2b889bab71d4a0c345bdc3ea2fe79a62b921a6b49292c605b6071"]
        ]
    elif "@" in library or "git+" in library:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    else:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    
    for pip_args in pip_args_list:
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {pip_args[3]}")
                print(f"Running command: {' '.join(pip_args)}")
                result = subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=180)
                print(f"Successfully installed {pip_args[3]}")
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
    "flash_attn @ https://github.com/bdashore3/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.2.2cxx11abiFALSE-cp311-cp311-win_amd64.whl",
    "torch==2.2.2",
    "triton @ https://github.com/jakaline-dev/Triton_win/releases/download/3.0.0/triton-3.0.0-cp311-cp311-win_amd64.whl#sha256=2c78f5f85cf88d46eb9664c23691052d6c153a6043656fc15c50a0d13bc5565c",
    "whisper_s2t @ git+https://github.com/shashikg/WhisperS2T.git@e7f7e6dbfdc7f3a39454feb9dd262fd3653add8c",
    "WhisperSpeech @ git+https://github.com/BBC-Esq/WhisperSpeech.git@41c9accb7d9ac1e4e5f5c110a4a973c566c56fd8"
]

other_libraries = [
    "accelerate==0.33.0",
    "aiohttp==3.9.5",
    "aiosignal==1.3.1",
    "anndata==0.10.8",
    "annotated-types==0.7.0",
    "antlr4-python3-runtime==4.9.3",
    "anyio==4.4.0",
    "array_api_compat==1.7.1",
    "attrs==23.2.0",
    "av==12.3.0",
    "backoff==2.2.1",
    "beautifulsoup4==4.12.3",
    "bitsandbytes==0.43.1",
    "braceexpand==0.1.7",
    "certifi==2024.7.4",
    "cffi==1.16.0",
    "chardet==5.2.0",
    "charset-normalizer==3.3.2",
    "chattts-fork==0.0.8",
    "click==8.1.7",
    "cloudpickle==2.2.1",
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "ctranslate2==4.3.1",
    "dataclasses-json==0.6.7",
    "datasets==2.20.0",
    "deepdiff==7.0.1",
    "dill==0.3.8",
    "distro==1.9.0",
    "docx2txt==0.8",
    "einops==0.8.0",
    "einx==0.3.0",
    "emoji==2.12.1",
    "encodec==0.1.1",
    "et-xmlfile==1.1.0",
    "fastcore==1.5.54",
    "fastprogress==1.0.3",
    "filetype==1.2.0",
    "filelock==3.15.4",
    "frozendict==2.4.4",
    "frozenlist==1.4.1",
    "fsspec==2024.5.0",
    "greenlet==3.0.3",
    "gTTS==2.5.1",
    "h11==0.14.0",
    "h5py==3.11.0",
    "httpcore==1.0.5",
    "httpx==0.27.0",
    "huggingface-hub==0.24.1",
    "humanfriendly==10.0",
    "HyperPyYAML==1.2.2",
    "idna==3.7",
    "importlib_metadata==8.1.0",
    "InstructorEmbedding==1.0.1",
    "Jinja2==3.1.4",
    "joblib==1.4.2",
    "jsonpatch==1.33",
    "jsonpath-python==1.0.6",
    "jsonpointer==3.0.0",
    "langchain==0.2.5",
    "langchain-community==0.2.5",
    "langchain-core==0.2.23",
    "langchain-huggingface==0.0.3",
    "langchain-text-splitters==0.2.2",
    "langdetect==1.0.9",
    "langsmith==0.1.93",
    "llvmlite==0.43.0",
    "lxml==5.2.2",
    "Markdown==3.6",
    "markdown-it-py==3.0.0",
    "MarkupSafe==2.1.5",
    "marshmallow==3.21.3",
    "mdurl==0.1.2",
    "more-itertools==10.3.0",
    "mpmath==1.3.0",
    "msg-parser==1.2.0",
    "multidict==6.0.5",
    "multiprocess==0.70.16",
    "mypy-extensions==1.0.0",
    "natsort==8.4.0",
    "nest-asyncio==1.6.0",
    "networkx==3.3",
    "nltk==3.8.1",
    "numba==0.60.0",
    "numpy==1.26.4",
    "nvidia-cublas-cu12==12.1.3.1",
    "nvidia-cuda-runtime-cu12==12.1.105",
    "nvidia-cuda-nvrtc-cu12==12.1.105",
    "nvidia-cudnn-cu12==8.9.7.29",
    "nvidia-ml-py==12.555.43",
    "olefile==0.47",
    "omegaconf==2.3.0",
    "openai==1.23.6",
    "openai-whisper==20231117",
    "openpyxl==3.1.2",
    "optimum==1.21.2",
    "ordered-set==4.1.0",
    "orjson==3.10.6",
    "packaging==24.1",
    "pandas==2.2.2",
    "pillow==10.4.0",
    "platformdirs==4.2.2",
    "protobuf==5.27.2",
    "psutil==6.0.0",
    "pyarrow==17.0.0",
    "pyarrow-hotfix==0.6",
    "pycparser==2.22",
    "pydantic==2.7.4",
    "pydantic_core==2.18.4",
    "Pygments==2.18.0",
    "pypandoc==1.13",
    "pypdf==4.3.1",
    "pyreadline3==3.4.1",
    "python-dateutil==2.9.0.post0",
    "python-docx==1.1.0",
    "python-iso639==2024.4.27",
    "python-magic==0.4.27",
    "pytz==2024.1",
    "PyYAML==6.0.1",
    "rapidfuzz==3.9.4",
    "regex==2024.5.15",
    "requests==2.32.3",
    "requests-toolbelt==1.0.0",
    "rich==13.7.1",
    "ruamel.yaml==0.18.6",
    "ruamel.yaml.clib==0.2.8",
    "safetensors==0.4.3",
    "scikit-learn==1.5.1",
    "scipy==1.14.0",
    "sentence-transformers==3.0.1",
    "sentencepiece==0.2.0",
    "shiboken6==6.6.3.1",
    "six==1.16.0",
    "sniffio==1.3.1",
    "sounddevice==0.4.6",
    "soundfile==0.12.1",
    "soupsieve==2.5",
    "speechbrain==0.5.16",
    "SQLAlchemy==2.0.31",
    "sympy==1.12.1", # anything above is not compatible with llava-next-vicuna vision models
    "tabulate==0.9.0",
    "tblib==1.7.0",
    "tenacity==8.5.0",
    "termcolor==2.4.0",
    "threadpoolctl==3.5.0",
    "tiktoken==0.7.0",
    "tiledb==0.27.1",
    "tiledb-cloud==0.12.18",
    "tiledb-vector-search==0.2.2",
    "timm==0.9.16",
    "tokenizers==0.19.1",
    "tqdm==4.66.4",
    "transformers==4.43.1",
    "typing-inspect==0.9.0",
    "typing_extensions==4.12.2",
    "unstructured==0.13.4",
    "unstructured-client==0.24.1",
    "tzdata==2024.1",
    "urllib3==2.2.2",
    "vector-quantize-pytorch==1.15.3",
    "vocos==0.1.0",
    "webdataset==0.2.86",
    "wrapt==1.16.0",
    "xformers==0.0.25.post1",
    "xlrd==2.0.1",
    "xxhash==3.4.1",
    "yarl==1.9.4",
    "zipp==3.19.2"
]

full_install_libraries = [
    "pyside6==6.7.2",
    "pymupdf==1.24.9",
    "unstructured==0.13.4"
]

def pip_install_with_deps(library, max_retries=5, delay=3):
    pip_args = ["uv", "pip", "install", library]

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library} with dependencies")
            print(f"Running command: {' '.join(pip_args)}")
            result = subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=300)
            print(f"Successfully installed {library} with dependencies")
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

print("\033[92mUpgrading pip, setuptools, and wheel:\033[0m")
upgrade_pip_setuptools_wheel()

print("\033[92m\nInstalling priority libraries:\033[0m")
priority_failed, priority_multiple = install_libraries(priority_libraries)

print("\033[92m\nInstalling other libraries:\033[0m")
other_failed, other_multiple = install_libraries(other_libraries)

print("\033[92m\nInstalling libraries with dependencies:\033[0m")
full_install_failed, full_install_multiple = install_libraries_with_deps(full_install_libraries)

print("\n--- Installation Summary ---")

all_failed = priority_failed + other_failed + full_install_failed
all_multiple = priority_multiple + other_multiple + full_install_multiple

if all_failed:
    print(f"\033[91m\nThe following libraries failed to install:\033[0m")
    for lib in all_failed:
        print(f"\033[91m- {lib}\033[0m")

if all_multiple:
    print(f"\033[93m\nThe following libraries required multiple attempts to install:\033[0m")
    for lib, attempts in all_multiple:
        print(f"\033[93m- {lib} (took {attempts} attempts)\033[0m")

if not all_failed and not all_multiple:
    print(f"\033[93m\nAll libraries installed successfully on the first attempt.\033[0m")
elif not all_failed:
    print(f"\033[93m\nAll libraries were eventually installed successfully.\033[0m")

if all_failed:
    sys.exit(1)
    
replace_pdf_file()
replace_instructor_file()
replace_sentence_transformer_file()

end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")
