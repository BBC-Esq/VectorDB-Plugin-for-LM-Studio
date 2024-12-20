import sys
import subprocess
import time
import os
import tkinter as tk
from tkinter import messagebox
from replace_sourcecode import replace_pdf_file, replace_instructor_file, replace_sentence_transformer_file

# ctranslate2==4.5.0 now requires cudnn 9+, which works with CUDA 12.3+; however, torch 2.3.1 only supports up to CUDA 12.1
# SUPPORTS Phi3.5 and Mistral Nemo...AWQ support was added in 4.4.0.  FA was added in 4.3.1 but removed in 4.4.0
# Therefore, torch 2.4.0+ is required to use cuDNN 9+ with ctranslate2 4.5.0+.
"""
******************************************
* PyTorch Wheel Names and Version Support
******************************************

# modern torch wheels contain either "cu121" or "cu124" in their name and there are prebuilt wheels for the following versions
+-----------------------+---------------------------------------------------------+
| Pytorch Wheel Moniker | PyTorch Versions Supported                              |
+-----------------------+---------------------------------------------------------+
| cu124                 | 2.5.1, 2.5.0, 2.4.1, 2.4.0                              |
| cu121                 | 2.5.1, 2.5.0, 2.4.1, 2.4.0, 2.3.1, 2.3.0, 2.2.2...2.1.0 |
+--------------+------------------------------------------------------------------+
* Per the next table, "cu124" and "cu121" refer to compatibility with CUDA release 12.4.1 and 12.1.1 specifically.

*************************************************
* PyTorch Dependencies and Version Requirements
*************************************************

# dependencies scraped from pypi
# technically, only for linux builds...but useful info for running Windows and pip-installing CUDA dependencies
+-------+------------------------+-------------------------+-------------------+------------------+--------+------------+--------+
| Torch | nvidia-cuda-nvrtc-cu12 | nvidia-cuda-runtime-cu12| nvidia-cublas-cu12| nvidia-cudnn-cu12| triton | mkl        | sympy  |
+-------+------------------------+-------------------------+-------------------+------------------+--------+------------+--------+
| 2.5.1 | 12.4.127               | 12.4.127                | 12.4.5.8          | 9.1.0.70         | 3.1.0  | -          | 1.13.1 |
| 2.5.0 | 12.4.127               | 12.4.127                | 12.4.5.8          | 9.1.0.70         | 3.1.0  | -          | 1.13.1 |
| 2.4.1 | 12.1.105               | 12.1.105                | 12.1.3.1          | 9.1.0.70         | 3.0.0  | -          | -      |
| 2.4.0 | 12.1.105               | 12.1.105                | 12.1.3.1          | 9.1.0.70         | 3.0.0  | -          | -      |
| 2.3.1 | 12.1.105               | 12.1.105                | 12.1.3.1          | 8.9.2.26         | 2.3.1  | <=2021.4.0 | -      |
| 2.3.0 | 12.1.105               | 12.1.105                | 12.1.3.1          | 8.9.2.26         | 2.3.0  | <=2021.4.0 | -      |
| 2.2.2 | 12.1.105               | 12.1.105                | 12.1.3.1          | 8.9.2.26         | 2.2.0  | -          | -      |
+-------+------------------------+-------------------------+-------------------+------------------+--------+------------+--------+
* 12.1.105 and 12.1.3.1 stem from CUDA release 12.1.1 specifically
* 12.4.127 and 12.4.5.8 stem from CUDA release 12.4.1 specifically
* In other words, torch is not 100% compatible with CUDA 12.1.0 or 12.4.0, for example, or any other version.

********************************************
* cuDNN and CUDA Compatibility Matrix
********************************************

# cuDNN is different...
# First, according to Nvidia, cuDNN 8.9.2.26 is only compatible up to CUDA 12.2
# Second, for cuDNN 9+ Nvidia promises compatibility for all 12.x releases, but the cuDNN + CUDA combination still controls static linking:
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
* PyTorch Official Release Compatibility
*********************************************

# Pytorch summarizes their compatibility as follows:
# https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
+------------------+----------------------------+------------------------------------------------+---------------------------+
| PyTorch Version | Python                     | Stable                                          | Experimental              |
+-----------------+----------------------------+-------------------------------------------------+---------------------------+
| 2.5             | >=3.9, <=3.12, (3.13 exp.) | CUDA 11.8, CUDA 12.1, CUDA 12.4, CUDNN 9.1.0.70 | None                      |
+-----------------+----------------------------+-------------------------------------------------+---------------------------+
| 2.4             | >=3.8, <=3.12              | CUDA 11.8, CUDA 12.1, CUDNN 9.1.0.70            | CUDA 12.4, CUDNN 9.1.0.70 |
+-----------------+----------------------------+-------------------------------------------------+---------------------------+
| 2.3             | >=3.8, <=3.11, (3.12 exp.) | CUDA 11.8, CUDNN 8.7.0.84                       | CUDA 12.1, CUDNN 8.9.2.26 |
+-----------------+----------------------------+-------------------------------------------------+---------------------------+
| 2.2             | >=3.8, <=3.11, (3.12 exp.) | CUDA 11.8, CUDNN 8.7.0.84                       | CUDA 12.1, CUDNN 8.9.2.26 |
+-----------------+----------------------------+-------------------------------------------------+---------------------------+

*********************************
* Xformers Version Compatibility
*********************************

# xformers is strictly tied to a specific version of torch
+------------------+---------------+
| Xformers Version | Torch Version |
+------------------+---------------+
| v0.0.28.post3    | 2.5.1         |
| v0.0.28.post2    | 2.5.0         |
| v0.0.28.post1    | 2.4.1         |
| v0.0.27.post2    | 2.4.0         |
| v0.0.27.post1    | 2.4.0         |
| v0.0.27          | 2.3.0         | # release notes confusingly say "some operation might require torch 2.4"
| v0.0.26.post1    | 2.3.0         |
| v0.0.25.post1    | 2.2.2         |
+------------------+---------------+

# old repo for windows triton: https://github.com/jakaline-dev/Triton_win/releases
# new repo for windows triton: https://github.com/woct0rdho/triton-windows/releases
# "Triton 3.1.0 works with torch >= 2.4.0, not 2.3.x."
# "The wheels are built against CUDA 12.5, and they should work with other CUDA 12.x."

*****************************************
* Flash Attention 2 (FA2) Compatibility
*****************************************

# this table represents the version of Torch and CUDA that FA2 is compatible with
# based on flash-attention/.github/workflows/publish.yml
+---------------+-----------------------------------------------+----------------+
| FA2 Version   | Torch Versions Supported                      | CUDA Versions  |
+---------------+-----------------------------------------------+----------------+
| v2.7.1.post4  | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241001 | 11.8.0, 12.3.2 |
| v2.7.1.post3  | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241001 | 11.8.0, 12.3.2 |
| v2.7.1.post2  | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241001 | 11.8.0, 12.3.2 |
| v2.7.1.post1* | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241010 | 11.8.0, 12.4.1 |
| v2.7.1        | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0.dev20241010 | 11.8.0, 12.4.1 |
| v2.7.0.post2* | 2.2.2, 2.3.1, 2.4.0, 2.5.1                    | 11.8.0, 12.4.1 |
| v2.7.0.post1  | 2.2.2, 2.3.1, 2.4.0, 2.5.1                    | 11.8.0, 12.4.1 |
| v2.7.0        | 2.2.2, 2.3.1, 2.4.0, 2.5.1                    | 11.8.0, 12.3.2 |
| v2.6.3*       | 2.2.2, 2.3.1, 2.4.0                           | 11.8.0, 12.3.2 |
| v2.6.2        | 2.2.2, 2.3.1, 2.4.0.dev20240527               | 11.8.0, 12.3.2 |
| v2.6.1*       | 2.2.2, 2.3.1, 2.4.0.dev20240514               | 11.8.0, 12.3.2 |
| v2.6.0.post1  | 2.2.2, 2.3.1, 2.4.0.dev20240514               | 11.8.0, 12.2.2 |
| v2.6.0        | 2.2.2, 2.3.1, 2.4.0.dev20240512               | 11.8.0, 12.2.2 |
| v2.5.9.post1* | 2.2.2, 2.3.0, 2.4.0.dev20240407               | 11.8.0, 12.2.2 |
+---------------+-----------------------------------------------+----------------+
* has a corresponding Windows build
* https://github.com/bdashore3/flash-attention/releases/

Adding support for Python 3.12 will be a huge effort.
Suspect libraries sare langchain, pydantic, aiohttp, and possibly more.
"""

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
                process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=240)
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
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=efbcfdd4399197d06b32f7c0e1711c615188cdd65427b933648c7478fb880b3f"],
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=10ad542aab6b47dbe73c441381986d50a7ed5021cbe01d593a14477ec1f067a0"],
            ["uv", "pip", "install", "https://download.pytorch.org/whl/cu121/torchaudio-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=c7dee68cd3d2b889bab71d4a0c345bdc3ea2fe79a62b921a6b49292c605b6071"]
        ]
        # pip_args_list = [
            # ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp311-cp311-win_amd64.whl#sha256=b1d40a13a6fd3f92aa5728ab84756571381b6b1ccae7ce62037c28d539687c25"],
            # ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torchaudio-2.4.0%2Bcu124-cp311-cp311-win_amd64.whl#sha256=12f7f2b1c0fb435875175247083c8aca056face6bc5388b9a494a90ca197632c"],
            # ["uv", "pip", "install", "https://download.pytorch.org/whl/cu124/torchvision-0.19.0%2Bcu124-cp311-cp311-win_amd64.whl#sha256=42ac55c0fd1cdea14c20168f0b24eba7fc2d2eb4ef75196ded7b76f81dee619f"]
        # ]
    elif "@" in library or "git+" in library:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    else:
        pip_args_list = [["uv", "pip", "install", library, "--no-deps"]]
    
    for pip_args in pip_args_list:
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {pip_args[3]}")
                subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=240)
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
    "flash_attn @ https://github.com/bdashore3/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.2.2cxx11abiFALSE-cp311-cp311-win_amd64.whl",
    # "https://github.com/bdashore3/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
    "torch==2.2.2",
    # "torch==2.5.1",
    "triton @ https://github.com/jakaline-dev/Triton_win/releases/download/3.0.0/triton-3.0.0-cp311-cp311-win_amd64.whl#sha256=2c78f5f85cf88d46eb9664c23691052d6c153a6043656fc15c50a0d13bc5565c", # required by alibaba vector models
    # "https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp311-cp311-win_amd64.whl",
    "whisper_s2t @ git+https://github.com/shashikg/WhisperS2T.git@e7f7e6dbfdc7f3a39454feb9dd262fd3653add8c",
    "WhisperSpeech @ git+https://github.com/BBC-Esq/WhisperSpeech.git@41c9accb7d9ac1e4e5f5c110a4a973c566c56fd8",
    # "nvidia-pyindex"
]

other_libraries = [
    "accelerate==1.2.0",
    "aiofiles==24.1.0",
    "aiohappyeyeballs==2.4.4",
    "aiohttp==3.11.10",
    "aiosignal==1.3.1",
    "anndata==0.11.1",
    "annotated-types==0.7.0",
    "antlr4-python3-runtime==4.9.3", # omegaconf 2.3.0 requires 4.9.*
    "anyio==4.7.0",
    "argcomplete==3.5.2", # required by minicpm3 chat, datamodel-code-generator 0.26.2 requires >=1.10,<4.0
    "array_api_compat==1.9.1",
    "attrs==24.2.0",
    "av==14.0.1",
    "backoff==2.2.1",
    "beautifulsoup4==4.12.3",
    "bitsandbytes==0.45.0",
    "black==24.10.0", # required by minicpm3 chat, datamodel-code-generator requires >=19.10b0
    "braceexpand==0.1.7",
    "certifi==2024.8.30",
    "cffi==1.17.1",
    "chardet==5.2.0",
    "charset-normalizer==3.4.0",
    "chattts-fork==0.0.8",
    "click==8.1.7",
    "cloudpickle==2.2.1", # tiledb-cloud 0.12.31 requires >=1.4.1,<3
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "ctranslate2==4.5.0",
    "cycler==0.12.1",
    "dataclasses-json==0.6.7",
    "datasets==3.1.0",
    "datamodel-code-generator==0.26.2", # required by minicpm3 chat
    "deepdiff==8.0.1",
    "dill==0.3.8", # datasets 3.1.0 requires >=0.3.0,<0.3.9
    "distro==1.9.0",
    "dnspython==2.7.0", # required by minicpm3 chat
    "docx2txt==0.8",
    "einops==0.8.0",
    "einx==0.3.0",
    "email-validator==2.2.0", # required by minicpm3 chat
    "emoji==2.14.0",
    "encodec==0.1.1",
    "et-xmlfile==1.1.0", # openpyxl requires; caution...openpyxl 3.1.5 (6/28/2024) predates et-xmlfile 2.0.0 (10/25/2024)
    "fastcore==1.7.24",
    "fastprogress==1.0.3", # only required by whisperspeech
    "filetype==1.2.0",
    "filelock==3.16.1",
    "frozendict==2.4.6",
    "frozenlist==1.5.0",
    "fsspec==2024.9.0", # datasets 3.1.0 requires >=2023.1.0,<=2024.9.0
    "genson==1.3.0", # required by minicpm3 chat; datamodel-code-generator requires >=1.2.1,<2.0
    "greenlet==3.1.1",
    "gTTS==2.5.3",
    "h11==0.14.0",
    "h5py==3.12.1",
    "httpcore==1.0.7",
    "httpx==0.28.1",
    "httpx-sse==0.4.0",
    "huggingface-hub==0.26.5", # tokenizers 0.20.3 requires >=0.16.4,<1.0
    "humanfriendly==10.0",
    "HyperPyYAML==1.2.2",
    "idna==3.10",
    "importlib_metadata==8.5.0",
    "inflect==5.6.2", # required by minicpm3 chat; datamodel-code-generator 0.26.2 requires >=4.1.0,<6.0
    "isort==5.13.2", # required by minicpm3 chat; datamodel-code-generator requires >=4.3.21,<6.0
    "InstructorEmbedding==1.0.1",
    "Jinja2==3.1.4", # datamodel-code-generator 0.26.2 requires >=2.10.1,<4.0
    "jiter==0.8.0", # required by openai newer versions
    "joblib==1.4.2",
    "jsonpatch==1.33",
    "jsonpath-python==1.0.6",
    "jsonpointer==3.0.0",
    "jsonschema==4.23.0", # required by minicpm3 chat
    "jsonschema-specifications==2024.10.1", # required by minicpm3 chat; jsonschema 4.23.0 requires >=2023.03.6
    "kiwisolver==1.4.7",
    "langchain-community==0.2.17",
    "langchain-core==0.2.43",
    "langchain-huggingface==0.0.3",
    "langchain-text-splitters==0.2.4",
    "langsmith==0.1.125",
    "langdetect==1.0.9",
    "langchain==0.2.17",
    # these versions work if/when I upgrade langchain, which requires revising my code due to a change to pydantic v2 in codebase
    # "langchain==0.3.7",
    # "langchain-community==0.3.5",
    # "langchain-core==0.3.15",
    # "langchain-huggingface==0.1.2",
    # "langchain-text-splitters==0.3.2",
    # "langdetect==1.0.9",
    # "langsmith==0.1.125",
    "llvmlite==0.43.0", # only required by numba
    "lxml==5.3.0",
    "Markdown==3.7",
    "markdown-it-py==3.0.0",
    "MarkupSafe==3.0.2",
    "marshmallow==3.23.1", # dataclasses-json 0.6.7 requires less than 4.0
    "matplotlib==3.9.3", # uniquely requires pyparsing==3.1.2 cycler==0.12.1 kiwisolver==1.4.5
    "mdurl==0.1.2",
    "more-itertools==10.5.0",
    "mpmath==1.3.0", # sympy 1.12.1 requires less than 1.4
    "msg-parser==1.2.0",
    "multidict==6.1.0",
    "multiprocess==0.70.16", # datasets 3.1.0 requires <0.70.17
    "mypy-extensions==1.0.0",
    "natsort==8.4.0",
    "nest-asyncio==1.6.0",
    "networkx==3.4.2",
    "nltk==3.8.1", # not higher; gives unexplained error
    "numba==0.60.0", # only required by openai-whisper
    "numpy==1.26.4",
    # TORCH 2.2.2 VERSIONS
    "nvidia-cublas-cu12==12.1.3.1",
    "nvidia-cuda-runtime-cu12==12.1.105",
    "nvidia-cuda-nvrtc-cu12==12.1.105",
    "nvidia-cufft-cu12==11.0.2.54",
    # "nvidia-cuda-cupti-cu12==12.1.105",
    # "nvidia-curand-cu12==10.3.2.106",
    # "nvidia-cusolver-cu12==11.4.5.107",
    # "nvidia-cusparse-cu12==12.1.0.106",
    # "nvidia-nvtx-cu12==12.1.105",
    # cudnn versions are flexible-ish; experiment with caution
    # "nvidia-cudnn-cu12==8.9.7.29",
    "nvidia-cudnn-cu12==9.1.0.70",
    # TORCH 2.4.0 VERSIONS
    # "nvidia-cuda-runtime-cu12==12.4.99",
    # "nvidia-cublas-cu12==12.4.2.65",
    # "nvidia-cuda-nvrtc-cu12==12.4.99",
    # "nvidia-cudnn-cu12==9.1.0.70",
    "nvidia-ml-py==12.560.30",
    "olefile==0.47",
    "omegaconf==2.3.0",
    "openai==1.57.0", # only required by chat_lm_studio.py script
    "openai-whisper==20231117", # only required by whisper_s2t if using openai vanilla backend
    "openpyxl==3.1.5",
    "optimum==1.23.3",
    "ordered-set==4.1.0",
    "orderly-set==5.2.2", # deepdiff 8.0.1 requires 5.2.2
    "orjson==3.10.12",
    "packaging==24.2",
    "pandas==2.2.3",
    "pathspec==0.12.1", # required by minicpm3 chat
    "peft==0.14.0", # only required by mississippi model
    "pillow==11.0.0",
    "platformdirs==4.3.6",
    "propcache==0.2.1",
    "protobuf==5.29.1",
    "psutil==6.1.0",
    "pyarrow==18.1.0",
    "pyarrow-hotfix==0.6",
    "pycparser==2.22",
    "pydantic==2.9.2", # datamodel-code-generator requires >=1.10.0,<3.0,!=2.4.0
    "pydantic_core==2.23.4", # pydantic 2.9.2 requires 2.23.4
    # "pydantic-settings==2.6.1", # not sure if required...
    "Pygments==2.18.0",
    "pypandoc==1.14",
    "pyparsing==3.2.0",
    "pypdf==5.1.0",
    "pyreadline3==3.5.4",
    "python-dateutil==2.9.0.post0",
    "python-docx==1.1.2",
    # "python-dotenv==1.0.1", # only required by pydantic-settings
    "python-iso639==2024.10.22",
    "python-magic==0.4.27",
    "pytz==2024.2",
    "PyYAML==6.0.2",
    "rapidfuzz==3.10.1",
    "referencing==0.35.1", # required by minicpm3 chat
    "regex==2024.9.11", # 2024.11.6 is the newest version but pypi gives an error for some reason
    "requests==2.32.3",
    "requests-toolbelt==1.0.0",
    "rich==13.9.4",
    "rpds-py==0.22.3", # required by minicpm3 chat
    "ruamel.yaml==0.18.6",
    "ruamel.yaml.clib==0.2.12",
    "safetensors==0.4.5",
    "scikit-learn==1.5.2",
    "scipy==1.14.1",
    "sentence-transformers==3.0.1",
    "sentencepiece==0.2.0",
    "six==1.17.0",
    "sniffio==1.3.1",
    "sounddevice==0.5.1",
    "soundfile==0.12.1",
    "soupsieve==2.6",
    "speechbrain==0.5.16",
    "SQLAlchemy==2.0.36", # langchain and langchain-community pre-0.3 require less than 3.0.0
    "sseclient-py==1.8.0",
    "sympy==1.12.1", # anything above is not compatible with llava-next-vicuna vision models
    "tabulate==0.9.0",
    "tblib==1.7.0", # tiledb-cloud requires >=1.7.0, <2.0.0
    "tenacity==8.5.0", # langchain requires >=8.1.0,<9.0.0,!=8.4.0
    "termcolor==2.5.0",
    "threadpoolctl==3.5.0",
    "tiktoken==0.8.0",
    "tiledb==0.32.5",
    "tiledb-cloud==0.12.31",
    "tiledb-vector-search==0.10.3",
    "timm==1.0.12",
    "tokenizers==0.21.0",
    "tqdm==4.67.1",
    "transformers==4.47.0",
    "typing-inspect==0.9.0",
    "typing_extensions==4.12.2",
    "unstructured-client==0.24.1",
    "tzdata==2024.2",
    "urllib3==2.2.3",
    "vector-quantize-pytorch==1.20.11",
    "vocos==0.1.0",
    "watchdog==6.0.0",
    # "webdataset==0.2.86",
    "webdataset==0.2.100",
    "wrapt==1.17.0",
    "xformers==0.0.25.post1", # highly-specific to torch version
    # "xformers==0.0.28.post1", # requires torch 2.4.1
    "xlrd==2.0.1",
    "xxhash==3.5.0",
    # "yarl==1.12.0", # langchain-related libraries and optimum require less than 2.0
    "yarl==1.18.3", # langchain-related libraries and optimum require less than 2.0
    "zipp==3.21.0",
]

full_install_libraries = [
    "PySide6==6.8.1",
    "pymupdf==1.25.0",
    "unstructured==0.13.4"
]

def pip_install_with_deps(library, max_retries=5, delay=3):
    pip_args = ["uv", "pip", "install", library]

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library} with dependencies")
            subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=300)
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

# 9. update config.yaml to include jeeves database
def update_config_yaml():
    import yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    vector_model_path = os.path.join(script_dir, 'Models', 'vector', 'thenlper--gte-base')
    
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