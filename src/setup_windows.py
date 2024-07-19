import sys
import subprocess
import os
import tkinter as tk
from tkinter import messagebox
import constants as c
from replace_sourcecode import replace_pdf_file, replace_instructor_file, replace_sentence_transformer_file

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tk.Tk()
    root.withdraw()
    if type == "info":
        messagebox.showinfo(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
    elif type == "yesno" and yes_no:
        response = messagebox.askyesno(title, message)
        root.destroy()
        return response
    root.destroy()

def check_python_version_and_confirm():
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if major == 3 and minor == 11:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\n Click YES to proceed or NO to exit.", type="yesno", yes_no=True)
    else:
        tkinter_message_box("Python Version Error", "This program requires Python 3.11.x.\n\n  The Pytorch library does not support Python 3.12 yet and I have chosen to no longer support Python 3.10.\n\n Exiting the installer...", type="error")
        return False

def get_platform_info():  # returns windows, linux or darwin
    os_name = platform.system().lower()
    return os_name

def is_nvidia_gpu_installed():
    try:
        output = subprocess.check_output(["nvidia-smi"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def display_gpu_message():
    if is_nvidia_gpu_installed():
        proceed_with_gpu = tkinter_message_box("GPU Check", "NVIDIA GPU detected. Click YES to install with gpu-acceleration or NO to exit the installer.", type="yesno", yes_no=True)
        return proceed_with_gpu
    else:
        proceed_without_gpu = tkinter_message_box("GPU Check", "No NVIDIA GPU detected. Click YES to install without GPU-acceleration or NO to exit the installer.", type="yesno", yes_no=True)
        return proceed_without_gpu

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

def install_pytorch(gpu_available):
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if gpu_available:
        os.system("pip install --no-cache-dir https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp311-cp311-win_amd64.whl https://download.pytorch.org/whl/cu121/torchaudio-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl https://github.com/jakaline-dev/Triton_win/releases/download/3.0.0/triton-3.0.0-cp311-cp311-win_amd64.whl")
        os.system("pip install --no-cache-dir https://github.com/bdashore3/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.2.2cxx11abiFALSE-cp311-cp311-win_amd64.whl") # need to make this conditional on cuda compute level
    else:
        os.system("pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl#sha256=88e63c916e3275fa30a220ee736423a95573b96072ded85e5c0171fd8f37a755 https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp311-cp311-win_amd64.whl#sha256=54ae4b89038065e7393c65bc8ff141d1bf3c2f70f88badc834247666608ba9f4 https://download.pytorch.org/whl/cpu/torchaudio-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl#sha256=6e718df4834f9cef28b7dc1edc9ceabfe477d4dbd5527b51234e96bf91465d9d")

def setup_windows_installation():
    if not check_python_version_and_confirm():
        return
    if not manual_installation_confirmation():
        return
    gpu_available = display_gpu_message()
    if not gpu_available:
        return
    # os.system("python -m pip install --no-cache-dir --upgrade pip")
    os.system("python -m pip install --no-cache-dir --upgrade pip setuptools wheel")
    
    # if is_nvidia_gpu_installed():
    os.system("pip install --no-cache-dir nvidia-cuda-runtime-cu12==12.2.140 nvidia-cublas-cu12==12.2.5.6 nvidia-cudnn-cu12==8.9.7.29")
    # nvidia-cuda-nvrtc-cu12==12.2.140 nvidia-cufft-cu12==11.0.8.103 nvidia-cusolver-cu12==11.5.2.141 nvidia-cusparse-cu12==12.1.2.141 nvidia-nvml-dev-cu12==12.2.140 nvidia-cuda-opencl-cu12==12.2.140 nvidia-cuda-nvcc-cu12==12.2.140
    
    os.system("pip install --no-cache-dir fsspec==2024.5.0") # accelerate requires pytorch, and pytorch will force this version if not specified
    os.system("pip install --no-cache-dir numpy==1.26.4") # langchain will force install this verison if not specified
    
    install_pytorch(gpu_available)
    
    os.system("pip install --no-cache-dir openai==1.23.6")
    os.system("pip install --no-cache-dir langchain==0.2.5")
    os.system("pip install --no-cache-dir langchain-community==0.2.5")
    
    os.system("pip install --no-cache-dir -r requirements.txt")
    
    if is_nvidia_gpu_installed():
        os.system("pip install --no-cache-dir xformers==0.0.25.post1")
        os.system("pip install --no-cache-dir nvidia-ml-py")

    os.system("pip install --no-cache-dir --no-deps -U git+https://github.com/shashikg/WhisperS2T.git")
    os.system("pip install --no-cache-dir --no-deps -U git+https://github.com/BBC-Esq/WhisperSpeech.git@add_cache_dir")
    # os.system("pip install --no-deps -U git+https://github.com/collabora/WhisperSpeech.git") # will force unwanted torch version if installed with dependencies
    os.system("pip install --no-cache-dir --no-deps chattts-fork==0.0.8")
    
    replace_pdf_file() # replaces pymupdf parser within langchain
    replace_instructor_file() # replaces instructor-embeddings
    replace_sentence_transformer_file() # replaces SentenceTransformer

setup_windows_installation()