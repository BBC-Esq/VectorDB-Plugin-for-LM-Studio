import sys
import subprocess
import os
import tkinter as tk
from tkinter import messagebox
import constants as c
from replace_pdf import replace_pdf_file

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
    if major < 3 or (major == 3 and minor < 10):
        tkinter_message_box("Python Version Error", "This program is currently only compatible with Python 3.10 or 3.11.", type="error")
        return False
    elif major >= 3 and minor >= 12:
        tkinter_message_box("Python Version Error", "Python 3.12+ detected. PyTorch is not currently compatible with Python 3.12 - exiting installer.", type="error")
        return False
    else:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} detected. Click OK to proceed with the installation, or Cancel to stop.", type="yesno", yes_no=True)

def check_cuda_version():
    try:
        cuda_version_output = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        if "release" in cuda_version_output:
            cuda_version = cuda_version_output.split("release ")[1].split(",")[0]
            major, minor = cuda_version.split('.')[:2]
            cuda_version_num = float(f"{major}.{minor}")
            return cuda_version_num
        else:
            return None
    except FileNotFoundError:
        return None

def display_cuda_message():
    cuda_version_num = check_cuda_version()
    if cuda_version_num is None:
        proceed_without_cuda = tkinter_message_box("CUDA Check", "No CUDA installation detected. Would you like to proceed with a CPU-only installation?", type="yesno", yes_no=True)
        return None, proceed_without_cuda
    elif cuda_version_num >= 12.1:
        proceed_with_cuda = tkinter_message_box("CUDA Check", f"CUDA version {cuda_version_num} detected. Would you like to proceed with the GPU-accelerated installation?", type="yesno", yes_no=True)
        return cuda_version_num, proceed_with_cuda
    else:
        update_cuda = tkinter_message_box("CUDA Check", f"Incorrect version of CUDA installed (Version: {cuda_version}). Would you like to proceed with a CPU-only installation?", type="yesno", yes_no=True)
        if update_cuda:
            return None, True
        else:
            print("Exiting installer.")
            sys.exit(0)

def manual_installation_confirmation():
    if not tkinter_message_box("Confirmation", c.MESSAGE_GIT, type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", c.MESSAGE_GIT_LFS, type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", c.MESSAGE_PANDOC, type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", c.MESSAGE_MS_BUILD_TOOLS, type="yesno", yes_no=True):
        return False
    return True

def install_pytorch(cuda_version_num, cuda_installed):
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if cuda_installed and cuda_version_num >= 12.1:
        if minor == 11:
            os.system("pip3 install https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp311-cp311-win_amd64.whl#sha256=d79324159c622243429ec214a86b8613c1d7d46fc4821374d324800f1df6ade1 https://download.pytorch.org/whl/cu121/torchvision-0.17.0%2Bcu121-cp311-cp311-win_amd64.whl#sha256=307e52c2887c1d2b50cc3581cf5f4c169130b8352462e361e71eeda19e0dd263 https://download.pytorch.org/whl/cu121/torchaudio-2.2.0%2Bcu121-cp311-cp311-win_amd64.whl#sha256=67a33d2066668a2754d9dd5d000419f60102dd17eff9803f9b0a5e1d9261f79d")
        elif minor == 10:
            os.system("pip3 install https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp310-cp310-win_amd64.whl#sha256=8f54c647ee19c8b4c0aad158c73b83b2c06cb62351e9cfa981540ce7295a9015 https://download.pytorch.org/whl/cu121/torchaudio-2.2.0%2Bcu121-cp310-cp310-win_amd64.whl#sha256=f5181424ea9c01d4199d37ebc394040c96ca36836c6e3a9fb9d6af58d30d8ed0 https://download.pytorch.org/whl/cu121/torchvision-0.17.0%2Bcu121-cp310-cp310-win_amd64.whl#sha256=b5ba1adc6f9f1a40af9608ebc447ceed6c8816dcb926d59675c81111b8676966")
    else:
        if minor == 11:
            os.system("pip3 install https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=58194066e594cd8aff27ddb746399d040900cc0e8a331d67ea98499777fa4d31 https://download.pytorch.org/whl/cpu/torchaudio-2.2.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=c775e5d3e176161f33eaf4aba2708b39eb474925779ad8f3cf1df6ad10ed5213 https://download.pytorch.org/whl/cpu/torchvision-0.17.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=eb1e9d061c528c8bb40436d445599ca05fa997701ac395db3aaec5cb7660b6ee")
        elif minor == 10:
            os.system("pip3 install https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp310-cp310-win_amd64.whl#sha256=15a657038eea92ac5db6ab97b30bd4b5345741b49553b2a7e552e80001297124 https://download.pytorch.org/whl/cpu/torchaudio-2.2.0%2Bcpu-cp310-cp310-win_amd64.whl#sha256=da25e6bc800aa8436b4d3220ff5c44df5b4250ec86bd20345da36b041a19bfb6 https://download.pytorch.org/whl/cpu/torchvision-0.17.0%2Bcpu-cp310-cp310-win_amd64.whl#sha256=569ebc5f47bb765ae73cd380ace01ddcb074c67df05d7f15f5ddd0fa3062881a")

def setup_windows_installation():
    if not check_python_version_and_confirm():
        return
    if not manual_installation_confirmation():
        return
    cuda_version_num, proceed = display_cuda_message()
    if not proceed:
        return
    os.system("python -m pip install --upgrade pip")
    install_pytorch(cuda_version_num, proceed)
    os.system("pip3 install -r requirements.txt")
    os.system("pip3 install --no-deps whisper-s2t==1.3.0")
    os.system(c.BITSANDBYTES_INSTALL_COMMAND)
    
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if proceed and cuda_version_num >= 12.1 and (major == 3 and minor in [10, 11]):
        os.system("pip3 install xformers==0.0.24")
        os.system("pip3 install nvidia-ml-py==12.535.133")

    replace_pdf_file()

setup_windows_installation()
