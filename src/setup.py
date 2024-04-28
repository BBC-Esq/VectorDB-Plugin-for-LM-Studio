import sys
import subprocess
import os
import tkinter as tk
from tkinter import messagebox
import constants as c
from replace_sourcecode import replace_pdf_file, replace_instructor_file

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

def install_pytorch(cuda_version_num, cuda_installed): # also installs Triton for if cuda available
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if cuda_installed and cuda_version_num is not None and cuda_version_num >= 12.1:
        if minor == 11:
            os.system("pip3 install https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp311-cp311-win_amd64.whl https://download.pytorch.org/whl/cu121/torchaudio-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl https://github.com/jakaline-dev/Triton_win/releases/download/3.0.0/triton-3.0.0-cp311-cp311-win_amd64.whl")
        elif minor == 10:
            os.system("pip3 install https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp310-cp310-win_amd64.whl https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp310-cp310-win_amd64.whl https://download.pytorch.org/whl/cu121/torchaudio-2.2.2%2Bcu121-cp310-cp310-win_amd64.whl https://github.com/jakaline-dev/Triton_win/releases/download/3.0.0/triton-3.0.0-cp310-cp310-win_amd64.whl")
    else:
        if minor == 11:
            os.system("pip3 install https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl#sha256=88e63c916e3275fa30a220ee736423a95573b96072ded85e5c0171fd8f37a755 https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp311-cp311-win_amd64.whl#sha256=54ae4b89038065e7393c65bc8ff141d1bf3c2f70f88badc834247666608ba9f4 https://download.pytorch.org/whl/cpu/torchaudio-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl#sha256=6e718df4834f9cef28b7dc1edc9ceabfe477d4dbd5527b51234e96bf91465d9d")
        elif minor == 10:
            os.system("pip3 install https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp310-cp310-win_amd64.whl#sha256=fc29dda2795dd7220d769c5926b1c50ddac9b4827897e30a10467063691cdf54 https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp310-cp310-win_amd64.whl#sha256=acad6f9573b9d6b50a5a3942d0145cb0f9100608acb53a09bfc11ed5720dcfe3 https://download.pytorch.org/whl/cpu/torchaudio-2.2.2%2Bcpu-cp310-cp310-win_amd64.whl#sha256=012cd8efbd9e0011abcd79daff98d312136b5e49417062bef1d38cd208f0c05f")

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
    #os.system("pip install git+https://github.com/SilasMarvin/instructor-embedding.git@silas-update-for-newer-sentence-transformers")
    os.system("pip3 install --no-deps -U git+https://github.com/shashikg/WhisperS2T.git")
    os.system("pip3 install git+https://github.com/collabora/WhisperSpeech.git")
    
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if proceed and cuda_version_num >= 12.1 and (major == 3 and minor in [10, 11]):
        os.system("pip install xformers==0.0.25.post1")
        os.system("pip3 install nvidia-ml-py==12.535.133")

    replace_pdf_file()
    replace_instructor_file()

setup_windows_installation()
