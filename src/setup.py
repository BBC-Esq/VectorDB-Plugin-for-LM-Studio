import sys
import subprocess
import os
import shutil
import hashlib
import tkinter as tk
from tkinter import messagebox

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
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
            return cuda_version
        else:
            return None
    except FileNotFoundError:
        return None

def display_cuda_message():
    cuda_version = check_cuda_version()
    if cuda_version is None:
        proceed_without_cuda = tkinter_message_box("CUDA Check", "No CUDA installation detected. Would you like to proceed with a CPU-only installation?", type="yesno", yes_no=True)
        return None, proceed_without_cuda
    elif cuda_version == "11.8":
        return "11.8", tkinter_message_box("CUDA Check", "You have the correct CUDA version (11.8). Click OK to proceed with the installation.", type="yesno", yes_no=True)
    elif cuda_version == "12.1":
        return "12.1", tkinter_message_box("CUDA Check", "CUDA version 12.1 detected. CUDA 11.8 is required. Click OK to proceed with CPU-only installation of PyTorch or Cancel to exit installer.", type="yesno", yes_no=True)
    else:
        update_cuda = tkinter_message_box("CUDA Check", f"Incorrect version of CUDA installed (Version: {cuda_version}). Would you like to proceed with a CPU-only installation?", type="yesno", yes_no=True)
        return None, update_cuda

def manual_installation_confirmation():
    if not tkinter_message_box("Confirmation", "Have you installed Git? Click OK to confirm, or Cancel to exit installation.  Git can be dowloaded here: https://git-scm.com/", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Git Large File Storage? Click OK to confirm, or Cancel to exit installation.  Git Large File Storage can be downloaded here: https://git-lfs.com/", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Pandoc? Click OK to confirm, or Cancel to exit installation.  Pandoc can be downloaded here: https://pandoc.org/", type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Microsoft Build Tools and Visual Studio? Click OK to confirm, or Cancel to exit installation.  Microsoft Build Tools can be downloaded here: https://visualstudio.microsoft.com/visual-cpp-build-tools/", type="yesno", yes_no=True):
        return False
    return True

def install_pytorch(cuda_version, cuda_installed):
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if cuda_installed:
        if cuda_version == "11.8":
            if major == 3 and minor == 11:
                os.system("pip install https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp311-cp311-win_amd64.whl#sha256=623af3c2b94c58951b71e247f39b1b7377cc94d13162a548c59ed9cf81b2b0b2")
                os.system("pip install https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp311-cp311-win_amd64.whl#sha256=036391a65f3c2ac6dbe4b73ea0acc303dd1c0a667e2a3592a194b2d2db377da1")
                os.system("pip install https://download.pytorch.org/whl/cu118/torchaudio-2.1.2%2Bcu118-cp311-cp311-win_amd64.whl#sha256=598e885648ac94c24920104f185e72fe9f4a9519c2d29b009e47cbc0866e6244")
            elif major == 3 and minor == 10:
                os.system("pip install https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl#sha256=0ddfa0336d678316ff4c35172d85cddab5aa5ded4f781158e725096926491db9")
                os.system("pip install https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-win_amd64.whl#sha256=689f2458e8924c47b7ba9f50dca353423b75214184b905d540f69d9b962b2fdf")
                os.system("pip install https://download.pytorch.org/whl/cu118/torchaudio-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl#sha256=0d02bc0336ee4b3553f0d13f88f61121db2fc21de7b147f4957ecdbcc1dc1c89")
    else:
        os.system("pip install torch torchvision torchaudio")

def setup_windows_installation():
    if not check_python_version_and_confirm():
        return
    if not manual_installation_confirmation():
        return
    cuda_version, cuda_installed = display_cuda_message()
    os.system("python -m pip install --upgrade pip")
    install_pytorch(cuda_version, cuda_installed)
    os.system("pip install -r requirements.txt")
    os.system("pip install bitsandbytes==0.41.2.post2 --prefer-binary --index-url=https://jllllll.github.io/bitsandbytes-windows-webui")

    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if cuda_installed:
        if cuda_version == "11.8":
            if major == 3 and minor == 10:
                os.system("pip install https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp310-cp310-win_amd64.whl#sha256=bb845f1dfe21dec3ccaf2c94adabf46bd604ac5bbfb35379340816914b1ce00a")
            elif major == 3 and minor == 11:
                os.system("pip install https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp311-cp311-win_amd64.whl#sha256=8c232bccf88e19de91b545a2b29886c5684bf5d1f7014b6a3d126e481b5e01ee")
            else:
                print("Unsupported Python version. Please install Python 3.10 or 3.11.")
                return
        else:
            print("Unsupported CUDA version.  Please install CUDA 11.8.")
    else:
        print("No CUDA detected.  Not installing xformers.")

    source_path = "User_Manual/pdf.py"
    target_path = "Lib/site-packages/langchain/document_loaders/parsers/pdf.py"

    def calculate_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as file:
            buf = file.read()
            hasher.update(buf)
        return hasher.hexdigest()

    try:
        source_hash = calculate_hash(source_path)
        print(f"Hash of source file: {source_hash}")
        try:
            target_hash = calculate_hash(target_path)
            print(f"Hash of target file: {target_hash}")
        except FileNotFoundError:
            target_hash = None

        if source_hash != target_hash:
            shutil.copy(source_path, target_path)
            print("File moved as the hashes are different.")
        else:
            print("Files are identical. No action taken.")

    except FileNotFoundError:
        print("Warning: pdf.py not found in User_Manual folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("Installation completed successfully.")
    print("Run 'Python gui.py' to start program.")

setup_windows_installation()
