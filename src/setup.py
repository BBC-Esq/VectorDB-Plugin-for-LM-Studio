import sys
import subprocess
import os
import shutil
import hashlib
import tkinter as tk
from tkinter import messagebox
import constants as c

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
    if not tkinter_message_box("Confirmation", c.MESSAGE_GIT, type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", c.MESSAGE_GIT_LFS, type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", c.MESSAGE_PANDOC, type="yesno", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", c.MESSAGE_MS_BUILD_TOOLS, type="yesno", yes_no=True):
        return False
    return True

def install_pytorch(cuda_version, cuda_installed):
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if cuda_installed:
        if cuda_version == "11.8":
            if major == 3 and minor == 11:
                os.system(c.PYTORCH_CUDA_11_8_CP311)
                os.system(c.PYTORCH_CUDA_11_8_TORCHVISION_CP311)
                os.system(c.PYTORCH_CUDA_11_8_TORCHAUDIO_CP311)
            elif major == 3 and minor == 10:
                os.system(c.PYTORCH_CUDA_11_8_CP310)
                os.system(c.PYTORCH_CUDA_11_8_TORCHVISION_CP310)
                os.system(c.PYTORCH_CUDA_11_8_TORCHAUDIO_CP310)
    else:
        if major == 3 and minor == 11:
            os.system(c.PYTORCH_CPU_CP311)
            os.system(c.PYTORCH_CPU_TORCHAUDIO_CP311)
            os.system(c.PYTORCH_CPU_TORCHVISION_CP311)
        elif major == 3 and minor == 10:
            os.system(c.PYTORCH_CPU_CP310)
            os.system(c.PYTORCH_CPU_TORCHAUDIO_CP310)
            os.system(c.PYTORCH_CPU_TORCHVISION_CP310)

def setup_windows_installation():
    if not check_python_version_and_confirm():
        return
    if not manual_installation_confirmation():
        return
    cuda_version, cuda_installed = display_cuda_message()
    os.system("python -m pip install --upgrade pip")
    install_pytorch(cuda_version, cuda_installed)
    os.system("pip install -r requirements.txt")
    os.system(c.BITSANDBYTES_INSTALL_COMMAND)

    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if cuda_installed:
        if cuda_version == "11.8":
            if major == 3 and minor == 10:
                os.system(c.XFORMERS_CUDA_11_8_CP310)
                os.system(c.NVIDIA_ML_PY_CP310)
            elif major == 3 and minor == 11:
                os.system(c.XFORMERS_CUDA_11_8_CP311)
                os.system(c.NVIDIA_ML_PY_CP311)
            else:
                print("Unsupported Python version. Please install Python 3.10 or 3.11.")
                return
        else:
            print("Unsupported CUDA version.  Please install CUDA 11.8.")
    else:
        print("No CUDA detected.  Not installing xformers.")

    def calculate_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as file:
            buf = file.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def find_site_packages_path():
        for path in sys.path:
            if "site-packages" in path.lower():
                return path
        return None

    def construct_target_path(base_path):
        target_sub_path = os.path.join("langchain", "document_loaders", "parsers")
        parts = target_sub_path.split(os.sep)
        for part in parts:
            found = next((d for d in os.listdir(base_path) if d.lower() == part.lower()), None)
            if found:
                base_path = os.path.join(base_path, found)
            else:
                print(f"Expected directory '{part}' not found under '{base_path}'.")
                return None
        return os.path.join(base_path, "pdf.py")

    site_packages_path = find_site_packages_path()
    if site_packages_path:
        target_pdf_path = construct_target_path(site_packages_path)
        if target_pdf_path:
            source_path = "User_Manual/pdf.py"
            try:
                source_hash = calculate_hash(source_path)
                print(f"Hash of source file: {source_hash}")
                try:
                    target_hash = calculate_hash(target_pdf_path)
                    print(f"Hash of target file: {target_hash}")
                except FileNotFoundError:
                    target_hash = None

                if source_hash != target_hash:
                    shutil.copy(source_path, target_pdf_path)
                    print("PDF.py file moved as the hashes are different.")
                else:
                    print("PDF.py files are identical. No action taken.")
            except FileNotFoundError:
                print("Warning: Source PDF.py not found in User_Manual folder.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Target path for PDF.py could not be constructed.")
    else:
        print("Site-packages directory not found.")

setup_windows_installation()
