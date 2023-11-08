<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!</h1>
</div>
<div align="center">
  <h3>Ask questions about your documents and get an answer from LM Studio!</h3>
</div>

<div align="center">
  <h4>⚡GPU Acceleration⚡
  <table>
    <thead>
      <tr>
        <th>GPU</th>
        <th>Windows</th>
        <th>Linux</th>
        <th>Requirements</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Nvidia</td>
        <td>✅</td>
        <td>✅</td>
        <td>CUDA</td>
      </tr>
      <tr>
        <td>AMD</td>
        <td>⚠️ (see below)</td>
        <td>✅</td>
        <td>ROCm 5.6</td>
      </tr>
      <tr>
        <td>Apple/Metal</td>
        <td colspan="3" align="center"> ✅ </td>
      </tr>
    </tbody>
  </table></h4>
</div>

# Prerequisites

You must install the these before using my program:

> ‼️ 🐍[Python 3.10](https://www.python.org/downloads/release/python-31011/) (I have not tested above this.).<br>
> ‼️ [Git](https://git-scm.com/downloads)<br>
> ‼️ [Git Large File Storage](https://git-lfs.com/).<br>
> ‼️ [Pandoc](https://github.com/jgm/pandoc) (only if you want to process ```.rtf``` files).

# Installation
> ‼️‼️ If you have Python 2 and Python 3 installed on your system, make sure and use ```Python3``` andn ```pip3``` instead when installing.
<details>
  <summary>🪟WINDOWS INSTRUCTIONS🪟</summary>
  
### Step 1
* 🟢 Nvidia GPU ➜ Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* 🔴 AMD GPU - Unfortunately, PyTorch does not currently support AMD GPUs on Windows.  It's only supported on Linux.  There are several ways to possibly get around this limitation, but I'm unable to verify since I don't have an AMD GPU.  See [HERE](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview), [HERE](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview), and possibly [HERE](https://user-images.githubusercontent.com/108230321/275660295-e2d6e097-38c5-4e38-9a1f-f28441ba8812.png).
### Step 2
* Download the ZIP file from the latest "release," unzip anywhere on your computer, and go into the ```src``` folder.
### Step 3
* Within the ```src``` folder, open a command prompt and create a virtual environment:
```
python -m venv .
```
### Step 4
* Activate the virtual environment:
```
.\Scripts\activate
```
### Step 5
```
python -m pip install --upgrade pip
```
### Step 6
* 🟢 Nvidia GPUs:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
* 🔴 AMD GPUs - To reiterate, PyTorch does not supprot AMD GPUs Windows, only Linux.
* 🔵 CPU only:
```
pip install torch torchvision torchaudio
```
### Step 7
```
pip install -r requirements.txt
```
### Optional Step 8 - Double check GPU-Acceleration
Run this script if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```
</details>

<details>
  <summary>🐧LINUX INSTRUCTIONS🐧</summary>

### Step 1
  * 🟢 Nvidia GPUs ➜ Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
  * 🔴 AMD GPUs ➜ Install [ROCm version 5.6](https://docs.amd.com/en/docs-5.6.0/deploy/windows/gui/index.html).
    > [THIS REPO](https://github.com/nktice/AMD-AI) might also help if AMD's instructions aren't clear.
### Step 2
* Download the ZIP file from the latest "release," unzip anywhere on your computer, and go into the ```src``` folder.
### Step 3
* Within the ```src``` folder, open a terminal window and create a virtual environment:
```
python -m venv .
```
### Step 4
* Activate the virtual environment:
```
source bin/activate
```
### Step 5
```
python -m pip install --upgrade pip
```
### Step 6
* 🟢 Nvidia GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
* 🔴 AMD GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```
* 🔵 CPU only:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
### Step 7
```
sudo apt-get install portaudio19-dev
```
### Step 8
```
sudo apt-get install python3-dev
```
### Step 9
```
pip install -r requirements.txt
```
### Optional Step 10
Run this script if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```
</details>

<details>
  <summary>🍎APPLE INSTRUCTIONS🍎</summary>

### Step 1
* All Macs with MacOS 12.3+ come with 🔘 Metal/MPS, which is Apple's implementation of gpu-acceleration (like CUDA for Nvidia and ROCm for AMD).  I'm not sure if it's possible to install on an older MacOS since I don't have an Apple.
### Step 2
* Install [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).
### Step 3
* Download the ZIP file from the latest "release," unzip anywhere on your computer, and go into the ```src``` folder.
### Step 4
* Within the ```src``` folder, open a terminal window and create a virtual environment:
```
python -m venv .
```
### Step 5
* Activate the virtual environment:
```
source bin/activate
```
### Step 6
```
python -m pip install --upgrade pip
```
### Step 7
```
pip install torch torchvision torchaudio
```
### Step 8
```
brew install portaudio
```
### Step 9
```
pip install -r requirements.txt
```
### Optional Step 10
Run this script if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```

</details>

# Transcription Instructions

> As of release 2.1+, my program allows you to speak a question and have it transcribed to the system clipboard, which you can then paste into the LM Studio question box.  It uses the "faster-whisper" library, which relies upon the powerful Ctranslate2 library and the state-of-the-art "Whisper" models.

<details>
  <summary>🔥TRANSCRIPTION INSTRUCTIONS🔥</summary>
  
### Compatibility Overview

<div align="center">
  <h4>⚡Transcription Acceleration⚡</h4>
  <table>
    <thead>
      <tr>
        <th></th>
        <th>Acceleration Support</th>
        <th>Requirements</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Intel CPU</td>
        <td>✅</td>
        <td></td>
      </tr>
      <tr>
        <td>AMD CPU</td>
        <td>✅</td>
        <td></td>
      </tr>
      <tr>
        <td>Nvidia GPU</td>
        <td>✅</td>
        <td>CUDA</td>
      </tr>
      <tr>
        <td>AMD GPU</td>
        <td>❌</td>
        <td>Will default to CPU</td>
      </tr>
      <tr>
        <td>Apple CPU</td>
        <td>✅</td>
        <td></td>
      </tr>
      <tr>
        <td>Apple Metal/MPS</td>
        <td>❌</td>
        <td>Will default to CPU</td>
      </tr>
    </tbody>
  </table>
</div>

  > The type of acceleration you'll be using also determines the supported quantizations (discussed below).

### Compatibility Checker

On Windows, simpy run [```ctranslate2_compatibility.exe```](https://github.com/BBC-Esq/ctranslate2-compatibility-checker/releases/tag/v1.0).<br>

On Linux or MacOS, follow the instructions [HERE](https://github.com/BBC-Esq/ctranslate2-compatibility-checker).

### Changing Transcription Model or Quantization

All transcription settings can be changed in-program as of Version 2.5!

</details>

# Usage
<details>
  <summary>🔥USAGE INSTRUCTIONS🔥</summary>

### Step 1 - Virtual Environment
Open a command prompt within my repository folder and activate the virtual environment:<br>
> NOTE: For 🍎Macs and 🐧Linux the command is: ```source bin/activate```
```
.\Scripts\activate
```

### Step 2 - Run Program
```
python gui.py
```
* ‼️ Only systems running Windows with an Nvidia GPU will display metrics in the GUI.  Feel free to request that I add AMD or Apple support.

### Step 3 - Download Embedding Model
The best embedding model depends on the type of text being entered into the vector database and the style of question you intend to ask.  I've selected multiple models that are good, but feel free to read about each one because they're suitable for different tasks.
> ‼️ You must wait until the download is complete AND unpacked before trying to create the database.

### Step 4 - Select Embedding Model Directory
Selects the directory of the embedding model you want to use.

### Step 5 - Choose Documents for Database
Select one or more files (```.pdf```, ```.docx```, ```.txt```, ```.json```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```).
> ‼️ PDF files must already have had OCR done on them.  Put in a feature request if you want to incorporate Pytesseract for OCR.

### Step 6 - Create Vector Database
GPU usage will spike as the vector database is created.  Wait for this to complete before querying database.

### Step 7 - Start LM Studio
Open LM Studio and load a model.  Click the server tab on the left side.  Click "Start Server" in the server tab.
> ‼️ Only Llama2-based models are currently supported due to their prompt format.

### Step 8 - Submit Question
Enter a question and click "submit question."  The vector database will be queried and your question along with the results will be fed to LM Studio for an answer.

### Step 9 - Transcribe Question Instead
Click the 'Start Record' button, speak, and then click the 'Stop' button.  Paste transcription into question box and click Submit Question.

</details>

# Contact

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
