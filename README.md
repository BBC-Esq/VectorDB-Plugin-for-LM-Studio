<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!</h1>
  <h3>Ask questions about your documents and get an answer from LM Studio!</h3>
</div>
<div align="center">
  <h4>⚡GPU Acceleration⚡</h4>
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
        <td>❌</td>
        <td>✅</td>
        <td>ROCm 5.6</td>
      </tr>
      <tr>
        <td>Apple/Metal</td>
        <td colspan="3" align="center"> ✅ </td>
      </tr>
    </tbody>
  </table>
</div>

<div align="center"> <h2>🔥Requirements🔥</h2></div>
You must install the these before following the installation instructions below:

> ‼️ 🐍[Python 3.10](https://www.python.org/downloads/release/python-31011/) (I have not tested above this.).<br>
> ‼️ [Git](https://git-scm.com/downloads)<br>
> ‼️ [Git Large File Storage](https://git-lfs.com/).<br>
> ‼️ [Pandoc](https://github.com/jgm/pandoc) (only if you want to process ```.rtf``` files).

<div align="center"> <h2>🔥Installation🔥</h2>
‼️If you have Python 2 and Python 3 installed on your system, make sure and use "Python3" and "pip3" instead when installing.‼️
</div><br>
<details>
  <summary>🪟WINDOWS INSTRUCTIONS🪟</summary>
  
### Step 1
🟢 Nvidia GPU ➜ Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)<br>
🔴 AMD GPU - Unfortunately, PyTorch does not currently support AMD GPUs on Windows.  It's only supported on Linux.  There are several ways to possibly get around this limitation, but I'm unable to verify since I don't have an AMD GPU.  See [HERE](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview), [HERE](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview), and possibly [HERE](https://user-images.githubusercontent.com/108230321/275660295-e2d6e097-38c5-4e38-9a1f-f28441ba8812.png).
### Step 2
Download the ZIP file from the latest "release," unzip anywhere on your computer, and go into the ```src``` folder.
### Step 3
Within the ```src``` folder, open a command prompt and create a virtual environment:
```
python -m venv .
```
### Step 4
Activate the virtual environment:
```
.\Scripts\activate
```
### Step 5
```
python setup.py
```
> And just follow the instructions.

### Optional Step 6 - Double check GPU-Acceleration
Run this script if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```
</details>

<details>
  <summary>🐧LINUX INSTRUCTIONS🐧</summary>

### Step 1
🟢 Nvidia GPUs ➜ Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)<br>
🔴 AMD GPUs ➜ Install [ROCm version 5.6](https://docs.amd.com/en/docs-5.6.0/deploy/windows/gui/index.html).
> [THIS REPO](https://github.com/nktice/AMD-AI) might also help if AMD's instructions aren't clear.

### Step 2
Download the ZIP file from the latest "release," unzip anywhere on your computer, and go into the ```src``` folder.
### Step 3
Within the ```src``` folder, open a terminal window and create a virtual environment:
```
python -m venv .
```
### Step 4
Activate the virtual environment:
```
source bin/activate
```
### Step 5
```
python -m pip install --upgrade pip
```
### Step 6
🟢 Nvidia GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
🔴 AMD GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```
🔵 CPU only:
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
All Macs with MacOS 12.3+ come with 🔘 Metal/MPS, which is Apple's implementation of gpu-acceleration (like CUDA for Nvidia and ROCm for AMD).  I'm not sure if it's possible to install on an older MacOS since I don't have an Apple.
### Step 2
Install [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).
### Step 3
Download the ZIP file from the latest "release," unzip anywhere on your computer, and go into the ```src``` folder.
### Step 4
Within the ```src``` folder, open a terminal window and create a virtual environment:
```
python -m venv .
```
### Step 5
Activate the virtual environment:
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

<div align="center"> <h2>🔥Transcription🔥</h2></div>

> As of release v2.1+, my program allows you to transcribe a question into the clipboard and paste it into LM Studio.  It uses the "faster-whisper" library, which relies upon the powerful Ctranslate2 library and the state-of-the-art "Whisper" models.

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

### Compatibility Checker
After following the installation instructions above, you can check which quantized versions of the Whisper models your CPU and GPU support.  On Windows, use [```ctranslate2_compatibility.exe```](https://github.com/BBC-Esq/ctranslate2-compatibility-checker/releases/tag/v1.0) and on Linux or MacOS follow the instructions [HERE](https://github.com/BBC-Esq/ctranslate2-compatibility-checker).
> As of Release v2.5, however, this is no longer mandatory because the program only displays compatible quantizations to choose from.

<div align="center"> <h2>🔥Usage🔥</h2></div>
<details>
  <summary>🔥USAGE INSTRUCTIONS🔥</summary>

### Step 1 - Virtual Environment
Make sure you are in theh ```src``` folder, have opened a command prompt/terminal, and activated the virtual environment (see installation instructions).
### Step 2
Run Program
```
python gui.py
```
> Only systems running Windows with an Nvidia GPU will display metrics in the GUI.  Feel free to request that I add AMD or Apple support.
### Step 3
The download embedding model button lets you choose to download multiple embedding models.  The command prompt/terminal will state when the download is complete and unpacked.  Don't attempt to create the vector database before.
### Step 4
The set model directory allows you to choose which embedding model to create the vector database.  You can choose any of the embedding models you previously downloaded to see which works best.  Remember, you must recreate the database if you want to use a different embedding model.  Creating the database with one embedding model and then trying to search with a different embedding model will throw an error.  Recreating the vector database will automatically delete the old one.
### Step 5
The choose documents allows you to select which documents you want in the database.  Symbolic links to the files are put within the "Docs_for_DB" folder, but you can also manually copy/paste files into the folder if you prefer having the actual files there.  Also, you can click this button multiple times if your files are in different directories and doing will not delete the files you've already added.  You can remove some/all files to be processed by simply deleting them from the "Docs_for_DB" folder.
Remember, you must recreate the database anytime you want to add/remove documents.  Adding/removing documents from the "Docs_for_DB" folder does not automatically modify the database.
The supported file types are: ```.pdf```, ```.docx```, ```.txt```, ```.json```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```.
> ‼️ PDF files must have had OCR done on them.<br>
> ‼️ As of release 2.6.1, you can now transcribe audio files and put them in the database as well!

### Step 6
The create database button is self-explanatory.  The command prompt will tell you when it's done and it's safe to search.  However, you can also tell by seeing the GPU usage spike if you're using gpu-acceleration.  Do not attempt to query the database until it's created.

### Step 7
1) After the database is created, open LM Studio and load a model.
> ‼️ My program uses the Llama2 prompt format by default (although it can be changed).  Therefore, I highly recommend that you only use Llama2 based models unless you know for sure how to modify the prefix and suffix for the prompts for various models.
2) Click the server tab on the left side.
3) Click "Start Server" in the server tab.

### Step 8
Type or transcribe a question into my program and click "Submit Questions."  The vector database will be queried and your question along with the results will be fed to LM Studio for an answer.

</details>

<div align="center"><h2>🔥Contact🔥</h2></div>

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
