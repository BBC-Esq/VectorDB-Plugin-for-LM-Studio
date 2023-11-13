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
You must install these before following the installation instructions below:

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
🟢 Nvidia GPU ➜ Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive).<br>
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

> My program allows you to transcribe a question into the clipboard and paste it into LM Studio.  It uses the powerful Ctranslate2 library and the state-of-the-art "Whisper" models.

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

### Activate Virtual Environment
Make sure you are in theh ```src``` folder, have opened a command prompt/terminal, and activated the virtual environment (see installation instructions).
### Run Program
```
python gui.py
```
> Only systems running Windows with an Nvidia GPU will display metrics in the GUI.  Feel free to request that I add AMD or Apple support.
### Download Embedding Model
The download embedding model button lets you choose to download multiple embedding models.  The command prompt/terminal will state when the download is complete and unpacked.  Don't attempt to create the vector database before it's done.
### Set Model Directory
The set model directory button allows you to choose which embedding model to create/query the vector database.  You can choose any of the embedding models you previously downloaded by selecting the folder in which the model files were downloaded to.
### Choose Documents for Database
The choose documents button allows you to select which documents you want in the database.  Symbolic links to the files are put within the "Docs_for_DB" folder, not the actual files, but you can manually drag and drop files there as well if you want.  Feel free to select multiple files or click the add files button multiple times.  To delete any files you have to manually delete them from the "Docs_for_DB" folder, however.
> Remember, anytime you add/remove files you must recreate the vector database.

The file types that are supported are ```.pdf```, ```.docx```, ```.txt```, ```.json```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```.<br>
> ‼️ However, PDF files must have had OCR done on them.

As of release 2.6.1, you can also transcribe audio files for the database.  The transcription process will automatically create a ```.txt``` file within the "Docs_for_DB" folder.
> Remember, anytime you add/remove documents you must re-create the database.

### Create Databaase
The create database button...wait for it...creates the vector database!  The command prompt will state when it's been "persisted."  You should only conduct a search after you see this message.

### Setup LM Studio
1) Before searching, open LM Studio and load a model.
> ‼️ Remember, only models that use the Llama-2 prompt format are supported by default.  You can change the "prefix" or "suffix" to test out other models, but for 99% of use cases a basic model that uses the Llama-2 prompt format is sufficient.
> Mistral models use the formate and are excellent.
2) ...then click the server tab on the left side.
3) ...then click "Start Server" in the server tab.

### Search Database
Now you type/transcribe your question and click "Submit Questions."  The vector database will be queried.  Your question along with any "contexts" from the database will be sent to the LLM within LM Studio for answer!

</details>

<div align="center"><h2>🔥Contact🔥</h2></div>

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example2.png" alt="Example Image2">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example3.png" alt="Example Image3">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example4.png" alt="Example Image4">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example5.png" alt="Example Image5">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example6.png" alt="Example Image6">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example7.png" alt="Example Image7">
</div>
