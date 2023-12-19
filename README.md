<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!</h1>
  <h3>Ask questions about your documents and get an answer from LM Studio!</h3>
</div>
<div align="center">
  <h4>⚡GPU Acceleration for Database⚡</h4>
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
        <td>CUDA 11.8</td>
      </tr>
      <tr>
        <td>AMD</td>
        <td>❌</td>
        <td>✅</td>
        <td>ROCm 5.6 (5.7 unknown)</td>
      </tr>
      <tr>
        <td>Apple/Metal</td>
        <td colspan="3" align="center"> ✅ </td>
      </tr>
    </tbody>
  </table>
</div>

<div align="center"> <h2><u>REQUIREMENTS</h2></div>
You <b>MUST</b> install these before installing my program:<p>

1) 🐍[Python 3.10](https://www.python.org/downloads/release/python-31011/) or [Python 3.11](https://www.python.org/downloads/release/python-3116/) (PyTorch is NOT compatible with 3.12, at least as of 12/19/2023).
2) [Git](https://git-scm.com/downloads)
3) [Git Large File Storage](https://git-lfs.com/).
4) [Pandoc](https://github.com/jgm/pandoc) (only if you want to process ```.rtf``` files).

<div align="center"> <h2>INSTALLATION</h2></div>

<details>
  <summary>🪟WINDOWS INSTRUCTIONS🪟</summary>
  
### Step 1
🟢 Nvidia GPU ➜ [Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
> CUDA 12+ is currently NOT compatible since the faster-whisper library is only compatible up to CUDA 11.8.  This will be addressed in upcoming releases.<br>

🔴 AMD GPU - PyTorch does not currently support AMD GPUs on Windows - only Linux.  There are several possible workarounds but I'm unable to verify since I don't have an AMD GPU.  You can look [HERE](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html), [HERE](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview), [HERE](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview), and possibly [HERE](https://user-images.githubusercontent.com/108230321/275660295-e2d6e097-38c5-4e38-9a1f-f28441ba8812.png).
### Step 2
Download the ZIP file from the latest "release" and extract the contents anywhere you want.  DO NOT simply clone this repository...there may be incremental changes to scripts that will be undone before an official release is created.
### Step 3
Navigate to the ```src``` folder, open a command prompt, and create a virtual environment:
```
python -m venv .
```
### Step 4
Activate the virtual environment:
```
.\Scripts\activate
```
### Step 5
Run setup:
```
python setup.py
```

### Optional Step 6
Run this command if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
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
Download the ZIP file from the latest "release" and extract the contents anywhere you want.  DO NOT simply clone this repository...there may be incremental changes to scripts that will be undone before an official release is created.
### Step 3
Navigate to the ```src``` folder, open a command prompt, and create a virtual environment:
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
### Step 11
🚨🚨🚨You must copy the ```pdf.py``` file within the ```User_Manual``` folder to the following folder: ```[folder holding this program]\Lib\site-packages\langchain\document_loaders\parsers```.🚨🚨🚨.<br><br>This will REPLACE the ```pdf.py``` file there.  This is crucial to get the 80x speedup on PDF loading in version 2.7.2+.  Moreover, the PDF loading WILL NOT WORK at all unless you do this properly.

</details>

<details>
  <summary>🍎APPLE INSTRUCTIONS🍎</summary>

### Step 1
All Macs with MacOS 12.3+ come with 🔘 Metal/MPS, which is Apple's implementation of gpu-acceleration (like CUDA for Nvidia and ROCm for AMD).  I'm not sure if it's possible to install on an older MacOS since I don't have an Apple.
### Step 2
Install [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).
### Step 3
Download the ZIP file from the latest "release" and extract the contents anywhere you want.  DO NOT simply clone this repository...there may be incremental changes to scripts that will be undone before an official release is created.
### Step 4
Navigate to the ```src``` folder, open a command prompt, and create a virtual environment:
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
* Homebrew can be installed with:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
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
### Step 11
🚨🚨🚨You must copy the ```pdf.py``` file within the ```User_Manual``` folder to the following folder: ```[folder holding this program]\Lib\site-packages\langchain\document_loaders\parsers```.🚨🚨🚨.<br><br>This will REPLACE the ```pdf.py``` file there.  This is crucial to get the 80x speedup on PDF loading in version 2.7.2+.  Moreover, the PDF loading WILL NOT WORK at all unless you do this properly.

</details>

<div align="center"> <h2>USING THE PROGRAM</h2></div>
<details>
  <summary>🖥️INSTRUCTIONS🖥️</summary>

## Activate Virtual Environment
* Open a command prompt/terminal from within the ```src``` folder and activate the virtual environment (see installation instructions above).
## Start the Program
```
python gui.py
```
> NOTE - only systems running Windows with an Nvidia GPU will display metrics in the GUI.

# 🔥Important🔥
* Read the User Guide before proceeding further!

## Download Embedding Model
* Choose the embedding model you want to download.  Do not attempt to create the vector database until the command prompt says that the model is downloaded AND unpacked.
## Set Model Directory
* Choose the directory containing the embedding model you want to use to create the vector database.
  > Do not simply choose the "Embedding_Models" folder.
## Adding Documents
* Choose the documents you want to enter into the vector database.  You can select multiple documents at once and/or click this button multiple times.
  > NOTE - Symbolic links to the files are created within the "Docs_for_DB" folder, not the actual files.
* Supported file types are ```.pdf```, ```.docx```, ```.txt```, ```.json```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```.
* You can also transcribe audio files to ```.txt``` to be put into the database.  Look within the "Tools" tab.
  > ⚠️ Anytime you add documents you must recreate the vector database.

## Removing Documents
* You must manually delete the symbolic link/links from the "Docs_for_DB" folder and recreate the vector database.

## Creating the Databaase
* The create database button creates the vector database!
  > ⚠️ Wait until the command prompt says "persisted" before proceeding to the next step.
  > ⚠️ Remember, you must recreate the database anytime you add/remove documents.

## Load LM Studio
* Open LM Studio and load a model.
  > ⚠️ Only models that use the Llama-2 prompt format are supported by default.  You can change the "prefix" or "suffix" to test out other models.
* Click the server tab on the left side.
* Click "Start Server" in the server tab.
  * ⚠️ As of LM Studio ```.2.8```, there's a setting to allow you to set the prompt format (and other settings) within LM Studio.
  * It is recommended to DISABLE this setting to allow the program to work out-of-the-box.  However, experienced users can click the "disable" prompt formatting checkbox in the "Server" settings, which will enable you to experiment with the prompt formats provided by LM Studio.

## Search Database
* Type your question and click "Submit Questions."
* You can speak your question to LM Studio using the powerful Ctranslate2 library and state-of-the-art "Whisper" models.  Simply click the Start Recording button...talk...click the Stop Recording button.

<div align="center">
  <h4>⚡Acceleration for Transcription⚡</h4>
  <table>
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
        <td>Requires CUDA 11.8 (not 12.1)</td>
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

## Tools
* You can transcribe audio files to the database folder, which will then be put into the database when you create it.
* However, just like transcribing a question, if you are using an NVIDIA GPU you must have installed ⚠️CUDA 11.8⚠️ and not CUDA 12.1 since the ```faster-whisper``` library currently doesn't support 12.1 WITHOUTH having to compile from source, which most users don't/can't do.

</details>

<div align="center"><h2>CONTACT</h2></div>

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example4.png" alt="Example Image4">
</div>
