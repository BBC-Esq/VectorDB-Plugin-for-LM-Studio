<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!</h1>
</div>
<div align="center">
  <h3>Ask questions about your documents and get an answer from LM Studio!</h3>
  <h3>Start a new "issue" if you want to request a feature or report a bug!</h3>
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

> ‼️ Make sure to have at least 🐍[Python 3.10](https://www.python.org/downloads/release/python-31011/) (haven't tested higher).<br>
> ‼️ You must have both [Git](https://git-scm.com/downloads) and [git-lfs](https://git-lfs.com/) installed.<br>


# Installation
> ‼️ For any commands that begin with ```python``` or ```pip``` in these instructions, if you installed Python 3 but still have Python 2 installed, you should use ```Python3``` or ```pip3``` instead to make sure that the correct version of Python is used.
<details>
  <summary>🪟WINDOWS INSTRUCTIONS🪟</summary>
  
### Step 1 - Install GPU Acceleration Software
* 🟢 Nvidia GPU ➜ install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
    > Note that this installation is system-wide and it's not necessary to install within a virtual environment.
* 🔴 AMD GPU - Unfortunately, PyTorch does not currently support AMD GPUs on Windows (only Linux).  However, it may be possible by using WSL within Windows pursuant to the instructions [HERE](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) and then access GPU-acceleration via [HERE](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview).  This might [also be helpful](https://user-images.githubusercontent.com/108230321/275660295-e2d6e097-38c5-4e38-9a1f-f28441ba8812.png)  However, I do not have an AMD GPU so please let me know if you get it working with this method.  If this does work for you, proceed to the instructions below on how to install my program within Linux.

### Step 2 - Obtain Repository
* Download the latest "release" and unzip anywhere on your computer.

### Step 3 - Virtual Environment
* Open the folder containing my repository files.  Open a command prompt.  Create a virtual environment:
```
python -m venv .
```
* Activate the virtual environment:
```
.\Scripts\activate
```

### Step 4 - Upgrade pip
```
python -m pip install --upgrade pip
```

### Step 5 - Install PyTorch
* 🟢 Nvidia GPUs:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
> 🔴 AMD GPU - Unfortunately, PyTorch does not currently support AMD GPUs on Windows (only Linux).  However, it may be possible by using WSL within Windows pursuant to the instructions [HERE](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) and then access GPU-acceleration via [HERE](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview).  However, I do not have an AMD GPU so please let me know if you get it working with this method.  If this does work for you, proceed to the instructions below on how to install my program within Linux.
* 🔵 CPU only:
```
pip install torch torchvision torchaudio
```

### Step 6 - Install Dependencies
```
pip install -r requirements.txt
```

### Step 7 - Double check GPU-Acceleration
```
python check_gpu.py
```
</details>

<details>
  <summary>🐧LINUX INSTRUCTIONS🐧</summary>

### Step 1 - GPU Acceleration Software
  * 🟢 Nvidia GPUs ➜ install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
      > Note that this installation is system-wide and it's not necessary to install within a virtual environment.
  * 🔴 AMD GPUs ➜ install [ROCm version 5.6](https://docs.amd.com/en/docs-5.6.0/deploy/windows/gui/index.html) according to the instructions.
    > Additionally, [this repo](https://github.com/nktice/AMD-AI) might help, but I can't verify since I don't have an AMD GPU nor Linux.

### Step 2 - Obtain Repository
* Download the latest "release" and unzip anywhere on your computer.

### Step 3 - Virtual Environment
* Open the folder containing my repository files.  Open a command prompt.  Create a virtual environment:
```
python -m venv .
```
* Activate the virtual environment:
```
source bin/activate
```

### Step 4 - Update Pip
```
python -m pip install --upgrade pip
```

### Step 5 - Install PyTorch
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

### Step 6 - Install Dependencies
> ‼️On Linux systems run this first: ```sudo apt-get install portaudio19-dev``` and ```sudo apt-get install python3-dev```
```
pip install -r requirements.txt
```

### Step 7 - Double check GPU-acceleration
```
python check_gpu.py
```
</details>

<details>
  <summary>🍎APPLE INSTRUCTIONS🍎</summary>

### Step 1 - GPU Acceleration Software
* All Macs with MacOS 12.3+ come with 🔘 Metal/MPS support, which is the equivalent of CUDA and ROCm.  However, you still need to install [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).

### Step 2 - Obtain Repository
* Download the ZIP file containing the latest release for my repository.  Inside the ZIP file is a folder holding my repository.  Unzip and place this folder anywhere you want on your computer.

### Step 3 - Virtual Environment
* Open the folder containing my repository files.  Open a command prompt.  Create a virtual environment:
```
python -m venv .
```
* Activate the virtual environment:
```
source bin/activate
```

### Step 4 - Update Pip
```
python -m pip install --upgrade pip
```

### Step 5 - Install PyTorch
```
pip install torch torchvision torchaudio
```

### Step 6 - Install Dependencies
> ‼️On MacOS systems run this first: ```brew install portaudio```
```
pip install -r requirements.txt
```

### Step 7 - Double check Metal/MPS-acceleration
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

The program by default uses the ```base.en``` ```float32``` model, which usesabout 1.5 G.B. of memory.  You can change the model size and quantization to achieve the desired quality balanced with memory usage.  Detailed instructions are located in the "Whisper" tab within the GUI.

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
Select one or more files (```.pdf```, ```.docx```, ```.txt```, ```.json```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```).
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
