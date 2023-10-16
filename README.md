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
        <td>❌</td>
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

# Installation

> First, make sure have [Python 3.10+](https://www.python.org/downloads/release/python-31011/).  Also, you must have both [Git](https://git-scm.com/downloads) and [git-lfs](https://git-lfs.com/) installed.<br>
> NOTE: For any ```python``` or ```pip``` commands in these instructions, if you installed Python 3 but still have Python 2 installed, you should use ```Python3``` or ```pip3``` instead to make sure that the correct version of Python is used.

<details>
  <summary>🪟 WINDOWS INSTRUCTIONS</summary>
  
### Step 1 - Install GPU Acceleration Software
* Nvidia GPU ➜ install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
    > Note that this installation is system-wide and it's not necessary to install within a virtual environment.
* AMD GPU - Unfortuantely, PyTorch does not currently support AMD GPUs on Windows (only Linux).

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
* Nvidia GPUs:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
> Unfortunately, PyTorch only currently supports AMD GPU's on Linux system.
* CPU Only command:
```
pip install torch torchvision torchaudio
```

### Step 7 - Install Dependencies
```
pip install -r requirements.txt
```

### Step 6 - Doublecheck GPU-Acceleration
```
python check_gpu.py
```
</details>

<details>
  <summary>🐧LINUX INSTRUCTIONS</summary>

### Step 1 - GPU Acceleration Software
  * Nvidia GPUs ➜ install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
      > Note that this installation is system-wide and it's not necessary to install within a virtual environment.
  * AMD GPUs ➜ install ROCm version 5.6 according to the instructions https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html](https://rocm.docs.amd.com/en/docs-5.6.1/.
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
.\Scripts\activate
```
  >On Linux try ```source bin/activate``` if the above doesn't work.

### Step 4 - Update Pip
```
python -m pip install --upgrade pip
```

### Step 5 - Install PyTorch
* Nvidia GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
* AMD GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```
* CPU Only command:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 6 - Install Dependencies
```
pip install -r requirements.txt
```

### Step 7 - Doublecheck GPU-acceleration
```
python check_gpu.py
```
</details>

<details>
  <summary>🍎 APPLE INSTRUCTIONS</summary>

### Step 1 - GPU Acceleration Software
* All Macs with MacOS 12.3+ come with Metal/MPS support, which is the equivalent of CUDA (NVIDIA) and ROCm (AMD).  However, you still need to install [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).

### Step 2 - Obtain Repository
* Download the ZIP file containing the latest release for my repository.  Inside the ZIP file is a folder holding my repository.  Unzip and place this folder anywhere you want on your computer.

### Step 3 - Virtual Environment
* Open the folder containing my repository files.  Open a command prompt.  Create a virtual environment:
```
python3 -m venv .
```
* Activate the virtual environment:
```
source bin/activate
```

### Step 4 - Update Pip
```
python3 -m pip install --upgrade pip
```

### Step 5 - Install PyTorch
```
pip3 install torch torchvision torchaudio
```

### Step 7 - Install Dependencies
```
pip3 install -r requirements.txt
```

### Step 8 - Doublecheck Metal/MPS-acceleration
```
python3 check_gpu.py
```

</details>

# Transcription Instructions

> As of release 2.1+, my program allows you to speak a question and have it transcribed to the system clipboard, which you can then paste into the question box.  This is based on the "faster-whisper" library, which relies upon the powerful Ctranslate2 library and the state-of-the-art "Whisper" models.  Ctranslate2 supports both CPU and GPU acceleration as follows:

<details>
  <summary>🔥TRANSCRIPTION INSTRUCTIONS</summary>
  
### Step 1 - Faster-Whisper Compatibility

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

  > Ctranslate2 will use the best acceleration method available.  However, if you encounter any problems with the voice transcript that prevents the program from working simply install a release prior to 2.1 and follow the normal installation instructions.

### Step2 - Ctranslate2 Compatibility Checker

Easily download and run [```ctranslate2_compatibility.exe```](https://github.com/BBC-Esq/ctranslate2-compatibility-checker/releases/tag/v1.0) to check which quantizations your CPU and GPU support.

### Step 3 - Option to Change Ctranslate2 Whisper Models
The program automatically downloads and uses the ```base.en``` Ctranslate2 Whisper by default.  To use more/less powerful models, change [```line 13```](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/blob/ee718ea9d37dc3d21b2c14cdcbb93f6b3b9385ed/voice_recorder_module.py#L13) of ```voice_recorder_module.py``` pursuant to the instructions contained in the "Whisper" tab within the GUI.

</details>

# Usage
<details>
  <summary>🔥USAGE INSTRUCTIONS</summary>

### Step 1 - Virtual Environment
Open a command prompt within my repository folder and activate the virtual environment:<br>
> NOTE: For Macs the preferred command is ```source bin/activate```
```
.\Scripts\activate
```

### Step 2 - Run Program
```
python gui.py
```
* NOTE: Only systems running Windows with an Nvidia GPU will display metrics in the GUI.  Working on a fix.

### Step 3 - Download Embedding Model
The efficacy of an embedding model depends on both the type of text and type of questions you intend to ask.  Do some research on the different models, but I've selected ones that are overall good.
> You must wait until the download is complete AND unpacked before trying to create the database.

### Step 4 - Select Embedding Model Directory
Selects the directory of the embedding model you want to use.

### Step 5 - Choose Documents for Database
Select one or more files (```.pdf```, ```.docx```, ```.txt```, ```.json```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```).

### Step 6 - Create Vector Database
GPU usage will spike as the vector database is created.  Wait for this to complete before querying database.

### Step 7 - Start LM Studio
Open LM Studio and load a model.  Click the server tab on the lefhand side.  Click "Start Server" in the server tab.
> Only Llama2-based models are currently supported due to their prompt format.

### Step 8 - Submit Question
Enter a question and click "submit question."  The vector database will be queried and your question along with the results will be fed to LM Studio for an answer.

### Step 9 - Transcribe Question Instead
Click start record button.  Talk.  Click stop button.  Paste transcription into question box.  Click Submit Question.

</details>

# Contact

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
