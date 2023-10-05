<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!</h1>
</div>
<div align="center">
  <h2>Ask questions about your documents and get an answer from an LLM!</h2>
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
        <td>ROCm 5.4.2</td>
      </tr>
      <tr>
        <td>Apple/Metal</td>
        <td colspan="3" align="center"> ✅ </td>
      </tr>
    </tbody>
  </table></h4>
</div>

## IMPORTANT: Please read the "Transcription Instructions" first before installing.

# Installation

> First, make sure have [Python 3.10+](https://www.python.org/downloads/release/python-31011/).  Also, you must have both [Git](https://git-scm.com/downloads) and [git-lfs](https://git-lfs.com/) installed.<br>

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
  * AMD GPUs ➜ install ROCm version 5.4.2 according to the instructions [HERE](https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html) and [HERE](https://rocmdocs.amd.com/en/latest/deploy/linux/index.html)
  * Additionally, [this repo](https://github.com/nktice/AMD-AI) might help, but I can't verify since I don't have an AMD GPU nor Linux.

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
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
  <summary>🍎APPLE INSTRUCTIONS</summary>

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

> As of release 2.1+, my program includes a transcription feature that allows you to speak a question and have it transcribed to the system clipboard, which you can then paste into the question box - thus saving time.  This is based on the "faster-whisper" library, which, in-turn, relies upon the powerful Ctranslate2 library and the state-of-the-art "Whisper" models.

### Step 1 - Faster-Whisper Compatibility

<details>
  <summary>TRANSCRIPTION INSTRUCTIONS</summary>

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
        <td> Will default to CPU</td>
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

The faster-whisper documentation is sometimes not clear.  For example, Ctranslate2's documentation states one thing but faster-whisper (which relies on Ctranslate2) simply doesn't discuss it.  Therefore, if you encounter any problems with the transcription functionality causing the entire program to ```fail```, simply install a release prior to 2.1 and follow the normal installation instructions.

### Step 2 - Obtain Quantized Ctranslate2 Whisper Models
In addition to my repository, you must download one or more models that are in ZIP files in [Release 2.1 specifically](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v2.1).  Feel free to test different models out!  Smallest (lesser quality) to bigger (higher quality) are as follows:
* ```tiny/tiny.en```
* ```base/base.en```
* ```small/small.en```
* ```medium/medium.en```

  > Contact me if you want the ```large-v2``` model - it's too large to upload.  Moreover, if you're super-tech-savvy and want other [quantizations](https://opennmt.net/CTranslate2/quantization.html) for even higher quality or more customizability, contact me, I have the following additional quants for each size:
    > * ```float32```, ```bfloat16```, ```float16```, ```Int8_bfloat16```, ```int8_float16```, and ```int8```
</details>

# Usage
<details>
  <summary>USAGE INSTRUCTIONS</summary>
  
### Step 1 - Download Transctiption Model
> Only do this if you've read the transcription instructions and are using release 2.1+.

* Download one or more of the ZIP files [HERE](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v2.1) and put the folder within the ZIP file in my repository folder.  It must be within the repository folder otherwise I won't work.  My program defaults to the ```small.en``` model so try that first.
  > Feel free to try other models!  Simply change ```line 18``` of the ```voice_recorder_module.py``` script to match another model's exact folder name.

### Step 2 - Virtual Environment
> Open a command prompt within my repository folder and activate the virtual environment:<br>
> NOTE: For Macs the preferred command is ```source bin/activate```
```
.\Scripts\activate
```

### Step 3 - Run Program
```
python gui.py
```
* NOTE: Only systems running Windows with an Nvidia GPU will display metrics in the GUI.  Working on a fix.

### Step 4 - "Download Embedding Model"
The efficacy of an embedding model depends on both the type of text and type of questions you intend to ask.  Do some research on the different models in my program, but I've selected ones that are overall good.  Experiment with different ones.
> You must wait until the download is complete AND unpacked before trying to create the database.

### Step 5 - "Select Embedding Model Directory"
Selects the directory of the model you want to use.

### Step 6 - "Choose Documents for Database"
Select one or more files (pdf, docx, txt, json, enex, eml, msg, csv, xls, xlsx).

### Step 7 - "Create Vector Database."
GPU usage will spike as the vector database is created.  Wait for this to complete before querying database.

### Step 8 - LM Studio
Open LM Studio and load a model.  Click the server tab on the lefhand side.  Click "Start Server" in the server tab.
> Only Llama2-based models are currently supported due to their prompt format.

### Step 9 - "Submit Question"
Enter a question and click "submit question."  The vector database will be queried and your question along with the results will be fed to LM Studio for an answer.
> If you're curious, within the repository folder you'll find a file named "relevant_context.txt," which shows you exactly what the vector database produced.  This is useful to test different embedding models.

### Step 10 - Transcribe Question Instead
Click start record button.  Talk.  Click stop button.  Paste transcription into question box.  Click Submit Question.

</details>

# Contact

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
