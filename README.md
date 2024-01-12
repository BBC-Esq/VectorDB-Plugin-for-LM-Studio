<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!  Now with Vision Models!</h1>
  <h3>Ask questions about your documents and get an answer from LM Studio!<br>(https://www.youtube.com/watch?v=KXYH8zqN8c8)</h3>
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
4) [Pandoc](https://github.com/jgm/pandoc).

<div align="center"> <h2>INSTALLATION</h2></div>

<details>
  <summary>🪟WINDOWS INSTRUCTIONS🪟</summary>
  
### Step 1
🟢 Nvidia GPU ➜ [Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
> CUDA 12+ is currently NOT compatible since the faster-whisper library is only compatible up to CUDA 11.8, but it will be soon!<br>

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
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```
🔴 AMD GPU:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```
```
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```
🔵 CPU only:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
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
pip3 install -r requirements.txt
```
### Step 10
You must run:
```
python replace_pdf.py
```

### Step 11
```
sudo apt -y install libxcb-cursor0
```

### Optional Step 12
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
pip3 install torch torchvision torchaudio
```
* And if that fails OR YOU GET CUDA-RELATED ERRORS when creating the database (e.g. if using M1, which is ARM-based), run:
```
pip uninstall torch torchvision torchaudio
```
The reinstall using this:
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html
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
### Step 10
You must run:
```
python replace_pdf.py
```
### Optional Step 11
Run this script if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```
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
* Within the ```Vector Models tab```, choose the embedding model you want to download.

## Set Model Directory
* Within the ```Databases Tab```, choose the directory containing the embedding model you want to use from among the ones you've already downloaded.
  > Do not choose the ```Embedding_Models``` itself.

## Adding Documents, Images, and Audio to the Database
* Click the ```Choose Documents or Images``` button to add image or non-image files.
  * * Supported non-image extensions are: ```.pdf```, ```.docx```, ```.epub```, ```.txt```, ```.html```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```.
  * * Supported image extensions are: ```.png```, ```.jpg```, ```.jpeg```, ```.bmp```, ```.gif```, ```.tif```, ```.tiff```
* In the ```Tools Tab``` you can transcribe an audio file into ```.txt``` to be put into the vector databse if you want.
    > Remember to test the vision model settings within the Tools Tab first.
* ⚠️ Anytime you add/remove documents you must recreate the vector database.

## Removing Documents
* Within the ```Databases Tab``` you can select or more files, right click, and delete.

## Creating the Databaase
* Clicks the ```Create Vector Database``` button.  Wait until the command prompt says "persisted" before proceeding to the next step.

## Connecting to LM Studio
* Start LM Studio and load a model.

## Choosing a Prompt Format
The LLM within LM Studio works best with the appropriate "prompt format."  Within the ```Settings Tab``` choose the appropriate prompt format or enter one manually.  However, you must first disable "automatic prompt formatting" within LM Studio; there is a toggle to do this.
  > You don't need to do this if you're using ```LM Studio v0.2.9``` or earlier.
Additionally, within ```LM Studio v0.2.10``` there is a known bug preventing LM Studio from respecting the prompt format you choose.  Therefore, within LM Studio, you must also go to the Server settings (far right side) and:
* ⚠️ Delete any/all text within the ```User Message Prefix``` box; and
* ⚠️ Delete any/all text within the ```User Message Suffix``` box.

## Start the LM Studio Server
* Within LM Studio, open the server tab on the left side and click ```Start Server.```

## Search Database
* Type (or speak) your question and click ```Submit Questions.```

## Test Chunks
* If you wish to test the quality of the chunk settings check the ```Chunks Only``` checkbox.  LM Studio will not be connected to and you'll simply receive the relevant contexts from the vector database.
* This is also good if you want to obtain more results than the context window of the LLM can handle.  For example, if you want to obtain 100 results from the vector database, which would exceed the LLM's context window of 4096.

## Test to Voice
* This program uses Bark models to convert the response from LM Studio to audio.  You must wait until the ENTIRE response is received from the LLM and then click the ```Bark Response``` button.

## Voice to Text
* Both the voice recorder and audio file transcriber use the ```faster-whisper``` library and GPU acceleration is as follows:

  > Note, ```faster-whisper``` only supports CUDA 11.8 currently, but CUDA 12+ support is coming in the near future.

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

</details>

## NEW and Exciting Vision Models
As of release 3.0 the program includes Vision Models to process descriptions of images that are added to the vector database and can ber searched.  For example, "Find me all pictures that depict one or more people doing X."

<div align="center"><h2>CONTACT</h2></div>

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
