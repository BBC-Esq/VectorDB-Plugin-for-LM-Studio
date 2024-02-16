<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!  Now with Vision Models!</h1>
  <h3>Ask questions about your documents and get an answer from LM Studio!<br><a href="https://www.youtube.com/watch?v=KXYH8zqN8c8">Introductory Video</a><br><a href="https://medium.com/@vici0549/search-images-with-vector-database-retrieval-augmented-generation-rag-3d5a48881de5">Medium Article</a></h3>
</div>

<div align="center">
<table>
  <thead>
    <tr>
      <th></th>
      <th>Database</th>
      <th>Transcribe</th>
      <th>Bark/TTS</th>
      <th>Vision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Intel CPU</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>AMD CPU</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Nvidia GPU</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>AMD GPU on Windows</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>AMD GPU on Linux</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Apple CPU</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>Apple Metal/MPS</td>
      <td>✅</td>
      <td>✅</td>
      <td>❓</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>Apple MLX</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>Direct ML</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>Vulkan</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
  </tbody>
</table>
</div>

<div align="center"> <h2><u>REQUIREMENTS</h2></div>

<details>
  <summary>REQUIREMENTS</summary>
  
1) 🐍[Python 3.10](https://www.python.org/downloads/release/python-31011/) or [Python 3.11](https://www.python.org/downloads/release/python-3117/) (Python 3.12 coming soon).
2) 📁[Git](https://git-scm.com/downloads)
3) 📁[Git Large File Storage](https://git-lfs.com/).
4) 🌐[Pandoc](https://github.com/jgm/pandoc/releases).
5) Build Tools.
   > Certain dependencies don't have pre-compiled "wheels" so you must build them.  Therefore, you must install something that can build source code such as [Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and/or [Visual Studio](https://visualstudio.microsoft.com/).  If you decide to use both of these programs in conjunction, make sure to select the "Desktop development with C++" extension and check the four boxes on the right containing "SDK."  Most Linux systems as well as MacOS come with the ability to build.  If you still run into problems on those platforms; however, you should find something that can build.

   <details>
     <summary>EXAMPLE ERROR ON WINDOWS</summary>
     <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/sample_error.png">
   </details>

   <details>
     <summary>EXAMPLE SOLUTION ON WINDOWS</summary>
     <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/build_tools.png">
   </details>

6) 🍎MacOS Only.  [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).

6) 🟢Nvidia GPU acceleration (Windows or Linux) requires [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (CUDA 12+ coming soon).
7) 🔴AMD GPU acceleration on Linux requires [ROCm 5.6](https://docs.amd.com/en/docs-5.6.0/deploy/windows/gui/index.html) (ROCm 5.7 coming soon).

   > PyTorch does not support AMD GPUs on Windows yet.

</details>

<div align="center"> <h2>INSTALLATION</h2></div>

<details>
  <summary>🪟WINDOWS INSTRUCTIONS</summary>
  
### Step 1
🟢 Nvidia GPU ➜ [Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
> CUDA 12+ support is coming as soon as the faster-whisper library supports it.<br>

🔴 AMD GPU - PyTorch currently does not support AMD gpu-acceleration on Windows. There are several unofficial workarounds but I'm unable to verify since I don't have an AMD GPU nor use Linux. See [HERE](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html), [HERE](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview), [HERE](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview), and possibly [HERE](https://user-images.githubusercontent.com/108230321/275660295-e2d6e097-38c5-4e38-9a1f-f28441ba8812.png).
### Step 2
Navigate to a directory on your computer, open a command prompt and run:
```
git clone https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio.git
```
  * Alternatively, you can [download the latest release](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/latest), open the ZIP file, and copy the contents to a folder on your computer.
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
  <summary>🐧LINUX INSTRUCTIONS</summary>

### Step 1
🟢 Nvidia GPUs ➜ Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)<br>
🔴 AMD GPUs ➜ Install [ROCm version 5.6](https://docs.amd.com/en/docs-5.6.0/deploy/windows/gui/index.html).
> [THIS REPO](https://github.com/nktice/AMD-AI) also has instructions.
> Also, although I'm unable to test on my system...[here are some "wheels"](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/tag/rocm) that I believe should work.  However, you'd have to search and find the right one for your system.
### Step 2
Navigate to a directory on your computer, open a command prompt and run:
```
git clone https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio.git
```
  * Alternatively, you can [download the latest release](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/latest), open the ZIP file, and copy the contents to a folder on your computer.
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
python setup_linux.py
```
### Optional Step 6
Run this script if you want to doublecheck wherher you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```
</details>

<details>
  <summary>🍎APPLE INSTRUCTIONS</summary>

### Step 1
```
brew install portaudio
```
* This requires Homebrew to be installed first.  If it's not, run the following command before running ```brew install portaudio```:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
### Step 2
For Pytorch to use 🔘Metal/MPS it requires MacOS 12.3+.  Metal/MPS provides gpu-acceleration similiar to CUDA (for NVIDIA gpus) and rocM (for AMD gpus).
### Step 3
Navigate to a directory on your computer, open a command prompt and run:
```
git clone https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio.git
```
  * Alternatively, you can [download the latest release](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/latest), open the ZIP file, and copy the contents to a folder on your computer.
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
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```
### Step 8
```
pip install -r requirements.txt
```
### Step 9
Upgrade PDF loader by running:
```
python replace_pdf.py
```
### Optional Step 10
Run this script if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```
</details>

<div align="center"> <h2>USING THE PROGRAM</h2></div>
<details>
  <summary>🖥️INSTRUCTIONS🖥</summary>

## Activate Virtual Environment
* Once you install the program you've already created a virtual environment, so you just need to activate it each time you want to restart it.  Remember to run the appropriate command to do so (based on your platform) within the ```src``` folder.
## Start the Program
```
python gui.py
```
> Only systems with an Nvidia GPU will display gpu power, usage, and VRAM metrics.

# 🔥Important🔥
* Read the User Guide before sending me questions.

## Download Vector Model
* In the ```Models Tab``` tab, choose the embedding model you want to download.  The ```User Guide Tab``` explains the difference characteristics of the various models.

## Set Vector Model
* In the ```Databases Tab```, click ```Choose Model``` and click once on the directory containing the model you want to use and click ```Select Folder``` in the lower right.
  > 🔥 Do not select the ```Embedding_Models``` folder itself.

## Set Chunk Size and Overlap
* In the ```Settings Tab```, set the chunk size and chunk overlap.
  > 🔥 Anytime you want to change these two settings you must re-create the database for the changes to take effect.

## Add Files
* In the ```Databases Tab```, click the ```Choose Files``` and select one or more files.  This can be repeated multiple times for files located in different directories.
  * * Supported "document" files are: ```.pdf```, ```.docx```, ```.epub```, ```.txt```, ```.html```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```.
  * * Supported "image" files are: ```.png```, ```.jpg```, ```.jpeg```, ```.bmp```, ```.gif```, ```.tif```, ```.tiff```
* To add "audio" files you must go to the ```Tools Tab``` and transcribe an audio file.  This process can be repeated for multiple audio files, however.  The transcription(s) will automatically be saved to the appropriate folder to be added when you create the vector database.
  * * Most "audio" files should be supported: ```.mp3```, ```.wav```, ```.m4a```, ```.ogg```, ```.wma```

## Removing Files
* In the ```Databases Tab```, select one or more files, right click, and delete.  Re-create the database.

## Creating the Databaase
* Click the ```Create Vector Database``` button.  Wait until the command prompt says "persisted" before proceeding to the next step.

## Connecting to LM Studio
* Start LM Studio and load a model.

## Choosing a Prompt Format
The LLM within LM Studio works best with an appropriate "prompt format."  In the ```Settings Tab```, choose the appropriate prompt format matching the model being used within LM Studio.  You can also enter one manually if a preset is not available.  However, you must turn the ```automatic prompt formatting``` setting in LM Studio to ```off```.

Morever, a bug was introduced in ```LM Studio v0.2.10``` that I have been unable to verify is resolved; therefore, you must additionally:
* ⚠️ Delete any/all text within the ```User Message Prefix``` box; and
* ⚠️ Delete any/all text within the ```User Message Suffix``` box.

## Start the LM Studio Server
* In LM Studio,  click ```Start Server.```

## Search Database
* Type (or speak) your question and click ```Submit Question.```

## Test Chunks
* If you wish to test the quality of the chunk settings, check the ```Chunks Only``` checkbox.  The program will no longer connect to LM Studio and will instead provide you with the chunks directly from the vector database.

## Text to Voice
* This program uses fun "Bark" models to convert the response to audio.  However, you must wait until the ENTIRE response is received before clicking the ```Bark Response``` button.

## Voice to Text:
* The voice recorder and audio file transcriber use the ```faster-whisper``` library (CUDA 12+ coming soon).

## Image to Text
As of release 3.0, the program includes exciting "vision" models that generate summaries of one or more pictures, which are then added to the vector database.  I wrote a [Medium article](https://medium.com/@vici0549/search-images-with-vector-database-retrieval-augmented-generation-rag-3d5a48881de5) on this as well.
  > Remember, the ```Tools Tab``` allows you to test the vision model settings on a single image before creating the database and spending a lot of time creating captions.

</details>

<div align="center"><h2>CONTACT</h2></div>

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example1.png" alt="Example Image">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example2.png" alt="Example Image">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example3.png" alt="Example Image">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example4.png" alt="Example Image">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example5.png" alt="Example Image">
</div>
