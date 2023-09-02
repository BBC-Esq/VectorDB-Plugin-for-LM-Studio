<a name="top"></a>

<div align="center">
  <h1>Adds a chromadb vector database to <a href="https://lmstudio.ai/">LM Studio</a>!</h1>
</div>

<div align="center">
  <strong>Tested with <a href="https://www.python.org/downloads/release/python-31011/">Python 3.10.</a></strong>
</div>

<div align="center">
  <a href="https://medium.com/@vici0549/chromadb-plugin-for-lm-studio-5b3e2097154f">Link to Medium article</a>
</div>

<!-- GPU Acceleration Support Table -->

<div align="center">
  <h2>GPU Acceleration Support</h2>
  <table>
    <thead>
      <tr>
        <th>GPU Brand</th>
        <th>Windows Support</th>
        <th>Linux Support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Nvidia</td>
        <td>✅ (requires CUDA 11.7 or 11.8)</td>
        <td>✅ (requires CUDA 11.7 or 11.8)</td>
      </tr>
      <tr>
        <td>AMD</td>
        <td>❌</td>
        <td>✅ (requires ROCm 5.4.2)</td>
      </tr>
      <tr>
        <td>Apple/Metal</td>
        <td colspan="2" align="center"> ✅ (must install "Xcode Command Line Tools")</td>
      </tr>
    </tbody>
  </table>
</div>


<!-- Table of Contents -->

<div align="center">
  <h2>Table of Contents</h2>
</div>

<div align="center">
  <a href="#installation">Installation</a> | 
  <a href="#usage-guide">Usage</a> | 
  <a href="#contact">Contact Me</a>
</div>

## Installation

* **Step 1**: Install the appropriate framework if you intend to use GPU-acceleration:

  * **For NVIDIA GPUs** install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) for your specific operating system.
  * **For AMD GPUs**: Unfortunately, gpu-accleration using PyTorch is [only available on Linux systems](https://github.com/RadeonOpenCompute/ROCm/blob/develop/docs/rocm.md).  If you use Linux, you must install [ROCm](https://en.wikipedia.org/wiki/ROCm) version 5.4.2.  Instructions are [HERE](https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html) and [HERE](https://rocmdocs.amd.com/en/latest/deploy/linux/index.html).
  * **For Apple/Metal/MPS:**  You must install [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).
* **Step 2**: Download or clone this repository to a directory on your computer.
* **Step 3**: Open a command prompt from within the directory and create a virtual environment:
```
python -m venv .
```
* **Step 4**: Activate the virtual environment:
```
.\Scripts\activate
```
* **Step 5**: Update [PIP](https://pip.pypa.io/en/stable/index.html):
```
python -m pip install --upgrade pip
```
* **Step 6**: Install PyTorch with the appropriate "build:"

  * **Windows/CUDA 11.8:** ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
  * **Windows/CUDA 11.7:** ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117```
  * **Windows/CPU-only:** ```pip install torch torchvision torchaudio```
  * **Linux/CUDA 11.8:** ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
  * **Linux/CUDA 11.7:** ```pip install torch torchvision torchaudio```
  * **Linux/ROCm 5.4.2:** ```install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2```
  * **Linux/CPU-only:** ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```
  * **Apple/Metal/MPS:** ```pip pip install torch torchvision torchaudio```
    * **Metal/MPS** [speedup comparison](https://explosion.ai/blog/metal-performance-shaders)

* **Step 7**: Install the dependencies listed in [requirements.txt](https://github.com/MicrosoftDocs/visualstudio-docs/blob/main/docs/python/managing-required-packages-with-requirements-txt.md):
```
pip install -r requirements.txt
```
* **Step 8**:
  * Lastly, you must install the appropriate version of Git (https://git-scm.com/downloads).

[Back to top](#top)

## Usage Guide

* **Step 1**: Open a command prompt in the directory of my scripts, activate the virtual environment, and run:
```
python gui.py
```
* **Step 2**: Click "Download Embedding Model" and download a model. The GUI will hang. Wait, then proceed to the next step.
  * **Note**: Git clone is used to download. Feel free to message me if you're wondering why I didn't use the normal "cache" folder method.
* **Step 3**: Click "Select Embedding Model Directory" and select the directory containing the model you want to use.
* **Step 4**: Click "Choose Documents for Database" and choose one or more PDF files to put into the database.
  * **Note**: The PDFs must have had OCR done on them or have text in them already. Additional file types will be added in the future.
* **Step 5**: Click "Create Vector Database." The GUI will hang. Watch "CUDA" usage. When CUDA drops to zero, proceed to the next step.
* **Step 6**: Open up LM Studio and load a model (remember, only Llama2-based models currently work with the vector database).
* **Step 7**: Click "Start Server" within LM Studio, enter your question, and click "Submit Question."


[Back to top](#top)

## Contact

All suggestions (positive and negative) are welcome.  I can be reached at "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

[Back to top](#top)

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
