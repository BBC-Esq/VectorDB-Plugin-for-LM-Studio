<a name="top"></a>

<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a>! with a Vector Database!</h1>
</div>

<div align="center">
  <strong>Tested with <a href="https://www.python.org/downloads/release/python-31011/">Python 3.10</a></strong>
</div>

<div align="center">
  <a href="https://medium.com/@vici0549/chromadb-plugin-for-lm-studio-5b3e2097154f">📖 Link to Medium article</a>
</div>

<!-- GPU Acceleration Support Table -->

<div align="center">
  <h2>⚡ GPU Acceleration Support⚡</h2>
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
        <td>CUDA 11.8 or 11.7</td>
      </tr>
      <tr>
        <td>AMD</td>
        <td>❌</td>
        <td>✅</td>
        <td>ROCm 5.4.2</td>
      </tr>
      <tr>
        <td>Apple/Metal</td>
        <td colspan="3" align="center"> ✅ ("Xcode Command Line Tools")</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- Table of Contents -->
<div align="center"> (PyTorch does not support ROCm on Windows systems)</div>
<div align="center">
  <h2>Table of Contents</h2>
</div>

<div align="center">
  <a href="#installation">🛠️ Installation</a> | 
  <a href="#usage">🔍 Usage</a> | 
  <a href="#contact">💌 Contact Me</a>
</div>

## Installation

* **Step 1** ➜ Install the appropriate software if you intend to use GPU-acceleration:
  * **For NVIDIA GPUs** ➜ install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) for your specific operating system.
  * **For AMD GPUs** ➜ install ROCm version 5.4.2; instructions are [HERE](https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html) and [HERE](https://rocmdocs.amd.com/en/latest/deploy/linux/index.html).
    * Unfortunately, gpu-accleration will [only work on Linux systems](https://github.com/RadeonOpenCompute/ROCm/blob/develop/docs/rocm.md).  
  * **For 🍎 Apple/Metal/MPS** ➜ install [Xcode Command Line Tools](https://www.makeuseof.com/install-xcode-command-line-tools/).
* **Step 2** ➜ Download or clone this repository to a directory on your computer.
* **Step 3** ➜ Open a command prompt from within the directory and create a virtual environment:
```
python -m venv .
```
* **Step 4** ➜ Activate the virtual environment:
```
.\Scripts\activate
```
* **Step 5** ➜ Update [PIP](https://pip.pypa.io/en/stable/index.html):
```
python -m pip install --upgrade pip
```
* **Step 6** ➜ Install PyTorch with the appropriate "build:"

  * **🪟Windows + CUDA 11.8** ➜ ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
  * **🪟Windows + CUDA 11.7** ➜ ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117```
  * **🪟Windows + CPU-only** ➜ ```pip install torch torchvision torchaudio```
  * **🐧Linux + CUDA 11.8** ➜ ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
  * **🐧Linux + CUDA 11.7** ➜ ```pip install torch torchvision torchaudio```
  * **🐧Linux + ROCm 5.4.2** ➜ ```install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2```
  * **🐧Linux + CPU-only** ➜ ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```
  * **🍎Apple + Metal** ➜ ```pip install torch torchvision torchaudio```
    * **Metal/MPS** [speedup comparison](https://explosion.ai/blog/metal-performance-shaders) to a 5950x and RTX 3090.

* **Step 7** ➜ Doublecheck that you installed gpu-acceleration properly:
```
python check_gpu.py
```

* **Step 8** ➜ Install the dependencies listed in [requirements.txt](https://github.com/MicrosoftDocs/visualstudio-docs/blob/main/docs/python/managing-required-packages-with-requirements-txt.md):
```
pip install -r requirements.txt
```
* **Step 9** ➜ Lastly, you must install the appropriate version of Git (https://git-scm.com/downloads).

[Back to top](#top)

## Usage

* **Step 1**: Open a command prompt in the directory of my scripts, activate the virtual environment, and run:
```
python gui.py
```
* **Step 2** ➜ "Download Embedding Model." Must wait until the download is complete and unpacked before creating the vector database.
* **Step 3** ➜ "Select Embedding Model Directory" so select the directory containing the model you want to use.
* **Step 4** ➜ "Choose Documents for Database" to choose one or more files to put in the database.
  * Current supported file extensions are: pdf, docx, txt, json, enex, eml, msg, csv, xls, xlsx.
* **Step 5** ➜ "Create Vector Database." When CUDA drops to zero, proceed to the next step.
* **Step 6** ➜ Open LM Studio and load a model (only Llama2-based models currently work).
* **Step 7** ➜ Click "Start Server" within LM Studio, enter your question, and click "Submit Question."

[Back to top](#top)

## Contact

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

[Back to top](#top)

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
