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
        <td>Yes (requires CUDA 11.7 or 11.8)</td>
        <td>Yes (requires CUDA 11.7 or 11.8)</td>
      </tr>
      <tr>
        <td>AMD</td>
        <td>No</td>
        <td>Yes (requires ROCm 5.4.2)</td>
      </tr>
      <tr>
        <td>Apple</td>
        <td colspan="2" align="center">n/a</td>
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
  <a href="#feedback">Feedback</a> | 
  <a href="#final-notes">Final Notes</a>
</div>

## Installation

* **Step 1**: If not already installed, install the appropriate "AI Framework:"
  * **For NVIDIA GPUs** install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) for your specific operating system.
  * **For AMD GPUs**: Unfortunately, gpu-accleration using PyTorch is [only available on Linux systems](https://github.com/RadeonOpenCompute/ROCm/blob/develop/docs/rocm.md).  If you use Linux, you must install [ROCm](https://en.wikipedia.org/wiki/ROCm) version 5.4.2.  Installation instructions are [here](https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html) and [here](https://rocmdocs.amd.com/en/latest/deploy/linux/index.html).  I do not have an AMD GPU so I can't verify the installation instructions.
* **Step 2**: Download or clone this repository to a directory on your computer.
* **Step 3**: Open a command prompt from within the directory and create a virtual environment:
```
python -m venv .
```
* **Step 4**: Activate the virtual environment:
```
.\Scripts\activate
```
* **Step 5**: Update ["pip"](https://pip.pypa.io/en/stable/index.html):
```
python -m pip install --upgrade pip
```
* **Step 6**: Install PyTorch with the appropriate Operating/AI Framework that you installed previously:

  * **Windows/CUDA 11.8:** ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
  * **Windows/CUDA 11.7:** ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117```
  * **Linux/CUDA 11.8:** ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
  * **Linux/CUDA 11.7:** ```pip install torch torchvision torchaudio```
  * **Linux/ROCm 5.4.2:** ```install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2```
* **Step 7**: Install the dependencies listed in [requirements.txt](https://github.com/MicrosoftDocs/visualstudio-docs/blob/main/docs/python/managing-required-packages-with-requirements-txt.md):
```
pip install -r requirements.txt
```
* **Step 8**:
  * Lastly, you must install the appropriate version of Git (https://git-scm.com/downloads).

[Back to top](#top)

## Usage Guide

* **Step 1**: In the same command prompt, run:```python gui.py```
* **Step 2**: Click "Download Embedding Model" and download a model. The GUI will hang. Wait, then proceed to the next step.
  * **Note**: Git clone is used to download. Feel free to message me if you're wondering why I didn't use the normal "cache" folder method.
* **Step 3**: Click "Select Embedding Model Directory" and select the directory containing the model you want to use.
* **Step 4**: Click "Choose Documents for Database" and choose one or more PDF files to put into the database.
  * **Note**: The PDFs must have had OCR done on them or have text in them already. Additional file types will be added in the future.
* **Step 5**: Click "Create Vector Database." The GUI will hang. Watch "CUDA" usage. When CUDA drops to zero, proceed to the next step.
* **Step 6**: Open up LM Studio and load a model (remember, only Llama2-based models currently work with the vector database).
* **Step 7**: Click "Start Server" within LM Studio, enter your question, and click "Submit Question."
  * **Note**: It's really cool to watch the LM Studio window showing the embedding model feeding the LLM in LM Studio!"

[Back to top](#top)

## Feedback

My motivation to improve this beyond what I personally use it for is directly related to people's interest and suggestions. All feedback, positive and negative, is welcome! I can be reached at the LM Studio discord server or "bbc@chintellalaw.com".

[Back to top](#top)

## Final Notes

* **Note**: I only tested this on Windows 10 but can possibly expand on this in later versions.
* **Note**: Every time you want to use the program again, enter the folder, activate the virtual environment using `.\Scripts\activate` and run `python gui.py`.

[Back to top](#top)

<div align="center">
  <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png" alt="Example Image">
</div>
