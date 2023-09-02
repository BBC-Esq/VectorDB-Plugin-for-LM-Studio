<a name="top"></a>



# Adds a chromadb vector database to [LM Studio](https://lmstudio.ai/)!

**COMPATIBLE with [Python 3.10.](https://www.python.org/downloads/release/python-31011/)**

**Link to [Medium article](https://medium.com/@vici0549/chromadb-plugin-for-lm-studio-5b3e2097154f)**

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage-guide)
3. [Feedback](#feedback)
4. [Final Notes](#final-notes)
5. [Clearing Database](#clearing-database)

## Installation
* **Step 1**: Download all the files in this repository and put them into a directory.
* **Step 2**: Install [CUDA 11.8](https://developer.nvidia.com/cuda-toolkit-archive) if it's not already installed.
* **Step 3**: In the folder where the files located, open a command prompt, and create a virtual environment by running: ```python -m venv .```
* **Step 4**: Run to activate virtual environment: ```.\Scripts\activate```
* **Step 5**: Run to update ["pip"](https://pip.pypa.io/en/stable/index.html): ```python -m pip install --upgrade pip```
* **Step 6**: Run to install PyTorch with CUDA support: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
* **Step 7**: Install the rest of the dependencies in the [requirements.txt](https://github.com/MicrosoftDocs/visualstudio-docs/blob/main/docs/python/managing-required-packages-with-requirements-txt.md): ```pip install -r requirements.txt```
* **Step 8**: You MUST install the appropriate version of [Git](https://git-scm.com/downloads) if it's not already installed.

[Back to top](#top)

## Usage Guide
* **Step 1**: In the same command prompt, run:```python gui.py```
* **Step 2**: Click "Download Embedding Model" and download a model.  The GUI will hang.  Wait, then proceed to next step.
  * **Note**: Git clone is used to download.  Feel free to message me if you're wondering why I didn't use the normal "cache" folder method.
* **Step 3**: Click "Select Embedding Model Directory" and select the directory containing the model you want to use.
* **Step 4**: Click "Choose Documents for Database" and choose one or more PDF files to put into the database.
  * **Note**: The PDFs must have had OCR done on them or have text in them already. Additional file types will be added in thefuture.
* **Step 5**: Click "Create Vector Database.  The GUI will hang.  Watch "CUDA" usage.  When CUDA drops to zero, proceed to the next step.
* **Step 6**: Open up LM Studio and load a model (remember, only Llama2-based models currently work with the vector database).
* **Step 7**: Click "Start Server" within LM Studio, enter your question, and click "Submit Question."
  * **Note**: It's really cool to watch the LM Studio window showing the embedding model feeding the LLM in LM Studio!"

[Back to top](#top)

## Feedback
My motivation to improve this beyond what I personally use it for is directly related to people's interest and suggestions. All feedback, positive and negative, is welcome! I can be reached at the LM Studio discord server or "bbc@chintellalaw.com".

[Back to top](#top)

## Final Notes
* **Note**: I only tested this on Windows 10 but can possibly expand on this in later versions.
* **Note**: Everytime you want to use the program again, enter the folder, activate the virtual enviroment using `.\Scripts\activate` and run `python gui.py`.

## Clearing Database
* **Delete all the files in the "Vector_DB" folder or, it automatically clears everytime you create a new database.**

[Back to top](#top)

![Example Image](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png)
