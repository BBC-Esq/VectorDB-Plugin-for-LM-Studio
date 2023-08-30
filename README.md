<a name="top"></a>

# **The most recent version's instructions are here (v1.1).  If you have an older version, find the appropriate instructions in the User Manual folder.**

# ChromaDB Plugin for LM Studio

The ChromaDB Plugin for [LM Studio](https://lmstudio.ai/) adds a vector database to LM Studio utilizing ChromaDB! Tested on a 1000 page legal treatise

**COMPATIBLE with [Python 3.10.](https://www.python.org/downloads/release/python-31011/)**

**Link to [Medium article](https://medium.com/@vici0549/chromadb-plugin-for-lm-studio-5b3e2097154f)**

## Table of Contents
1. [Installation Instructions](#installation-instructions)
2. [Usage Guide](#usage-guide)
3. [Important Notes](#important-notes)
4. [Feedback](#feedback)
5. [Final Notes](#final-notes)
6. [Clearing Database](#clearing-database)

## Installation Instructions
* **Step 1**: Download all the files in this repository and put them into a directory.
* **Step 2**: Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) if it's not already installed.
* **Step 3**: Go to the folder where my repository is located, open a command prompt, and create a virtual environment by running:
```bash
python -m venv .
```
* **Step 4**: Activate the virtual environment:
```bash
.\Scripts\activate
```
* **Step 5**: Make sure ["pip"](https://pip.pypa.io/en/stable/index.html) is updated:
```bash
python -m pip install --upgrade pip
```
* **Step 6**: Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
* **Step 7**: Install the rest of the dependencies in the [requirements.txt](https://github.com/MicrosoftDocs/visualstudio-docs/blob/main/docs/python/managing-required-packages-with-requirements-txt.md):
```bash
pip install -r requirements.txt
```
* **Step 8**: You MUST the appropriate version of [Git](https://git-scm.com/downloads) if it's not already installed.

[Back to top](#top)

## Usage Guide
* **Step 1**: In the same command prompt, run:
```bash
python gui.py
```
* **Step 2**: Click "Download Embedding Model" and download a model.  The GUI will hang.  Wait, then proceed to next step.
  * **Note**: Only PDFs with OCR done on them will work as of Version 1. A folder named "Docs_to_DB" will be created and populated.
* **Step 3**: Click "Select Embedding Model Directory" and select the directory of the model you want to use.
* **Step 4**: Click "Choose Documents for Database" and choose one or more PDF files to put into the database.
  * **Note**: The PDFs must have had OCR done on them or have text in them already. Additional file types will be added in thefuture.
* **Step 5**: Click "Create Vector Database.  The GUI will hang.  Watch "CUDA" usage and wait.  When CUDA drops to zero, proceed to the next step.
* **Step 6**: Open up LM Studio and load a model.  Remember, only Llama2-based models work with the vector database currently.
* **Step 7**: Click "Start Server" within LM Studio, enter your question, and click "Submit Question."
  * **Note**: It's really cool to watch the LM Studio window showing the embedding model feeding the LLM in LM Studio!"

[Back to top](#top)

## Important Notes
* **Compatibility**: This is a personal project and was specifically tested using CUDA 11.8 and the related PyTorch installation.
* **Embedding Model**: This plugin uses "hkunlp/instructor-large" as the embedding model. Look [here](https://huggingface.co/spaces/mteb/leaderboard) for more details. If people express an interest, I'll likely include other embedding models in future versions!

[Back to top](#top)

## Feedback
My motivation to improve this beyond what I personally use it for is directly related to people's interest and suggestions. All feedback, positive and negative, is welcome! I can be reached at the LM Studio discord server or "bbc@chintellalaw.com".

[Back to top](#top)

## Final Notes
* **Note**: I only tested this on Windows 10 but can possibly expand on this in later versions.
* **Note**: Please be aware that when you click "Create Database" as well as "Submit Query" the GUI will hang. Just wait...it'll resume. This is a minor characteristic of the scripts that can easily be fixed in future versions.
* **Note**: Everytime you want to use the program again, enter the folder, activate the virtual enviroment using `.\Scripts\activate` and run `python gui.py`.

## Clearing Database
* **Simply delete all the files in the "Docs_to_DB" and "Vector_DB" folders.**

[Back to top](#top)

![Example Image](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example.png)
