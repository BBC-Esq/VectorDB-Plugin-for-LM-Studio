<a name="top"></a>

# **URGENT - The readme has been updated as of August 13, 2023 at 9:10 p.m.  Apparently, the prior instructions instaled requirements first, which overrode some of the packages that PyTorch + CUDA needed...so follow the new instructions.**

# ChromaDB Plugin for LM Studio

The ChromaDB Plugin for [LM Studio](https://lmstudio.ai/) adds a vector database to LM Studio utilizing ChromaDB! Tested on a 1000 page legal treatise

**COMPATIBLE with [Python 3.10.](https://www.python.org/downloads/release/python-31011/)**

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
* **Step 3**: Go to the folder where my repository is located, open a command prompt and run:
```bash
python -m venv .
```
* **Step 4**: Then run:
```bash
.\Scripts\activate
```
* **Step 5**: Then run:
```bash
python -m pip install --upgrade pip
```
* **Step 6**: Then run:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
* **Step 7**: Then run:
```bash
pip install -r requirements.txt
```
[Back to top](#top)

## Usage Guide
* **Step 1**: In the same command prompt run:
```bash
python gui.py
```
* **Step 2**: Click the "Choose Documents" button and choose one or more documents to include in the vector database.
  * **Note**: Only PDFs with OCR done on them will work as of Version 1. A folder named "Docs_to_DB" will be created and populated.
* **Step 3**: Click the "Create ChromaDB" button. A folder named "Vector_DB" will be created if it doesn't already exist and the DB will be created in there (see notes below).
  * **Note**: A message will appear with instructions on how to monitor CUDA usage. Please follow them.
  * **Note**: The embedding model will be downloaded to your cache folder if it's not downloaded already. Once downloaded, the vector database will automatically be created. Watch your CUDA usage to verify that it's working - pretty awesome! The database is fully-created when CUDA usage drops to zero.
* **Step 4**: Open LM Studio, select a model, and click "Start Server."
  * **Note**: The formatting of the prompt in my scripts is specifically geared to work with any Llama2 "chat" models. Any others might not work if they provide an intelligible response at all.  This can be addressed in future versions.
* **Step 5**: Enter your query and click the "Submit Query" button and be amazed at the response you get.
  * **Note**: If will give an error if you don't start the server before clicking "Submit Query."
  * **Note**: For [extra entertainment](https://www.youtube.com/watch?v=5IsSpAOD6K8), watch LM Studio server's log window to watch it interact with the vector database!

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
