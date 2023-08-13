# ChromaDB Plugin for LM Studio

The ChromaDB Plugin for LM Studio adds a vector database to LM Studio utilizing ChromaDB!

## Table of Contents
1. [Installation Instructions](#installation-instructions)
2. [Usage Guide](#usage-guide)
3. [Important Notes](#important-notes)
4. [Compatibility and Feedback](#compatibility-and-feedback)
5. [Final Note](#final-note)

## Installation Instructions
* **Step 1**: Download the `.exe` and run it.
  * **NOTE**: You must have [CUDA 11.8 already installed](https://developer.nvidia.com/cuda-11-8-0-download-archive)
  * **NOTE**: Before running the .exe, you must run the following command:
  * `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Usage Guide
* **Step 2**: Choose a document to ingest into your database (only PDFs with OCR already done on them as of Version 1). A folder named "Docs_to_DB" will be created and populated if it doesn't already exist.
* **Step 3**: Click the "Create ChromaDB" button. Another folder named "Vector_DB" will be created (if it doesn't already exist) to store the vector database.  If it's the first time runnin the program, it'll create a "Model" folder and download the actual embedding model as well.
  * **Warning**: A message will appear prior to this. It's recommended to follow the warning and monitor your CUDA usage when running the program for the first time to verify compatibility.
* **Step 4**: After the vector database is created, your CUDA usage will drop to zero. Before typing in a query and clicking "Submit Query," you **MUST FIRST** open LM Studio, select a model, and click "Start Server."
* **Step 5**: Type your query and click "Submit Query" for an amazing response.  ALSO, view the LM Studio log while it's running to see how interestingy the vector database interacts with the LLM!

## Important Notes
* **Compatibility**: This is a personal project and will work best with a Llama2 "chat" model (e.g. 7B or 13b) due to varying prompting styles. Other models like Orca mini (3b) may work sometimes.
* **Embedding Model**: The plugin uses "hkunlp/instructor-large." More details can be found [here](https://huggingface.co/spaces/mteb/leaderboard). Interested in other embedding model options? Please provide feedback.

## Compatibility and Feedback
Your feedback is essential. If you'd like to see compatibility with other models or have any other requests, please reach out. The community's input drives this project, so don't hesitate to share your thoughts via the LM Studio discord.

## Final Notes
Please be aware that when you click "Create Database" the GUI will hanguntil the database is created and it will also hang when you submit a query until a response is received. These issues will be addressed in future versions. Also, I'm leaving the command prompt showing so people can see any error messages to troubleshoot and give me feedback as well.  Futher versions will likely omit this.
![ChromaDB Plugin Example](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/blob/main/example.png)
