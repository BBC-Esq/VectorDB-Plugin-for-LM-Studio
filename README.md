<div align="center">
  <h1>🚀 Supercharge your <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!</h1>
  <h2>Now with the ability to process images and audio files, Local chat models, and text to speech playback!<br><a href="https://youtu.be/8-ZAYI4MvtA">Introductory Video</a><br><a href="https://medium.com/@vici0549/search-images-with-vector-database-retrieval-augmented-generation-rag-3d5a48881de5">Medium Article</a></h2>
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
      <td>❌</td>
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
      <td>❌</td>
      <td>❓</td>
      <td>❌</td>
    </tr>
  </tbody>
</table>
</div>

* 🔥This table is only current as of ```Release v3.5.2``` due to the fact that I don't currently have time to support all backends/platforms.  Any release after version 3.5.2 only supports the ```Windows``` + ```Nvidia GPU``` combination.  Hopefully more time will free up in the future to support more platforms and backends.

<div align="center"> <h2><u>REQUIREMENTS</h2></div>
  
1) 🐍[Python 3.11](https://www.python.org/downloads/release/python-3119/) (Pytorch is not compatible with Python 3.12 yet)
2) 📁[Git](https://git-scm.com/downloads)
3) 📁[Git Large File Storage](https://git-lfs.com/).
4) 🌐[Pandoc](https://github.com/jgm/pandoc/releases).
5) Build Tools.
   > Certain dependencies don't have pre-compiled "wheels" so you must build them with something like [Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and/or [Visual Studio](https://visualstudio.microsoft.com/).  I recommend Visual Studio, but make sure to select the "Desktop development with C++" extension and check the four boxes on the right containing "SDK."

   <details>
     <summary>EXAMPLE ERROR ON WINDOWS</summary>
     <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/sample_error.png">
   </details>

   <details>
     <summary>EXAMPLE SOLUTION ON WINDOWS</summary>
     <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/build_tools.png">
   </details>

<div align="center"> <h2>INSTALLATION</h2></div>

<details>
  <summary>🪟WINDOWS INSTRUCTIONS</summary>
  
### Step 1
Download the ZIP file for the latest "release," extract its contents, navigate to the "src" folder to run the commands below.
  * NOTE: If you clone this repository you WILL NOT get the latest release.  Instead, you will development versions of this program which may or may not be stable.
### Step 2
Navigate to the ```src``` folder, open a command prompt, and create a virtual environment:
```
python -m venv .
```
### Step 3
Activate the virtual environment:
```
.\Scripts\activate
```
### Step 4
Run setup:
```
python setup.py
```

### Optional Step 5
Run this command if you want to doublecheck that you installed the Pytorch and gpu-acceleration software correctly:
```
python check_gpu.py
```
</details>

<details>
  <summary>🐧LINUX INSTRUCTIONS</summary>

Linux users must use Release v3.5.2 until I can update the codebase due to recent major changes.  Download the ZIP file for that release and follow the instructions in the readme.md.

</details>

<details>
  <summary>🍎APPLE INSTRUCTIONS</summary>

MacOS users must use Release v3.5.2 until I can update the codebase due to recent major changes.  Download the ZIP file for that release and follow the instructions in the readme.md.

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
* Read the User Guide.

## Download Vector Model
* In the ```Models Tab``` tab, choose the embedding model you want to download.  The ```User Guide``` tab contains more details about the models.

## Create a Vector Database
* In the ```Create Database``` tab, click ```Choose Files``` and select one or more files to add.  This can be repeated as many times as you wish.
  * Supported documents are: ```.pdf```, ```.docx```, ```.epub```, ```.txt```, ```.html```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```.
* If you selected any image files, I highly recommend that you adjust the vision model settings within the ```Settings``` tab and test a particular vision model in the ```Tools``` tab.
  * Supported images are: ```.png```, ```.jpg```, ```.jpeg```, ```.bmp```, ```.gif```, ```.tif```, ```.tiff```
* 🔥 To add audio files you must first transcribe them from the ```Tools Tab```.  The transcriptions will be saved and added when you create the vector database.
  * Supported audio extensions include, but are not limited to: ```.mp3```, ```.wav```, ```.m4a```, ```.ogg```, ```.wma```
* Select a vector model from the pulldown menu.
* Enter a name for the database you want to create.
* In the ```Settings``` tab, set the chunk size, chunk overlap, and the device you want to use.  More information is in the User Guide.
* Click the ```Create Vector Database``` button.
  * 🔥 MAKE SURE to wait until the command prompt states that the database has been successfully created before proceeding.

## Delete a Database
* In the ```Manage Databases``` tab, select a database from the pulldown menu and click ```Delete Database.```
  > The ability to delete one or more specific files is coming soon.

## Query a Database (No LM Studio)
* In the ```Query Database``` tab, select the database you want to use from the pulldown menu.
* Enter your question by typing it or using the ```Record Question``` button.
* Check the ```chunks only``` checkbox to only receive the relevant contexts.
* Click ```Submit Question```.
  * In the ```Settings``` tab, you can change multiple settings regarding querying the database.  More information can be found in the User Guide.

## Query a Database with a Response From LM Studio
This program gets relevant chunks from the vector database and forwarding them - along with your question - to LM Studio for an answer!
* Perform the above steps regarding entering a question and choosing settings, but make sure that ```Chunks Only``` is 🔥UNCHECKED🔥.
* Start LM Studio and go to the Server tab on the left.
* Load a model.  I've personally tested the following models as good:
  * ```Marx_3B_V3```
  * ```Mistral_7B_Instruct_v0_2```
  * ```Neural_Chat_7b_v3_3```
  * ```Llama_2_13b_chat_hf```
  * ```SOLAR_10_7B_Instruct_v1_0```
* Turn ```Apply Prompt Formatting``` to "OFF."
* On the right side within ```Prompt Format```, make sure that all of the following settings are blank:
  * ```System Message Prefix```
  * ```System Message Suffix```
  * ```User Message Prefix```
  * ```User Message Suffix```
* At the top, load a model within LM Studio.
* On the right, adjust the ```GPU Offload``` setting to your liking.
* Within my program, go to the ```Settings``` tab, select the appropriate prompt format for the model loaded in LM Studio, click ```Update Settings```.
* In LM Studio,  click ```Start Server.```
* In the ```Query Database``` tab, click ```Submit Question```.

</details>

## Request a Feature or Report a Bug

Feel free to report bugs or request enhancements by creating an issue on github or contacting me on the LM Studio Discord server (see below link)!

<div align="center"><h2>CONTACT</h2></div>

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).

<div align="center">
<img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example00.png" alt="Example Image">
<img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example0.png" alt="Example Image">
<img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example1.png" alt="Example Image">
<img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example2.png" alt="Example Image">
<img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example3.png" alt="Example Image">
<img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example4.png" alt="Example Image">
<img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example5.png" alt="Example Image">
</div>
