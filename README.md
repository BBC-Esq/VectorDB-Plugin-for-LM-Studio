<div align="center">
  <h2>🚀 Supercharge <a href="https://lmstudio.ai/">LM Studio</a> with a Vector Database!</h2>
  <h4>Search images, audio, and other general file types + Text to Speech<br><a href="https://www.youtube.com/watch?v=J1t95ecV11U">Introductory Video</a> &nbsp;&bull;&nbsp <a href="https://medium.com/@vici0549/search-images-with-vector-database-retrieval-augmented-generation-rag-3d5a48881de5">Medium Article</a></h4>
</div>

<h3 align="center">Table of Contents</h3>

<div align="center">
  <a href="#requirements">Requirements</a>
  &nbsp;&bull;&nbsp;
  <a href="#installation">Installation</a>
  &nbsp;&bull;&nbsp;
  <a href="#using-the-program">Using the Program</a>
  &nbsp;&bull;&nbsp;
  <a href="#request-a-feature-or-report-a-bug">Request a Feature or Report a Bug</a>
  &nbsp;&bull;&nbsp;
  <a href="#contact">Contact</a>
</div>
<br>
<div align="center">
    <a href="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example1.png" target="_blank">
        <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example1.png" alt="Example Image" width="350">
    </a>
    <a href="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example2.png" target="_blank">
        <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example2.png" alt="Example Image" width="350">
    </a>
    <a href="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example3.png" target="_blank">
        <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example3.png" alt="Example Image" width="350">
    </a>
    <a href="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example4.png" target="_blank">
        <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/example4.png" alt="Example Image" width="350">
    </a>
</div>

<a name="requirements"></a>
<div align="center"> <h3><u>Requirements</h3></div>
  
1) 🔥 Windows systems - sorry only able to test on Window but feel free to contribute!
2) 🐍[Python 3.11](https://www.python.org/downloads/release/python-3119/)
3) 📁[Git](https://git-scm.com/downloads)
4) 📁[Git Large File Storage](https://git-lfs.com/).
5) 🌐[Pandoc](https://github.com/jgm/pandoc/releases).
6) Compiler.
   > Certain dependencies don't have a pre-compiled "wheel" so you must build them with something like [Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and/or [Visual Studio](https://visualstudio.microsoft.com/).  If you choose Visual Studio, for example, make sure to select the "Desktop development with C++" extension and check the four boxes on the right containing the "SDK."

   <details>
     <summary>EXAMPLE ERROR ON WINDOWS</summary>
     <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/sample_error.png">
   </details>

   <details>
     <summary>EXAMPLE SOLUTION ON WINDOWS</summary>
     <img src="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/raw/main/build_tools.png">
   </details>

[Back to Top](#top)

<a name="installation"></a>
<div align="center"> <h3>INSTALLATION</h3></div>

<details>
  <summary>🪟WINDOWS INSTRUCTIONS</summary>
  
### Step 1
Download the latest "release," extract its contents, and navigate to the "src" folder to run the following commands:
  * NOTE: If you clone this repository you WILL NOT get the latest release.  Instead, you will development versions of this program which may or may not be stable.
### Step 2
Within the ```src``` folder, open a command prompt and create a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/):
```
python -m venv .
```
### Step 3
Activate the virtual environment:
```
.\Scripts\activate
```
### Step 4
Run the setup script:
   > Only for ```Windows``` for now.
```
python setup_windows.py
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

[Back to Top](#top)

<a name="using-the-program"></a>
<div align="center"> <h3>USING THE PROGRAM</h3></div>
<details>
  <summary>🖥️INSTRUCTIONS🖥</summary>

## Activate Virtual Environment
* Every time you want to use the program you must activate the virtual environment first from within the ```src``` folder.
## Start the Program
To start the program run this command:
```
python gui.py
```

## 🔥Important🔥
* Read the User Guide located within the graphical user interface itself.

## Download Vector Model
* Select and download a vector/embedding model from the ```Models Tab```.

## Create a Vector Database
This program extracts the text from a variety of file types and puts them into the vector database.  It also allows you to create summarizes of images and transcriptions of audio files to be put into the database.

### Entering General File Types

In the ```Create Database``` tab, select files you want to add to the database.  You can click the ```Choose Files``` button as many times as you want.
   > The supported file extensions are: ```.pdf```, ```.docx```, ```.epub```, ```.txt```, ```.html```, ```.enex```, ```.eml```, ```.msg```, ```.csv```, ```.xls```, ```.xlsx```, ```.rtf```, ```.odt```.

### Entering Images
This program uses "vision" models to create summaries of images, which can then be entered into the database and searched.  Before inputting images, I highly recommend that you test the various vision models for the one you like the most.

To test a vision model:
1) From the ```Create Database``` tab, select one or more images.
2) From the ```Settings``` tab, select the vision model you want to test.
3) From the ```Tools``` tab, process the images.

After determining which vision model you like, add images to the database by selecting them from the ```Create Database``` tab like any other file.  When you eventually create the database they will be automatically processed.
   > Supported file types are: ```.png```, ```.jpg```, ```.jpeg```, ```.bmp```, ```.gif```, ```.tif```, ```.tiff```

### Entering Audio Files
Audio files can be transcribed and put into the database to be searched.  Before transcribing a long audio file, I highly recommend testing the various ```Whisper``` models on a shorter audio file as well as experimenting with different ```batch``` settings.  Your goal should be to use as large of a ```Whisper``` model as your GPU supports and then adjust the batch size to keep the VRAM usage within your available VRAM> to ensure that the settings don't exceed your available VRAM.
   > Supported audio extensions include, but are not limited to: ```.mp3```, ```.wav```, ```.m4a```, ```.ogg```, ```.wma```

To test optimal settings:
1) Within the ```Tools``` tab, select a short audio file.
2) Select a ```Whisper``` model.  Read more about the size and quantization levels in the ```User Guide```.
3) Process the audio file.
4) Within the ```Create Database``` tab, doubleclick the transcription that was just created.
5) Skim the ```page content``` field to get a sense of whether the transcription is accurate enough for your use-case or if you need to selecta more accurate ```Whisper``` model.
   > To avoid getting a shoddy transcription, it is always better IMHO to choose as large a model as possible, reducing the ```batch``` setting if need be to keep the VRAM requirements within you're available VRAM.

Once you've obtained the optimal settings for your system, it's time to transcribe an audio file into the database:
1) Within the ```Create Database``` tab, delete any transcriptions you don't want entered into the database.
2) Create new transctiptions you want entered (repeate for multiple files).
   > Batch processing is not yet available.

### Actually Creating The Database
* Download a vector model from the ```Models``` tab.
* Within the ```Create Database``` tab, create the database.

### Manging the Database
* The ```Manage Database``` tab allows you to view the contents of all databases that you've created and delete them if you want.

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
* Load a model.
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

[Back to Top](#top)

<a name="request-a-feature-or-report-a-bug"></a>
## Request a Feature or Report a Bug

Feel free to report bugs or request enhancements by creating an issue on github or contacting me on the LM Studio Discord server (see below link)!

<a name="contact"></a>
<div align="center"><h3>CONTACT</h3></div>

All suggestions (positive and negative) are welcome.  "bbc@chintellalaw.com" or feel free to message me on the [LM Studio Discord Server](https://discord.gg/aPQfnNkxGC).


