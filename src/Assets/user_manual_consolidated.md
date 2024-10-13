## Overview of Program

This program provides a comprehensive suite of tools for working with different machine learning models.  You can:
- Create a vector database out of image summaries, audio transcriptions, and a host of other file extensions.
- Query the vector database and get a response from a chat model.
- Use text to speech to have the chat model's response spoken to you.
- Scrapte python documentation from online and create a vector database out of it.

### What is LM Studio?
**LM Studio** is an application that allows users to run and interact with local language models on their own hardware. This program
integrates with LM Studio, and the GitHub repository contains detailed instructions for setup and usage. When you query the vector
database within the Query Database tab you can choose LM Studio as the backend that ultimately receives the query (along with the
contexts from the vector database) and provides a response to your question.

### What are local models and how do I use them?
The "local models" option within the Query Database tab allows you to use models downloaded directly from Huggingface instead
of LM Studio or other backends when getting a response from the LLM.  When you select and use a model for the first time, the
program will automatically download the model.  Please note that certain models are "gated," which means that you must first
enter a huggingface access token.  You can create an access token on Huggingface's website and then enter it within the "File"
menu within this program in the upper left.  You must do this before trying to use certain "gated" "local models".

### What local models are available to use?
When you select the "local models" backend within the Query Database tab you can choose multiple different local models from the
combobox.  The availble models change from time to time, but they are arranged from top to bottom from lease VRAM required to most.

### How do I get a huggingface access token?
You must create a huggingface account and then go to your profile.  On the left-hand side will be an "Access Tokens" option.
Then in the upper right is a "Create new token" button.  Check the box that says "Read access to contents of all public gated
repos you can access" then click "Create token."  You can then enter the access token in this program by going to the "File"
menu and selecting "Huggingface Access Token."

### What are context limits?
The phrase "context limit" usually refers to chat models, which essentially the same thing as a "max sequence length" when referring
to vector or embedding models.  Both essentially refer to the maximum length of "tokens" that they can process at one time.  If you
send too many tokens to a vector embedding model it will truncate the input, leading to suboptimal search results.  However, if you
send too many tokens to a chat model it will usually give an error.  Whichever backend you choose from within the Query Database Tab,
keep in mind the contxt limit of the chat model you plan on using.  Likewise, when specifying a "chunk size" when creating a vector
database keep in mind the maximum sequence length of the vector or embedding model you plan on using.  Also, it is crucial to remember
that the "chunk size" setting refers to the number of tokens while the maximum sequence length refers to tokens, and that there are
approximately 3-4 characters per token.

### What happens if I exceed the context limit or maximum sequence length and how does the chunk size and overlap setting relate?
Let's say you have set a chunk size of 1000 characters, and the system returns 13 chunks/contexts. Given that each token averages
about 3.5 characters, this results in approximately 3714 tokens. If your query consists of 96 tokens, the total tokens used would
be about 3810, leaving only around 286 tokens (4096 - 3810) for the model's response generation.

### How many context should I retrieve when querying the vector database?
For simple question-answer use cases, 3-6 chunks should suffice. For a typical book, a chunk size of 1200 characters with an
overlap of 600 characters can return up to 6 contexts. Advanced models like `instructor-xl` often retrieve relevant context in
the first or second result.  However, it is EQUALLY important to experiment with how you write your question.  Oftentimes, the
specific phrasing of a question has more impact than simply returning more contexts.  If you practice, you should get the answer
to your question in 3-6 chunks.

### What does the chunks only checkbox do?
You can test chunk sizes using the "chunks only" checkbox in the search tab without connecting to LM Studio or the local model.
This helps assess whether your chunk size setting is appropriate for capturing key ideas. You may also compare responses by using
different chat models with the same query and monitoring token usage to avoid exceeding the context limit.

## What are embedding or vector models?
Embedding models convert chunks of text into numerical representations called vectors, capturing the semantic meaning of
the text. These vectors are stored in a vector database for efficient similarity searches. When a query is submitted, it
is also converted into a vector, and the database retrieves chunks with vectors similar to the query vector.

### Which embedding or vector model should I choose?
Initially, a wide variety of vector models were used. However, modern models are versatile enough for various tasks, leading
to a focus on two main categories: Sentence Retrieval Models (`sentence-t5` Series) and Generalist Models:
Sentence Retrieval Models are specifically trained to retrieve sentences similar to the query.  They serve a very narrow use case.
For example, you might use them with the following questions:
   - "Quote all sentences that discuss the main character eating food."
   - "Provide all sentences where the court discusses the elements of a defamation claim."
Generalist Models are suited for a vast majority of use cases such as question answering.

### What are the characteristics of vector or embedding models?
Understanding the characteristics of embedding models is crucial for selecting the right one:
- **Max Sequence Length:** Defines the maximum number of tokens the model can process at once. For instance, if a model has
a 512-token limit, inputs longer than that will be truncated.
- **Vector Dimensions:** Refers to the size of the vector representation (e.g., 768 or 1024 dimensions). While higher dimensions
can capture more nuanced information, they may also increase computational costs.
- **Model Size:** Larger models, with more parameters, typically offer better performance but require more resources.
Considerations include VRAM usage and processing time, especially on systems with limited hardware capabilities.

### What are the dimensions of a vector or embedding model?
The higher the dimensions the more nuanced meanings it can discer in test, leading to better search results.

### Tips for using vector or embedding models
Opt for higher-quality vector or embedding models if system resources allow..
Adjust the chunk size and chunk overlap settings to suit the text you are entering into the vector database.

Use the half-precision checkbox setting if you have a GPU, unless you absolutely must have the higest quality.  The difference
between using it and not is minimal in terms of quality but using it provides a 2x speedup.

### What Are Vision Models?
Vision models are specialized neural networks designed to interpret and generate descriptions of images. By combining
computer vision and natural language processing, they produce textual summaries of visual content. These summaries are
converted into vectors and stored in the vector database, enabling image searches based on textual queries.  This program
uses multiple vision models.  You can select one in the Settings Tab or test all of them in the Tools Tab.

### What vision models are available in this program?
The vision models in this program each with varying capabilities and resource requirements.  The following are available:
The two "Florence2" vision models (base and large) deliver very good quality and are the only ones that can be run on CPU.
The "Moondream2" vision model competes with the Florence2 models in terms of quality, is a little slower, but its summaries are
shorter on average, which makes it ideal for when you create a vector database with a smaller chunks size and overlap setting.
The two "Llava 1.6" vision models (7b and 13b) are relatively new and produce very high quality descriptions of images.
The MiniCPM-v-2.6 vision model, some say, has the highest quality but is also the slowest vision model offered by this program.

### Do you have any tips for choosing a vision model?
It’s recommended to test models by:
- Adding sample images
- Selecting a model in the settings
- Processing and reviewing the generated summaries
Adjust the `Chunk Size` to ensure it accommodates the longest summaries without splitting them, preserving coherence. For
improved search performance, consider creating separate databases specifically for images, rather than mixing them with text documents.

### What is whisper and how does this program use voice recording or transcribing an audio file?
Whisper is an advanced speech recognition model developed by OpenAI, designed for highly accurate transcription of audio into
text. This program leverages Whisper models for two primary functions: 
- Transcribing your spoken question when querying the vector database
- Converting audio files into `.txt` files for use in vector databases that can subsequently be entered into the vector database.

### How can I record my question for the vector database query?
To transcribe a spoken question, go to the "Query Database" tab, click the "Voice Recorder" button to begin recording
and then speak clearly. Click the button again to stop recording, and the transcribed text will appear in the question box.

### How can I transcribe an audio file to be put into the vector database?
To transcribe an audio file, navigate to the Tools tab, select an audio file (most formats are supported), and click the Transcribe
button. After the transcription is complete you can see it in the "Create Database" tab and it will be entered into the vector database
when you create it.  The transcribing functionality uses the powerful `WhisperS2T` library with the `Ctranslate2` backend.  Make
sure to adjust the "Batch" setting when transcribing an audio file depending on the size of the whisper model you choose.
Increasing the batch size can improve speed but demands more VRAM, so care should be taken not to exceed your GPU’s capacity.

### What are the distil variants of the whisper models when transcribing and audio file?
Distil variants of Whisper models use approximately 70% of the resources of their full counterparts and are faster with little loss
in quality.

### What whisper model should I choose to transcribe a file?
- For important audio files, choose the highest-quality Whisper model that your system can handle to improve accuracy.
Better transcriptions lead to more accurate searches in the vector database.  With that being said, the difference in quality
between a "float32" model and "float16" or "bfloat16" is very little.  Therefore, try to prefer using a half-precision version
of the larges size that your GPU can handle, and reduce the batch size if necessary to ensure enough VRAM is available.

### What are floating point formats, precision, and quantization?
Understanding floating point formats is key when making decisions about model selection and quantization.
Floating point formats represent real numbers in binary using a combination of sign, exponent, and fraction (mantissa) bits:
- **Sign Bit**: Indicates whether the number is positive or negative.
- **Exponent Bits**: Determine the range (or magnitude) of the value.
- **Fraction (Mantissa) Bits**: Control the precision of the value.

### What are the common floating point formats?
1. **float32 (32-bit floating point)**  
   With 1 sign bit, 8 exponent bits, and 23 fraction bits, this format provides high precision and a wide range, making it
   a standard choice for many computing tasks.
2. **float16 (16-bit floating point)**  
   Comprising 1 sign bit, 5 exponent bits, and 10 fraction bits, float16 offers reduced precision and range, but uses less
   memory and computational power.
3. **bfloat16 (Brain Floating Point)**  
   This format features 1 sign bit, 8 exponent bits, and 7 fraction bits. It has the same range as float32 but with lower
   precision, making it particularly useful for deep learning applications.
**Range and Precision Comparison:**

| Format    | Approximate Range                    | Precision (Decimal Digits) |
|-----------|--------------------------------------|----------------------------|
| float32   | ±1.4 × 10^-45 to ±3.4 × 10^38        | 6 to 9                     |
| float16   | ±6.1 × 10^-5 to ±6.5 × 10^4          | 3 to 4                     |
| bfloat16  | ±1.2 × 10^-38 to ±3.4 × 10^38        | 2 to 3                     |

### What are precision and range regarding floating point formats and which should I use?
The choice of floating point format has several key implications:
- **Precision** affects the detail and accuracy of computations.
- **Range** determines the scale of values that can be represented.
- **Trade-offs** arise when opting for lower precision formats, as they reduce memory usage and increase processing speed
but may slightly reduce accuracy.

### What is Quantization?
Quantization reduces the precision of the numbers used to represent a model's parameters, which results in smaller models
and lower computational requirements. The main goals of quantization are to:
- Improve model speed.
- Reduce memory usage (RAM/VRAM).
- Enable models to run on resource-constrained hardware.
There are two main methods of quantization:
- **Post-Training Quantization** is applied after the model is trained.
- **Quantization-Aware Training** incorporates quantization during the training process to minimize accuracy loss.
Common quantization levels include:
- **int8 (8-bit integer)**, which significantly reduces model size but may introduce quantization errors.
- **float16/bfloat16**, which reduces size with minimal impact on accuracy.

### What are the aspects or effects of quantization?
- **Model Size Reduction**: Smaller data types take up less storage.
- **Performance Increase**: Reduced data size speeds up computation.
- **Potential Accuracy Loss**: Reduced precision may introduce errors, though often negligible for many applications.


## What settings are available in this program and how can I adjust them?
The "Settings" Tab contains most of the settings for LM Studio, querying the database, creating the database, the text to speech
functionality, and the vision models.  Please ask me a question about the specific setting or group of settings you're interested in?

### What are the LM Studio Settings and what do they do?
The "Port" setting determines which port the LM Studio server is running on and must match.
The "Max Tokens" setting determins the maximum number of tokens the LLM has to generate in a response. Setting the value to `-1` allows
the model to use as many tokens as possible within its context window.
The "Temperature" setting controls the randomness of the model's output, ranging from 0 to 1. Lower values such as 0.1 result in more
deterministic, focused responses, while higher values like 0.9 generate more creative and varied outputs.
The "Prefix" and "Suffix" settings allow you to control the prompt formatting of what is sent to LM Studio.  Generally, it's
preferable to use the "Prompt Format" setting to choose a premade chat template if one is available that matches the LLM you are using
within LM Studio.
The "Disable" button disables all prompt formatting and allows you to control it directly within LM Studio.

### What are the database creation settings and what do they do?
When querying a vector database you can adjust the settings within the "Settings" Tab.  The available settings are:
- Device
- Similarity
- Contexts
- Search Term Filter
- Contexts
- File Type
All of these settings interact as follows:
When a query is submitted, only chunks from the selected "file type" are eligible to be returned. Out of these, only those
meeting the `similarity` threshold are eligible to be returned. If there are any remaining chunks, the most relevant ones
are returned up to the maximum amount specified by the `contexts` setting. After this, the `search term filter` setting removes
any chunks that do not contain the specified term. Finally, the remaining chunks are sent to the LLM along with the user’s
query for a response.  Ask me for more information about a specific setting.

### What is the Device setting?
In the "Settings" tab you can specify the device to either create the database with or search it with.  It is highly recommended
to always specify GPU when creating the database (if one is available) and CPU when querying the database.

### What is the chunk size setting?
The "Chunk Size" setting defines the maximum length of text chunks (in characters) for the vector database. Optimal sizes are around
1000-1200 characters for books and 200-400 characters for shorter texts like tweets. Ensure that the chunk size is less
than the model's max sequence length in tokens, with 1 token approximately equaling 3-4 characters.

### What is the chunk overlap setting?
The "Chunk Overlap" setting specifies how many characters overlap between consecutive chunks. This prevents important information from
being split across chunks and provides context continuity. It's recommended to set the overlap to 25-50% of the `Chunk Size`.
The "Half-Precision" checkbox runs the vector or embedding model in either float16 or bfloat, depending on your GPU's capability.  It
is disabled if you do not have a GPU available.

### What is the contexts setting?
The "Contexts" setting when querying the database determines the maximum number of chunks that can possibly be returned.
Make sure and strike a balance between providing enough context while staying within the model's context window.

### What is the similarity setting?
The "Similarity" setting when querying the vector database determines how similar chunks must be to the query.  You can set a value
between zero and 1. Higher values return more chunks of text while lower values require a higher degree of relevant in order to
be returned.

### What is the search term filter setting?
The "Search Term Filter" setting when querying a vector database allows you to only include chunks that contain a specific term.
It is not case-sensitive, but it does require an exact match (e.g., “child” will not match “children”). It is useful when looking for
specific terms within the contexts stored within the vector database.  It is important to note that this filter only applies after
context have been returend by the vector databaes.  In other words, it subsequently removes any contexts that do not contain the
specified search term before they are sent to the LLM for a response.  You can clear it by using the "Clear Filter" button.

### What is the File Type setting?
The "File Type" setting when querying a vector refers to the type of document that a context or chunk originated from.  When this
program creates the vector database is stores as metadata the type of document that each chunk came from.  You can choose "All Files"
have no filter, "Images Only" for only image descriptions, "Audio Only" for only audio transcriptions, and "Documents Only" to only
receive chunks that originate from a file that is not an audio transcription or image summary.

### What are text to speech models and how are they used in this program?
This program uses text to speech models to speak the response from the LLM after querying the vector database.
The "Bark" text to speech backend has two models: "Normal," which provides slightly better quality, and "Small," which is faster and
uses fewer resources.  You can select from various "speakers" such as `v2/en_speaker_6` (high-quality) and `v2/en_speaker_9`
(the only female voice).  Using Bark requires a GPU.
The WhisperSpeech text to speech backend consists of a Speech to Acoustics (S2A) model, which converts speech into acoustic features,
and Text to Speech model (T2S), which generates speech from text. You can mix and match S2A and T2S models for desired quality
versus performance.
The "Chat TTS" text to speech backend has no adjustable settings but can run on CPU.  It generally has inferior quality compared to Bark
and WhisperSpeech and struggles with non-standard words or symbols.
The "Google TTS" text to speech backend is the only one that does not rely on a GPU or CPU; however, it requires an Internet connection.
There are some slight unnatural pauses due to processing limitations of the free API.

### Which text to speech backend or models should I use
Generally it's recommended to experiment with each to your liking.  However, in general Bark and WhisperSpeech produce the highest
quality results, Chat TTS is below them but can be run on GPU as well as CPU, and Google TTS is comparable to Chat TTS in terms of
quality but requires an Internet connection.

### Can I back up or restore my databases and are they backed up automatically
When you create a vector database it is automatically backed up.  However, if you want to manually backup all databases you can go
to the "Tools" tab and click the Backup All Databases button.  Likewise, you can restore all backed up databases within the Tools Tab.

### What happens if I lose a configuration file and can I restore it?
The program cannot function without the config.yaml file.  If you lose it accidentally or it gets corrupted for some reason, you can
restore a default version by:
1. Try restoring database backups first.  
2. If necessary, copy the original `config.yaml` from the User Guide folder to the main directory.  
3. Delete old files and folders in "Vector_DB" and "Vector_DB_Backup" to prevent conflicts.

### What are some good tips for searching a vector database?
Understanding the interaction between `contexts`, `similarity`, and `search filter` is key to performing efficient searches.
To test your strategy, it is often effective to set a high contexts value along with a permissive similarity threshold. By
experimenting with different search terms, you can observe how the number of returned chunks changes and make adjustments to the
settings based on the relevance of the results.

### General VRAM Considerations
To conserve VRAM, disconnect secondary monitors from the GPU and, if available, use motherboard graphics ports instead. This requires
enabling integrated graphics in the BIOS, which is often disabled by default when a dedicated GPU is installed. This can be
particularly useful if your CPU has integrated graphics, such as Intel CPUs without an "F" suffix, which support motherboard
graphics ports.

### How can I manage vram?
For optimal performance, ensure that the entire LLM is loaded into VRAM. If only part of the model is loaded, performance can be
significantly degraded. It’s also important to manage VRAM efficiently by ejecting unused models when creating the vector database
and reloading the LLM after the database creation is complete. When querying the vector database, using the CPU instead of the GPU
is recommended to conserve VRAM for the LLM, as querying is less resource-intensive and can be effectively handled by the CPU.

### What are maximunm context length and maximum sequence length and how to they relate?
Each embedding model has a maximum sequence length, and exceeding this limit can result in truncation. To avoid this, regularly
check the maximum sequence length of the model and adjust your settings accordingly. Reducing chunk size or the number of contexts
can help stay within these limits. Maximum "context lengh" refers to chat models and is very similar to maximum sequence length.
The key thing to understand is that the chunks you put into the vector database should be within the max sequence length of the
vector or embedding model you choose and the maximum context or chunks you retrieve from the vector database multiplied by their
length should stay within the chat model's context length limit.  And make sure to leave enough context for a response.

### What is the scrape documentaton feature?
Within the Tools tab you can select multiple python libraries and scrape their documentation.  Multiple .html files will be downloaded
and you can subsequently create a vector database out of them.  Larger more complex libraries can take a significant amount of time
to scrape to make sure you have a stable Internet connection.

### Which vector or embedding models are available in this program?
This program includes vector models from six vendors and you can read more about them on the Models tab:
- Alibaba
- BAAI
- hkunlp
- intfloat
- sentence-transformers
- thenlper
To download a vector or embedding model simply click the radio button next to and then the Download Selected Model button.

### What is the manage databaes tab?
The Manage Databases tab allows you to see all of the vector databases that you've created thus far and what documents are in them.
You can also doubleclick a document and it will open unless you've moved it since creating the vector database.  Functionality will
be added eventually to add or remove specific files from a pre-existing vector database.

### How can I create a vector database?
Go to the Create Database tab, click the Choose Files button and select one or more files.  You can do this multiple times if you need
to select files in different directories.  If you select file types that are not compatible it will warn you and allow you to back out.
If you select any images, make sure and read about the various vision models that this program uses and their unique features.  Also,
if you want to add transcriptions to the vector database you must do that in the Tools tab before creating the vector database.  Once
you have added all the necessary files, select the embedding model from the combobox and simply click "Create Database."
