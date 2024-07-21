## KoboldAI Flags

| Parameter | Description |
|-----------|-------------|
| `model_param` | Model file to load (positional argument) |
| `port_param` | Port to listen on (positional argument) |
| `-h, --help` | Show help message and exit |
| `--model [filename]` | Model file to load |
| `--port [portnumber]` | Port to listen on |
| `--host [ipaddr]` | Host IP to listen on. If empty, all routable interfaces are accepted |
| `--launch` | Launches a web browser when load is completed |
| `--config [filename]` | Load settings from a .kcpps file. Other arguments will be ignored |
| `--threads [threads]` | Use a custom number of threads if specified. Otherwise, uses an amount based on CPU cores |
| `--usecublas [[lowvram|normal] [main GPU ID] [mmq] [rowsplit] ...]` | Use CuBLAS for GPU Acceleration. Requires CUDA |
| `--usevulkan [[Device ID] ...]` | Use Vulkan for GPU Acceleration. Can optionally specify GPU Device ID |
| `--useclblast {0,1,2,3,4,5,6,7,8} {0,1,2,3,4,5,6,7,8}` | Use CLBlast for GPU Acceleration. Must specify platform ID and device ID |
| `--noblas` | Do not use any accelerated prompt ingestion |
| `--contextsize [256,512,1024,2048,3072,4096,6144,8192,12288,16384,24576,32768,49152,65536,98304,131072]` | Controls the memory allocated for maximum context size |
| `--gpulayers [[GPU layers]]` | Set number of layers to offload to GPU when using GPU |
| `--tensor_split [Ratios] ...` | For CUDA and Vulkan only, ratio to split tensors across multiple GPUs |
| `--ropeconfig [rope-freq-scale] [[rope-freq-base] ...]` | Uses customized RoPE scaling from configured frequency scale and frequency base |
| `--blasbatchsize {-1,32,64,128,256,512,1024,2048}` | Sets the batch size used in BLAS processing |
| `--blasthreads [threads]` | Use a different number of threads during BLAS if specified |
| `--lora [lora_filename] [[lora_base] ...]` | LLAMA models only, applies a lora file on top of model |
| `--noshift` | Do not attempt to Trim and Shift the GGUF context |
| `--nommap` | Do not use mmap to load newer models |
| `--usemlock` | For Apple Systems. Force system to keep model in RAM |
| `--noavx2` | Do not use AVX2 instructions, a slower compatibility mode for older devices |
| `--debugmode [DEBUGMODE]` | Shows additional debug info in the terminal |
| `--skiplauncher` | Doesn't display or use the GUI launcher |
| `--onready [shell command]` | An optional shell command to execute after the model has been loaded |
| `--benchmark [[filename]]` | Run benchmarks instead of starting server |
| `--multiuser [limit]` | Runs in multiuser mode, queuing incoming requests |
| `--remotetunnel` | Uses Cloudflare to create a remote tunnel for internet access |
| `--highpriority` | Increases the process CPU priority |
| `--foreground` | Windows only. Sends terminal to foreground for new prompts |
| `--preloadstory PRELOADSTORY` | Configures a prepared story json save file to be hosted on the server |
| `--quiet` | Enable quiet mode, hiding generation inputs and outputs in the terminal |
| `--ssl [cert_pem] [[key_pem] ...]` | Serves content over SSL. Requires valid UNENCRYPTED SSL cert and key .pem files |
| `--nocertify` | Allows insecure SSL connections |
| `--mmproj MMPROJ` | Select a multimodal projector file for LLaVA |
| `--password PASSWORD` | Enter a password required to use this instance |
| `--ignoremissing` | Ignores all missing non-essential files |
| `--chatcompletionsadapter CHATCOMPLETIONSADAPTER` | Select an optional ChatCompletions Adapter JSON file |
| `--flashattention` | Enables flash attention |
| `--quantkv [quantization level 0/1/2]` | Sets the KV cache data type quantization |
| `--forceversion [version]` | Override the detected model file format |

## Chat

| Model | VRAM Usage | Tok/s |
|-------|------------|-------|
| [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | 1.9 GB | 66 |
| [Qwen/Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) | 1.9 GB | 60 |
| [cognitivecomputations/dolphin-2.9.3-qwen2-0.5b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-qwen2-0.5b) | 2.4 GB | 67 |
| [stabilityai/stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b) | 2.5 GB | 74 |
| [internlm/internlm2-chat-1_8b](https://huggingface.co/internlm/internlm2-chat-1_8b) | 2.8 GB | 55 |
| [stabilityai/stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) | 2.9 GB | 57 |
| [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | 3.0 GB | 53 |
| [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 4.0 GB | 50 |
| [cognitivecomputations/dolphin-2.9.3-qwen2-1.5b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-qwen2-1.5b) | 4.2 GB | 58 |
| [01-ai/Yi-1.5-6B-Chat](https://huggingface.co/01-ai/Yi-1.5-6B-Chat) | 5.2 GB | 45 |
| [Qwen/Qwen1.5-4B-Chat](https://huggingface.co/Qwen/Qwen1.5-4B-Chat) | 5.4 GB | 41 |
| [Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) | 5.7 GB | 65 |
| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 5.7 GB | 50 |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 5.8 GB | 45 |
| [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3) | 5.8 GB | 46 |
| [microsoft/Orca-2-7b](https://huggingface.co/microsoft/Orca-2-7b) | 5.9 GB | 47 |
| [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) | 6.7 GB | 36 |
| [01-ai/Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat) | 7 GB | 45 |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 7.1 GB | 44 |
| [cognitivecomputations/dolphin-2.9-llama3-8b](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b) | 7.1 GB | 41 |
| [cognitivecomputations/dolphin-2.9.1-yi-1.5-9b](https://huggingface.co/cognitivecomputations/dolphin-2.9.1-yi-1.5-9b) | 7.2 GB | 30 |
| [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | 8.0 GB | 54 |
| [NousResearch/Nous-Hermes-Llama2-13b](https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b) | 9.9 GB | 38 |
| [microsoft/Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b) | 9.9 GB | 36 |
| [microsoft/Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) | 9.8 GB | 34 |
| [cognitivecomputations/dolphin-2.9.2-qwen2-7b](https://huggingface.co/cognitivecomputations/dolphin-2.9.2-qwen2-7b) | 9.2 GB | 52 |
| [cognitivecomputations/dolphin-2.9.2-Phi-3-Medium](https://huggingface.co/cognitivecomputations/dolphin-2.9.2-Phi-3-Medium) | 9.3 GB | 40 |
| [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) | 9.3 GB | 28 |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 10.0 GB | 36 |
| [stabilityai/stablelm-2-12b-chat](https://huggingface.co/stabilityai/stablelm-2-12b-chat) | 11.3 GB | 28 |
| [internlm/internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b) | 14.2 GB | 20 |


## Vision

| Model Name                        | CPU Metrics (Tokens/sec, Memory)       | GPU Metrics (Tokens/sec, Memory)       |
|-----------------------------------|----------------------------------------|----------------------------------------|
| Florence2 - base                  | 36.41 tokens/sec, 5 GB                 | 163.06 tokens/sec, 2.6 GB              |
| Florence2 - large                 | 15.77 tokens/sec, 10 GB                | 113.32 tokens/sec, 5.3 GB              |
| Moondream2                        | 4.15 tokens/sec, 8 GB                  | 79.07 tokens/sec, 4.6 GB               |
| Bakllava - 7b                     | N/A                                    | 50.03 tokens/sec, 6 GB                 |
| Llava 1.5 - 7b                    | N/A                                    | 42.61 tokens/sec, 5.6 GB               |
| Qwen-VL-Chat - 4b                 | N/A                                    | 42.46 tokens/sec, 11 GB                |
| Llava 1.5 - 13b                   | N/A                                    | 35.54 tokens/sec, 9.8 GB               |
| MiniCPM_v2 - 3.4b                 | N/A                                    | 33.37 tokens/sec, 10 GB                |
| MiniCPM-Llama3 2.5 | N/A                                    | 13.25 tokens/sec, 9.3 GB               |
| Cogvlm-chat-hf - 17.6b            | N/A                                    | 9.8 tokens/sec, 12.8 GB                |


## GPU Info

| Model | Size (GB) | CUDA Cores | Architecture | CUDA Compute |
|-------------------------------|-----------|------------|--------------|--------------|
| GeForce RTX 3050 Mobile/Laptop| 4 | 2048 | Ampere | 8.6 |
| GeForce RTX 3050 | 8 | 2304 | Ampere | 8.6 |
| GeForce RTX 4050 Mobile/Laptop| 6 | 2560 | Ada Lovelace | 8.9 |
| GeForce RTX 3050 Ti Mobile/Laptop | 4 | 2560 | Ampere | 8.6 |
| GeForce RTX 4060 | 8 | 3072 | Ada Lovelace | 8.9 |
| GeForce RTX 3060 | 12 | 3584 | Ampere | 8.6 |
| GeForce RTX 3060 Mobile/Laptop| 6 | 3840 | Ampere | 8.6 |
| GeForce RTX 4060 Ti | 16 | 4352 | Ada Lovelace | 8.9 |
| GeForce RTX 4070 Mobile/Laptop| 8 | 4608 | Ada Lovelace | 8.9 |
| GeForce RTX 3060 Ti | 8 | 4864 | Ampere | 8.6 |
| GeForce RTX 3070 Mobile/Laptop| 8 | 5120 | Ampere | 8.6 |
| GeForce RTX 3070 | 8 | 5888 | Ampere | 8.6 |
| GeForce RTX 4070 | 12 | 5888 | Ada Lovelace | 8.9 |
| GeForce RTX 3070 Ti | 8 | 6144 | Ampere | 8.6 |
| GeForce RTX 3070 Ti Mobile/Laptop | 8-16 | 6144 | Ampere | 8.6 |
| GeForce RTX 4070 Super | 12 | 7168 | Ada Lovelace | 8.9 |
| GeForce RTX 4080 Mobile/Laptop| 12 | 7424 | Ada Lovelace | 8.9 |
| GeForce RTX 3080 Ti Mobile/Laptop | 16 | 7424 | Ampere | 8.6 |
| GeForce RTX 4070 Ti | 12 | 7680 | Ada Lovelace | 8.9 |
| GeForce RTX 4080 | 12 | 7680 | Ada Lovelace | 8.9 |
| GeForce RTX 3080 | 10 | 8704 | Ampere | 8.6 |
| GeForce RTX 4070 Ti Super | 16 | 8448 | Ada Lovelace | 8.9 |
| GeForce RTX 3080 Ti | 12 | 8960 | Ampere | 8.6 |
| GeForce RTX 4080 | 16 | 9728 | Ada Lovelace | 8.9 |
| GeForce RTX 4090 Mobile/Laptop| 16 | 9728 | Ada Lovelace | 8.9 |
| GeForce RTX 4080 Super | 16 | 10240 | Ada Lovelace | 8.9 |
| GeForce RTX 3090 | 24 | 10496 | Ampere | 8.6 |
| GeForce RTX 3090 Ti | 24 | 10752 | Ampere | 8.6 |
| GeForce RTX 4090 D | 24 | 14592 | Ada Lovelace | 8.9 |
| GeForce RTX 4090 | 24 | 16384 | Ada Lovelace | 8.9 |

## Using CUDA

| Name                    | Function                                                                                  |
|-------------------------|-------------------------------------------------------------------------------------------|
| nvidia-cublas           | Provides the cuBLAS library for GPU-accelerated dense linear algebra computations.        |
| nvidia-cuda-runtime     | Includes runtime components necessary to execute CUDA applications.                      |
| nvidia-cuda-cupti       | Provides the CUDA Profiling Tools Interface (CUPTI) for profiling and tracing applications.|
| nvidia-cuda-nvcc        | Contains the NVIDIA CUDA Compiler (NVCC) for compiling CUDA code.                        |
| nvidia-cuda-nvrtc       | Provides the NVIDIA Runtime Compilation (NVRTC) library for just-in-time compilation.     |
| nvidia-cuda-sanitizer-api| Offers tools for memory and thread error detection in CUDA applications.                 |
| nvidia-cufft            | Provides the cuFFT library for computing fast Fourier transforms on GPUs.                |
| nvidia-curand           | Includes the cuRAND library for generating pseudorandom and quasirandom numbers.         |
| nvidia-cusolver         | Provides the cuSOLVER library for solving linear systems, eigenvalue, and SVD problems.  |
| nvidia-cusparse         | Contains the cuSPARSE library for sparse matrix operations on GPUs.                      |
| nvidia-npp              | Provides the NVIDIA Performance Primitives (NPP) library for image and signal processing. |
| nvidia-nvfatbin         | Manages and executes fat binaries for multiple GPU architectures.                        |
| nvidia-nvjitlink        | Provides NVJITLINK for just-in-time linking of device code.                              |
| nvidia-nvjpeg           | Includes the nvJPEG library for fast JPEG decoding on GPUs.                              |
| nvidia-nvml-dev         | Provides the NVIDIA Management Library (NVML) for monitoring and managing GPU devices.    |
| nvidia-nvtx             | Includes the NVIDIA Tools Extension (NVTX) library for annotating and instrumenting code. |
| nvidia-opencl           | Provides the OpenCL library for running OpenCL applications on NVIDIA GPUs.              |
| nvidia-cudnn            | Provides the cuDNN library for deep neural networks, offering highly optimized primitives for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. |



## CUDA Releases

| Date         | CUDA Version     |
|--------------|------------------|
| May 2024     | CUDA Toolkit 12.5.0 |
| April 2024   | CUDA Toolkit 12.4.1 |
| March 2024   | CUDA Toolkit 12.4.0 |
| January 2024 | CUDA Toolkit 12.3.2 |
| November 2023| CUDA Toolkit 12.3.1 |
| October 2023 | CUDA Toolkit 12.3.0 |
| August 2023  | CUDA Toolkit 12.2.2 |
| July 2023    | CUDA Toolkit 12.2.1 |
| June 2023    | CUDA Toolkit 12.2.0 |
| April 2023   | CUDA Toolkit 12.1.1 |
| February 2023| CUDA Toolkit 12.1.0 |
| January 2023 | CUDA Toolkit 12.0.1 |
| December 2022| CUDA Toolkit 12.0.0 |
| October 2022 | CUDA Toolkit 11.8.0 |

## CUDA Compute Compatibility

| CUDA Compute Capability | Architecture   | GeForce |
|-------------------------|----------------|---------|
| 6.1                     | Pascal         | Nvidia TITAN Xp, Titan X, GeForce GTX 1080 Ti, GTX 1080, GTX 1070 Ti, GTX 1070, GTX 1060, GTX 1050 Ti, GTX 1050, GT 1030, GT 1010, MX350, MX330, MX250, MX230, MX150, MX130, MX110 |
| 7.0                     | Volta          | NVIDIA TITAN V |
| 7.5                     | Turing         | NVIDIA TITAN RTX, GeForce RTX 2080 Ti, RTX 2080 Super, RTX 2080, RTX 2070 Super, RTX 2070, RTX 2060 Super, RTX 2060 12GB, RTX 2060, GeForce GTX 1660 Ti, GTX 1660 Super, GTX 1660, GTX 1650 Super, GTX 1650, MX550, MX450 |
| 8.6                     | Ampere         | GeForce RTX 3090 Ti, RTX 3090, RTX 3080 Ti, RTX 3080 12GB, RTX 3080, RTX 3070 Ti, RTX 3070, RTX 3060 Ti, RTX 3060, RTX 3050, RTX 3050 Ti(mobile), RTX 3050(mobile), RTX 2050(mobile), MX570 |
| 8.9                     | Ada Lovelace   | GeForce RTX 4090, RTX 4080 Super, RTX 4080, RTX 4070 Ti Super, RTX 4070 Ti, RTX 4070 Super, RTX 4070, RTX 4060 Ti, RTX 4060 |


## Ctranslate2 Quantization Compatibility
* NOTE: Only Ampere and later Nvidia GPUs support the new ```flash_attention``` parameter in ```Ctranslate2```.
### CPU

| Architecture              | int8_float32 | int8_float16 | int8_bfloat16 | int16 | float16 | bfloat16 |
|---------------------------|--------------|--------------|---------------|-------|---------|----------|
| x86-64 (Intel)            | int8_float32 | int8_float32 | int8_float32  | int16 | float32 | float32  |
| x86-64 (other)            | int8_float32 | int8_float32 | int8_float32  | int8_float32 | float32 | float32  |
| AArch64/ARM64 (Apple)     | int8_float32 | int8_float32 | int8_float32  | int8_float32 | float32 | float32  |
| AArch64/ARM64 (other)     | int8_float32 | int8_float32 | int8_float32  | int8_float32 | float32 | float32  |

### Nvidia GPU

| Compute Capability | int8_float32 | int8_float16 | int8_bfloat16 | int16   | float16 | bfloat16 |
|--------------------|--------------|--------------|---------------|---------|---------|----------|
| >= 8.0             | int8_float32 | int8_float16 | int8_bfloat16 | float16 | float16 | bfloat16 |
| >= 7.0, < 8.0      | int8_float32 | int8_float16 | int8_float32  | float16 | float16 | float32  |
| 6.2                | float32      | float32      | float32       | float32 | float32 | float32  |
| 6.1                | int8_float32 | int8_float32 | int8_float32  | float32 | float32 | float32  |
| <= 6.0             | float32      | float32      | float32       | float32 | float32 | float32  |

## Chat Model Benchmarks
* Tested using ```ctranslate2``` running in ```int8``` on an RTX 4090.

| Model                       | Tokens per Second | VRAM Usage (GB) |
|-----------------------------|:-----------------:|:---------------:|
| gemma-1.1-2b-it             | 63.69             | 3.0             |
| Phi-3-mini-4k-instruct      | 36.46             | 4.5             |
| dolphin-llama2-7b           | 37.43             | 7.5             |
| Orca-2-7b                   | 30.47             | 7.5             |
| Llama-2-7b-chat-hf          | 37.78             | 7.6             |
| neural-chat-7b-v3-3         | 28.38             | 8.1             |
| Meta-Llama-3-8B-Instruct    | 30.12             | 8.8             |
| dolphin-2.9-llama3-8b       | 34.16             | 8.8             |
| Mistral-7B-Instruct-v0.3    | 32.24             | 7.9             |
| SOLAR-10.7B-Instruct-v1.0   | 23.32             | 11.7            |
| Llama-2-13b-chat-hf         | 25.12             | 14.0            |
| Orca-2-13b                  | 20.01             | 14.1            |

## Concurrency
| **Library/Tool**             | **Type**                         | **Best Use Case**                                      | **Pros**                                           | **Cons**                                         |
|------------------------------|----------------------------------|--------------------------------------------------------|----------------------------------------------------|--------------------------------------------------|
| **Python `threading`**       | Threading                        | I/O-bound tasks, concurrent I/O operations, maintaining GUI responsiveness | Simple API, good for I/O-bound tasks, useful for GUI responsiveness | GIL limits effectiveness for CPU-bound tasks     |
| **Python `multiprocessing`** | Multiprocessing                  | CPU-bound tasks, parallelizing across multiple cores   | True parallelism, bypasses GIL                     | Higher memory overhead, complex IPC              |
| **Python `subprocess`**      | Process control                  | Running external commands, integrating system commands and other languages | Simple process control, capture I/O                | Limited to external process management           |
| **concurrent.futures**       | High-level API for Threading & Multiprocessing | Unified task management                                | Simplifies task execution, combines threading and multiprocessing, easy switching between ThreadPoolExecutor and ProcessPoolExecutor | Limited flexibility, higher abstraction          |
| **asyncio**                  | Async/Coroutine                  | I/O-bound, high concurrency tasks, network programming, web scraping | Non-blocking I/O, single-threaded concurrency      | Steeper learning curve due to coroutines and event loop |
| **QThread**                  | Threading                        | Integrating threads into the Qt event loop, signal-slot communication, long-running background tasks | Seamless Qt integration, easy inter-thread communication | More boilerplate, requires subclassing           |
| **QRunnable/QThreadPool**    | Threading                        | Managing multiple short-lived tasks within Qt applications | Efficient task management, less boilerplate        | Requires understanding of Qt threading architecture |
| **QtConcurrent**             | Threading                        | High-level parallel tasks in Qt, map-reduce style operations | High-level functions for parallel execution, automatic thread pooling | Less control over individual threads             |
| **QProcess**                 | Process control                  | Running external commands in Qt applications           | Integrates with Qt, handles process I/O, more control and integration with Qt's event loop | Limited to process control                       |

### Summary
> - **Python `threading`**: Best for simple I/O-bound tasks, concurrent I/O operations, and maintaining responsiveness in GUI applications.
> - **Python `multiprocessing`**: Best for CPU-bound tasks requiring true parallelism, good for parallelizing CPU-intensive tasks across multiple cores.
> - **Python `subprocess`**: Simple external process management. Use when you need straightforward process control and portability across different environments. Useful for integrating with system commands and other programming languages.
> - **concurrent.futures**: Unified API for high-level task management. Provides both ThreadPoolExecutor and ProcessPoolExecutor, allowing easy switching between threading and multiprocessing.
> - **asyncio**: Suitable for I/O-bound tasks with high concurrency, single-threaded. Particularly effective for network programming and web scraping tasks.
> - **QThread**: Ideal for complex threading in Qt applications with signal-slot communication. Useful for long-running background tasks in Qt applications.
> - **QRunnable/QThreadPool**: Efficient for managing multiple short-lived tasks in Qt.
> - **QtConcurrent**: Simplifies parallel task execution in Qt applications. Particularly good for map-reduce style operations.
> - **QProcess**: Handles running and managing external processes within Qt applications. Use when you need tight integration with the Qt event loop and signal-slot mechanism. Provides more control and integration with Qt's event loop compared to Python's subprocess.
