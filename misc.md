## Vector Models
- Using chunks size/overlap of 800/400 on RTX 4090 processing a book.
- Approximately 5,500 chunks.

| Vector Model                                 | Compute Time (seconds) | Chunks/Second | Batch Size |
|----------------------------------------------|------------------------|---------------|------------|
| hkunlp--instructor-xl                        | 243.23                 | 22.70         | 2          |
| hkunlp--instructor-large                     | 82.17                  | 67.19         | 3          |
| Alibaba-NLP--gte-large-en-v1.5               | 39.30                  | 140.48        | 3          |
| thenlper--gte-large                          | 33.61                  | 164.27        | 3          |
| intfloat--e5-large-v2                        | 33.51                  | 164.76        | 3          |
| BAAI--bge-large-en-v1.5                      | 33.25                  | 166.05        | 3          |
| hkunlp--instructor-base                      | 27.00                  | 204.48        | 8          |
| Alibaba-NLP--gte-base-en-v1.5                | 14.78                  | 373.55        | 8          |
| sentence-transformers--all-mpnet-base-v2     | 14.42                  | 382.87        | 8          |
| BAAI--bge-base-en-v1.5                       | 12.01                  | 459.70        | 8          |
| intfloat--e5-base-v2                         | 11.98                  | 460.85        | 8          |
| thenlper--gte-base                           | 11.94                  | 462.40        | 8          |
| intfloat--e5-small-v2                        | 6.48                   | 852.01        | 10         |
| thenlper--gte-small                          | 6.36                   | 868.08        | 10         |
| BAAI--bge-small-en-v1.5                      | 6.31                   | 875.59        | 10         |
| sentence-transformers--all-MiniLM-L12-v2     | 4.31                   | 1281.21       | 30         |

## GPU Info

| Graphics Card                      | CUDA Cores |   Percentage of RTX 4090   |
|------------------------------------|------------|:--------------------------:|
| **GeForce RTX 4090**               |  16,384    |          **100.00%**       |
| GeForce RTX 4090 D                 |  14,592    |           89.06%           |
| GeForce RTX 3090 Ti                |  10,752    |           65.63%           |
| GeForce RTX 3090                   |  10,496    |           64.06%           |
| GeForce RTX 4080 Super             |  10,240    |           62.50%           |
| GeForce RTX 4080                   |   9,728    |           59.38%           |
| GeForce RTX 4090 Mobile/Laptop     |   9,728    |           59.38%           |
| GeForce RTX 3080 Ti                |   8,960    |           54.69%           |
| GeForce RTX 3080                   |   8,704    |           53.13%           |
| GeForce RTX 4070 Ti Super          |   8,448    |           51.56%           |
| GeForce RTX 4080 Mobile/Laptop     |   7,424    |           45.31%           |
| GeForce RTX 3080 Ti Mobile/Laptop  |   7,424    |           45.31%           |
| GeForce RTX 4070 Ti                |   7,680    |           46.88%           |
| GeForce RTX 4070 Super             |   7,168    |           43.75%           |
| GeForce RTX 3070 Ti                |   6,144    |           37.50%           |
| GeForce RTX 3070 Ti Mobile/Laptop  |   6,144    |           37.50%           |
| GeForce RTX 3070                   |   5,888    |           35.94%           |
| GeForce RTX 4070                   |   5,888    |           35.94%           |
| GeForce RTX 3070 Mobile/Laptop     |   5,120    |           31.25%           |
| GeForce RTX 3060 Ti                |   4,864    |           29.69%           |
| GeForce RTX 4070 Mobile/Laptop     |   4,608    |           28.13%           |
| Nvidia TITAN RTX                   |   4,608    |           28.13%           |
| GeForce RTX 4060 Ti                |   4,352    |           26.56%           |
| GeForce RTX 2080 Ti                |   4,352    |           26.56%           |
| GeForce RTX 3060 Mobile/Laptop     |   3,840    |           23.44%           |
| GeForce RTX 3060                   |   3,584    |           21.88%           |
| GeForce RTX 2080 Super             |   3,072    |           18.75%           |
| GeForce RTX 2080 Super Max-Q       |   3,072    |           18.75%           |
| GeForce RTX 4060                   |   3,072    |           18.75%           |
| GeForce RTX 2070 Super             |   2,560    |           15.63%           |
| GeForce RTX 2070 Super Max-Q       |   2,560    |           15.63%           |
| GeForce RTX 4050 Mobile/Laptop     |   2,560    |           15.63%           |
| GeForce RTX 3050 Ti Mobile/Laptop  |   2,560    |           15.63%           |
| GeForce RTX 2070                   |   2,304    |           14.06%           |
| GeForce RTX 2070 Max-Q             |   2,304    |           14.06%           |
| GeForce RTX 2060 (Dec 2021)        |   2,176    |           13.28%           |
| GeForce RTX 2060 Super             |   2,176    |           13.28%           |
| GeForce RTX 3050                   |   2,048    |           12.50%           |
| GeForce RTX 3050 Mobile/Laptop     |   2,048    |           12.50%           |
| GeForce RTX 2060                   |   1,920    |           11.72%           |
| GeForce RTX 2060 Max-Q             |   1,920    |           11.72%           |
| GeForce RTX 2060 (Jan 2019)        |   1,920    |           11.72%           |
| GeForce RTX 2060 (Jan 2020)        |   1,920    |           11.72%           |


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
