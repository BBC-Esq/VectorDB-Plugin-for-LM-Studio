Kobold AI Summary

| Choice | OpenBLAS | CLBlast | CuBLAS | Vulkan | No BLAS | NoAVX2 | NoMMAP | Low VRAM | MMQ | Row-Split | GPU Layers | Tensor Split |
|--------|----------|---------|--------|--------|---------|--------|--------|----------|-----|-----------|------------|--------------|
| Use OpenBLAS | ✅ | | | | | | | | | | | |
| Use CLBlast | | ✅ | | | | | | | | | ✅ | |
| Use CuBLAS | | | ✅ | | | | | ✅* | ✅* | ✅* | ✅ | ✅ |
| Use Vulkan | | | | ✅ | | | | | | | ✅ | ✅ |
| Use No BLAS | | | | | ✅ | | | | | | | |
| CLBlast NoAVX2 | | ✅ | | | | ✅ | | | | | ✅ | |
| Vulkan NoAVX2 | | | | ✅ | | ✅ | | | | | ✅ | ✅ |
| NoAVX2 Mode | | | | | | ✅ | | | | | | |
| Failsafe Mode | | | | | ✅ | ✅ | ✅ | | | | | |

✅* = Optional setting for CuBLAS

# Kobold AI API Documentation

## Required Arguments

- `--model [filename]`: Model file to load.
- `--port [portnumber]`: Port to listen on.
- You can use the foregoing as "positional" arguments but just using the flags makes things more uniform.

## Optional Arguments

- `--config [filename]`: Load settings from a `.kcpps` file. Other arguments will be ignored.
- `--noavx2`: Do not use AVX2 instructions; enables a slower compatibility mode for older devices.
- `--nommap`: If set, do not use `mmap` to load newer models.
- `--usemlock`: For Apple systems. Forces the system to keep the model in RAM rather than swapping or compressing. On systems with limited RAM, setting `--usemlock` can prevent frequent memory swapping and improve performance. Disabled by default.
- `--skiplauncher`: Doesn't display or use the GUI launcher.
- `--quiet`: Enable quiet mode, which hides generation inputs and outputs in the terminal. Quiet mode is automatically enabled when running a horde worker.
- `--onready [shell command]`: An optional shell command to execute after the model has been loaded.
  - This is an advanced parameter intended for script or command line usage. You can pass a terminal command (e.g., start a Python script) to be executed after Koboldcpp has finished loading. This runs as a subprocess and can be useful for starting Cloudflare tunnels, displaying URLs, etc.
  - Example: `--onready "python script.py"` runs the specified Python script after the model is loaded.
- `--threads [number]`: Specifies the number of CPU threads to use for text generation.
  - If a number is not specified a default value is calculated.
    - If CPU core count > 1: Uses half the physical cores, with a minimum of 3 and maximum of (physical cores - 1)
    - For systems with 1 core: Uses 1 thread.
  - Intel CPU Specific: For Intel processors, the maximum default is capped at 8 threads to avoid using efficiency cores
  - Usage: `--threads [number]`
    - Note: If not specified, the program uses the calculated default value

## --usecublas

The `--usecublas` argument enables GPU acceleration using CuBLAS (for NVIDIA GPUs) or hipBLAS (for AMD GPUs). For hipBLAS binaries, check the YellowRoseCx ROCm fork.

### Usage:

- `--usecublas [lowvram|normal] [main GPU ID] [mmq] [rowsplit]`
- Example: `--usecublas lowvram 0 mmq rowsplit`

### Optional Parameters:

- `lowvram` or `normal`
  - `lowvram`: Prevents offloading to the GPU the KV layers. Suitable for GPUs with limited memory.
  - `normal`: Default mode.
- `main GPU ID`: A number (e.g. 0, 1, 2, or 3) selecting a specific GPU. If not specified, all available GPUs will be used.
- `mmq`: Uses “quantized matrix multiplication” during prompt processing instead of cuBLAS. This is slightly faster and uses slightly less memory for Q4_0, but is slower for K-quants. Generally, cuBLAS is faster but uses slightly more VRAM.
- `rowsplit`: If multiple GPUs are being used, splitting occurs by “rows” instead of “layers,” which can be beneficial on some older GPUs.

### Unique Features:

- Can use `--flashattention`, which can be faster and more memory efficient.
  - If `--flashattention` is used `--quantkv [level]` can also be used but “context shifting” will be disabled. Here, level 0=f16, 1=q8, 2=q4.

## --usevulkan

Enables GPU acceleration using Vulkan, which is compatible with a broader range of GPUs and iGPUs. See more info at [Vulkan GPU Info](https://vulkan.gpuinfo.org/).

### Optional Parameter:

- `Device ID`: An integer specifying which GPU device to use. If not provided, it defaults to the first available Vulkan-compatible GPU.

### Usage:

- `--usevulkan [Device ID]`
- Example: `--usevulkan 0`

## Using Multiple GPUs

The program first determines how many layers are computed on the GPU(s) based on `--gpulayers`. Those layers are split according to the `--tensor_split` parameter. Layers not offloaded will be computed on the CPU. It is possible to specify `--usecublas`, `--usevulkan` or `--useclblast` and not specify `--gpulayers` in which case the prompt processing will occur on the GPU(s) but the per-token inference will not.

### Not Specifying GPU IDs:

- By default, if no GPU IDs are specified after `--usecublas` or `--usevulkan` all compatible GPUs will be used and layers will be distributed equally.
  - NOTE: This can be bad if the GPUs are different sizes.
- Use `--tensor_split` to control the ratio, e.g., `--tensor_split 4 1` for a 80%/20% split on two GPUs.
- The number of values in `--tensor_split` should match the total number of available GPUs.

### Specifying a Single GPU ID:

- Don't use `--tensor_split`. However, you can still use `--gpulayers`.

### Specifying Some GPUs:

- If some (but not all) GPU IDs are provided after `--usecublas` or `--usevulkan`, only those GPUs will be used for layer offloading.
- Use `--tensor_split` to control the distribution ratio among the specified GPUs.
- The number of values in `--tensor_split` should match the number of GPUs selected.
  - Example: With four GPUs available but only specifying the last two with `--usecublas 2 3`, also using `--tensor_split 1 1` would offload an equal amount of layers to the third and fourth GPUs but none to the first two.

### Usage with `--useclblast`:

- `--gpulayers` is supported by `--useclblast` but `--tensor_split` is not.

## --useclblast

Enables GPU acceleration using CLBlast, based on OpenCL. Compatible with a wide range of GPUs including NVIDIA, AMD, Intel, and Intel iGPUs. More info can be found at [CLBlast README](https://github.com/CNugteren/CLBlast/blob/master/README.md).

### Required Arguments:

- `Platform ID`: An integer between 0 and 8 (inclusive).
- `Device ID`: An integer between 0 and 8 (inclusive).

### Usage:

- `--useclblast [Platform ID] [Device ID]`
  - Platform ID: An integer between 0 and 8 (inclusive).
  - Device ID: An integer between 0 and 8 (inclusive).
- Both arguments are required.
- Example: `--useclblast 1 0`
  - The API instructions are unclear whether more than one compatible device can be specified. In any event, `--tensor_split` cannot be used.

## OpenBLAS:

- Only used by CPU, not GPU.
- Enabled in Windows by default, but other platforms require a separate installation.

## BLAS Configuration

All BLAS acceleration (including OpenBLAS) can be disabled using `--noblas` or `--blasbatchsize -1`. Setting to -1 disables BLAS mode but retains other benefits like GPU offload.

### --blasbatchsize

Sets the batch size used in BLAS processing.

- Default: 512
- Options: -1, 32, 64, 128, 256, 512, 1024, 2048

### --blasthreads

Specifies the number of threads to use during BLAS processing.

- If not specified, it uses the same value as `--threads`.
- If left blank, it will automatically set to a value slightly less than the CPU count.
- Recommendation: When running with full GPU offload, setting it to 1 thread may be sufficient.

## Samplers in KoboldCpp

Samplers determine how the AI selects the next token from a list of possible tokens. There are various samplers with different properties, but generally, you will only need a few.

### Sampler Order:

- Controls the sequence in which samplers are applied to the list of token candidates when choosing the next token.
- Hardcoded into the source code as `[6,0,1,3,4,2,5]` to avoid poor outputs (0 = Top K  1 = Top P, 2 = Typical P, 3 = Top A, 4 = Min P, 5 = Temperature, 6 = TFS) 
- Don't change.

### Good Default Settings:

- `top_p`: 0.92
- `rep_pen`: 1.1
- `Temperature`: 0.7
- Leave everything else disabled by default

### Sampler Descriptions:

1. **Top-K**: 
   - Parameter: `top_k`
   - Function: Limits the number of possible words to the top K most likely options, removing everything else.
   - Usage: Can be used with Top-P. Set value to 0 to disable its effect.
2. **Top-A**: 
   - Parameter: `top_a`
   - Function: Alternative to Top-P. Removes all tokens with a softmax probability less than `top_a * m^2` where `m` is the maximum softmax probability.
   - Usage: Set value to 0 to disable its effect.
3. **Top-P**: 
   - Parameter: `top_p`
   - Function: Discards unlikely text during sampling. Considers words with the highest cumulative probabilities summing up to P.
   - Effect: Low values make the text predictable by removing uncommon tokens.
   - Usage: Set value to 1 to disable its effect.
4. **TFS (Top-Filter Sampling)**: 
   - Parameter: `tfs`
   - Function: Alternative to Top-P. Removes the least probable words from consideration, using second-order derivatives.
   - Benefit: Can improve the quality and coherence of generated text.
5. **Typical**: 
   - Parameter: `typical_p`
   - Function: Selects words randomly with equal probability.
   - Effect: Produces more diverse but potentially less coherent text.
   - Usage: Set value to 1 to disable its effect.
6. **Temperature**: 
   - Parameter: `temperature`
   - Function: Controls the randomness of the output by scaling probabilities without removing options.
   - Effect: Lower values produce more logical, less creative text.
7. **Repetition Penalty**: 
   - Parameter: `rep_pen`
   - Function: Applies a penalty to reduce the usage of recently used words, making the output less repetitive.

## --contextsize

- Controls the memory allocated for maximum context size. Adjust this if you need more RAM for larger contexts.
- Default: 4096
- Supported Values:
  - 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072
- Warning: Use values outside the supported range at your own risk.

### Usage:

- `--contextsize [Value]`
- Example: `--contextsize 8192` allocates memory for a context size of 8192.

## Context Shifting

Context Shifting is a better version of Smart Context that only works for GGUF models. This feature utilizes KV cache shifting to automatically remove old tokens from context and add new ones without requiring any reprocessing. It is on by default. To disable Context Shifting, use the flag `--noshift`.

## Streaming

KoboldCpp now supports a variety of streaming options. Kobold Lite UI supports streaming out of the box, which can be toggled in Kobold Lite settings. Note: the `--stream` parameter is now deprecated and should not be used.

### Streaming Methods:

1. **Polled-Streaming (Recommended)**:
   - Default Method: Used by the Kobold Lite UI.
   - Mechanism: Polls for updates on the `/api/extra/generate/check` endpoint every second.
   - Advantages: Relatively fast and simple to use.
   - Drawback: Some may find it a bit "chunky" as it does not update instantaneously for every single token.
2. **Pseudo-Streaming**:
   - Status: An older method no longer recommended due to performance overheads.
   - Usage with Kobold Lite: Enable streaming and append `&streamamount=x` to the end of the Lite URL, where `x` is the number of tokens per request.
   - Drawback: Has a negative performance impact.
3. **SSE (True Streaming)**:
   - Supported by: A few third-party clients such as SillyTavern and Agnaistic, available only via the API.
   - Mechanism: Provides instantaneous per-token updates.
   - Requirements: Requires a persistent connection and special handling on the client side with SSE support.
   - Usage: This mode is not used in Lite or the main KoboldAI client. It uses a different API endpoint, so configure it from your third-party client according to their provided instructions.


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
