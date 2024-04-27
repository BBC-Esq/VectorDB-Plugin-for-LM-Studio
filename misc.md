## CUDA Compute Compatibility

| CUDA Compute Capability | GeForce |
|-------------------------|---------|
| 6.1                     | Nvidia TITAN Xp, Titan X, GeForce GTX 1080 Ti, GTX 1080, GTX 1070 Ti, GTX 1070, GTX 1060, GTX 1050 Ti, GTX 1050, GT 1030, GT 1010, MX350, MX330, MX250, MX230, MX150, MX130, MX110 |
| 7.0                     | NVIDIA TITAN V |
| 7.5                     | NVIDIA TITAN RTX, GeForce RTX 2080 Ti, RTX 2080 Super, RTX 2080, RTX 2070 Super, RTX 2070, RTX 2060 Super, RTX 2060 12GB, RTX 2060, GeForce GTX 1660 Ti, GTX 1660 Super, GTX 1660, GTX 1650 Super, GTX 1650, MX550, MX450 |
| 8.6                     | GeForce RTX 3090 Ti, RTX 3090, RTX 3080 Ti, RTX 3080 12GB, RTX 3080, RTX 3070 Ti, RTX 3070, RTX 3060 Ti, RTX 3060, RTX 3050, RTX 3050 Ti(mobile), RTX 3050(mobile), RTX 2050(mobile), MX570 |
| 8.9                     | GeForce RTX 4090, RTX 4080 Super, RTX 4080, RTX 4070 Ti Super, RTX 4070 Ti, RTX 4070 Super, RTX 4070, RTX 4060 Ti, RTX 4060 |

## Ctranslate2 Quantization Compatibility

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
