| API | Backend Selection | Discrete GPU + Vulkan? | iGPU + Vulkan? | Device ID Determination |
|-----|-------------------|-------------------------|------------|-------------------------|
| .kcppt | Auto selects CUDA, then Vulkan, then CPU. | Yes. | No. | Automatic |
| .kcpps | Onlky uses the specified backend. | Yes. | Yes. | Will use the specified ID, or if it's an empty list, it'll autoselect (including iGPUs) |
| --usevulkan | Force Vulkan if a compatible device is available; otherwise, cpu. | Yes. | Yes. | Can be specified after the flag, or if not, autoselects. |
