import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from constants import GPUS_AMD, GPUS_NVIDIA, GPUS_INTEL
import numpy as np

def create_gpu_comparison_plot(min_vram_threshold=6, max_vram_threshold=8):
    # Filter and process AMD data
    filtered_amd = {k: v for k, v in GPUS_AMD.items() if min_vram_threshold <= int(str(v["Size (GB)"]).split('-')[0]) <= max_vram_threshold}
    sorted_amd = sorted(filtered_amd.items(), key=lambda item: item[1]["Shaders"], reverse=True)
    names_amd = [item[0] for item in sorted_amd]
    shaders_amd = [item[1]["Shaders"] for item in sorted_amd]
    sizes_amd = [int(str(item[1]["Size (GB)"]).split('-')[0]) for item in sorted_amd]

    # Filter and process NVIDIA data
    filtered_nvidia = {k: v for k, v in GPUS_NVIDIA.items() if min_vram_threshold <= int(str(v["Size (GB)"]).split('-')[0]) <= max_vram_threshold}
    sorted_nvidia = sorted(filtered_nvidia.items(), key=lambda item: item[1]["CUDA Cores"], reverse=True)
    names_nvidia = [item[0] for item in sorted_nvidia]
    cuda_cores_nvidia = [item[1]["CUDA Cores"] for item in sorted_nvidia]
    sizes_nvidia = [int(str(item[1]["Size (GB)"]).split('-')[0]) for item in sorted_nvidia]

    # Filter and process Intel data
    filtered_intel = {k: v for k, v in GPUS_INTEL.items() if min_vram_threshold <= v["Size (GB)"] <= max_vram_threshold}
    sorted_intel = sorted(filtered_intel.items(), key=lambda item: item[1]["Shading Cores"], reverse=True)
    names_intel = [item[0] for item in sorted_intel]
    shading_cores_intel = [item[1]["Shading Cores"] for item in sorted_intel]
    sizes_intel = [item[1]["Size (GB)"] for item in sorted_intel]

    names = names_amd + names_nvidia + names_intel
    sizes = sizes_amd + sizes_nvidia + sizes_intel
    compute_units = shaders_amd + cuda_cores_nvidia + shading_cores_intel

    # Create gradients
    gradient_amd = LinearSegmentedColormap.from_list("", ["#990000", "#FF9999"])
    gradient_nvidia = LinearSegmentedColormap.from_list("", ["#003328", "#00CC66"])
    gradient_intel = LinearSegmentedColormap.from_list("", ["#001f4d", "#0066cc"])

    # Setup plot
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#4A4A4A')
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('#4A4A4A')

    # Create bars
    bars_amd = ax1.barh(names_amd, shaders_amd, color=gradient_amd(np.linspace(0, 1, len(shaders_amd))), label='AMD Shaders')
    bars_nvidia = ax1.barh(names_nvidia, cuda_cores_nvidia, color=gradient_nvidia(np.linspace(0, 1, len(cuda_cores_nvidia))), label='NVIDIA CUDA Cores')
    bars_intel = ax1.barh(names_intel, shading_cores_intel, color=gradient_intel(np.linspace(0, 1, len(shading_cores_intel))), label='Intel Shading Cores')

    # Add text to bars
    max_shaders_amd = max(shaders_amd) if shaders_amd else 0
    max_cuda_cores_nvidia = max(cuda_cores_nvidia) if cuda_cores_nvidia else 0
    max_shading_cores_intel = max(shading_cores_intel) if shading_cores_intel else 0

    for i, bar in enumerate(bars_amd):
        percentage = (shaders_amd[i] / max_shaders_amd) * 100
        ax1.text(150, bar.get_y() + bar.get_height() / 2,
                 f'{shaders_amd[i]:,} - {percentage:.2f}%', va='center', ha='left', color='white', fontsize=10)

    for i, bar in enumerate(bars_nvidia):
        percentage = (cuda_cores_nvidia[i] / max_cuda_cores_nvidia) * 100
        ax1.text(150, bar.get_y() + bar.get_height() / 2,
                 f'{cuda_cores_nvidia[i]:,} - {percentage:.2f}%', va='center', ha='left', color='white', fontsize=10)

    for i, bar in enumerate(bars_intel):
        percentage = (shading_cores_intel[i] / max_shading_cores_intel) * 100
        ax1.text(150, bar.get_y() + bar.get_height() / 2,
                 f'{shading_cores_intel[i]:,} - {percentage:.2f}%', va='center', ha='left', color='white', fontsize=10)

    # Set labels and title
    ax1.set_xlabel('Compute Units (Shaders/CUDA Cores/Shading Cores)', color='white')
    ax1.set_ylabel('Graphics Cards', color='white', labelpad=15)
    ax1.set_title(f'Graphics Cards: Compute Units and VRAM Comparison ({min_vram_threshold}GB <= VRAM <= {max_vram_threshold}GB)', color='white', pad=20)
    ax1.tick_params(axis='both', colors='white')

    # Create second x-axis for VRAM
    ax2 = ax1.twiny()
    ax2.plot(sizes, names, 'o-', color='orange', label='VRAM (GB)')
    ax2.set_xlabel('VRAM (GB)', color='white')
    ax2.xaxis.set_label_position('bottom')
    ax2.xaxis.tick_bottom()
    ax2.tick_params(axis='x', colors='white')

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    # Create custom legend elements
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=gradient_amd(0.5), edgecolor='none', label='AMD Shaders'),
        plt.Rectangle((0,0),1,1, facecolor=gradient_nvidia(0.5), edgecolor='none', label='NVIDIA CUDA Cores'),
        plt.Rectangle((0,0),1,1, facecolor=gradient_intel(0.5), edgecolor='none', label='Intel Shading Cores'),
        plt.Line2D([0], [0], color='orange', marker='o', linestyle='-', label='VRAM (GB)')
    ]

    ax2.legend(handles=legend_elements, loc='upper right', facecolor='#4A4A4A', edgecolor='white', labelcolor='white')

    # Set spine colors
    for spine in ax1.spines.values():
        spine.set_edgecolor('white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')

    # Add VRAM lines
    vram_lines = [2, 4, 6, 8, 10, 11, 12, 16, 20, 24, 32]
    for vram_value in vram_lines:
        if vram_value in sizes:
            ax2.axvline(x=vram_value, color='#A8A8A8', linestyle='--', linewidth=0.5)

    ax2.set_xticks(vram_lines)
    ax2.set_xlim(0, 33)

    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
    
    return fig

# If you want to run this script standalone, you can use:
if __name__ == "__main__":
    fig = create_gpu_comparison_plot(12, 24)
    plt.show()