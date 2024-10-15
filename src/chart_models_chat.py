import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np

from constants import CHAT_MODELS

def create_chat_models_comparison_plot():
    model_categories = {
        "coding": {
            "models": [
                "DeepSeek Coder v2 - 16b",
                "Yi Coder - 9b",
                "Qwen 2.5 Coder - 7b",
            ],
            "color": "#DAA520",
            "label": "Coding Focused & 8k Context"
        },
        "long_context": {
            "models": [
                "Phi 3.5 Mini - 4b",
                "Qwen 2.5 - 7b",
                "Internlm2_5 - 7b",
                "Yi Coder - 9b",
                "Dolphin-Llama 3.1 - 8b",
                "Yi-9b-16k-BNB",
                "Qwen 2.5 - 14b",
                "Mistral Small - 22b",
                "Internlm2_5 - 20b",
            ],
            "color": "#CD5C5C",
            "label": "8k Context"
        },
    }

    df = pd.DataFrame([
        {"model": model, "cps": data["cps"], "vram": data["vram"]}
        for model, data in CHAT_MODELS.items()
    ])
    df = df.sort_values(by="vram")

    plt.rcParams['font.family'] = 'Arial'

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#2e2e2e')
    ax1.set_facecolor('#2e2e2e')

    ax1.set_title("BitsAndBytes (4-bit); RTX 4090; num_beams=1; max context=4096", fontsize=16, color='white', pad=5)

    ax2 = ax1.twinx()

    gradient = LinearSegmentedColormap.from_list("", ["#001f4d", "#0066cc"])

    bars = []
    for i, (index, row) in enumerate(df.iterrows()):
        border_color = None
        border_width = 0
        for category in model_categories.values():
            if row["model"] in category["models"]:
                border_color = category["color"]
                border_width = 3
                break
        
        bar = ax1.bar(i, row["vram"], color=gradient(i/len(df)), alpha=0.7, 
                      edgecolor=border_color, linewidth=border_width)
        bars.append(bar[0])

    ax1.bar(0, 0, color='none', label="VRAM Usage")

    ax1.set_xlabel("Model", color="white")
    ax1.set_ylabel("Average VRAM Usage (MB)", color="white")
    ax1.tick_params(axis="y", labelcolor="white", colors="white")
    ax1.tick_params(axis="x", labelcolor="white", colors="white", rotation=45)

    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='white')

    ax1.set_xticks(range(len(df)))

    model_names = df["model"]
    ax1.set_xticklabels(model_names, rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', verticalalignment='bottom', color='white', ha='center')

    line = ax2.plot(range(len(df)), df["cps"], color="#6699CC", marker="D", markersize=6, linewidth=2, label="Characters per Second")
    ax2.set_ylabel("Characters per Second", color="white")
    ax2.tick_params(axis="y", labelcolor="white")

    for i, cps in enumerate(df["cps"]):
        ax2.annotate(f'{cps:.2f}', (i, cps), textcoords="offset points", xytext=(0,10), ha='center', color='white', fontweight='bold')

    category_patches = [Patch(facecolor='none', edgecolor=cat["color"], label=cat["label"], linewidth=2) 
                        for cat in model_categories.values()]

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_handles = lines1 + lines2 + category_patches
    all_labels = labels1 + labels2 + [cat["label"] for cat in model_categories.values()]
    ax1.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), 
               fancybox=True, shadow=True, ncol=len(all_handles))

    fig.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.85, bottom=0.15)
    
    return fig

if __name__ == "__main__":
    # This block will only run if the script is executed directly
    fig = create_chat_models_comparison_plot()
    plt.show()