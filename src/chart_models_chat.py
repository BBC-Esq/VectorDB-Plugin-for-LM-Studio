import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

from constants import CHAT_MODELS

def create_chat_models_comparison_plot():
    model_categories = {
        "coding": {
            "models": [
                "Qwen Coder - 7b",
                "Qwen Coder - 14b",
                "Qwen Coder - 32b"
            ],
            "color": "#DAA520",
            "label": "Coding Focused (8k context)"
        },
        "long_context": {
            "models": [
                # "Granite - 2b",
                "Exaone - 2.4b",
                "Granite - 8b",
                "Exaone - 7.8b",
                "Deepseek R1 - 7b",
                "InternLM 3 - 8b",
                "Deepseek R1 - 14b",
                "Qwen - 14b",
                "Mistral Small - 22b",
                "Qwen - 32b",
                "Deepseek R1 - 32b",
                "Exaone - 32b",
            ],
            "color": "#2E8B57",
            "label": "8k Context"
        },
        "coding_standard_context": {
            "models": [
                "Qwen Coder - 1.5b",
                "Qwen Coder - 3b",
            ],
            "color": "#4682B4",
            "label": "Coding Focused (4k context)"
        }
    }

    df = pd.DataFrame([
        {"model": model, "cps": data["cps"], "vram": data["vram"] / 1024}  # Convert MB to GB
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
    ax1.set_ylabel("Average VRAM Usage (GB)", color="white", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="white", colors="white")
    ax1.tick_params(axis="x", labelcolor="white", colors="white", rotation=45)

    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='white')

    ax1.set_xticks(range(len(df)))

    model_names = df["model"]
    ax1.set_xticklabels(model_names, rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', 
                 verticalalignment='bottom', color='white', ha='center')

    line = ax2.plot(range(len(df)), df["cps"], color="#6699CC", marker="D", markersize=6, linewidth=2, label="Characters per Second")
    ax2.set_ylabel("Characters per Second", color="white", fontsize=14)
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