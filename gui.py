import tkinter as tk
from tkinter import font as tkfont
from gui_table import create_table
import threading
from nvml import CudaVramLogic
import torch
import yaml

def determine_compute_device():
    if torch.cuda.is_available():
        COMPUTE_DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        COMPUTE_DEVICE = "mps"
    else:
        COMPUTE_DEVICE = "cpu"
    
    with open("config.yaml", 'r') as stream:
        config_data = yaml.safe_load(stream)
    config_data['COMPUTE_DEVICE'] = COMPUTE_DEVICE
    with open("config.yaml", 'w') as stream:
        yaml.safe_dump(config_data, stream)

class DocQA_GUI:
    def __init__(self, root):
        self.root = root

        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=1)

        left_frame = tk.Frame(main_pane)

        self.download_embedding_model_button = tk.Button(left_frame, text="Download Embedding Model", width=26)
        self.download_embedding_model_button.pack(pady=5)

        self.select_embedding_model_button = tk.Button(left_frame, text="Select Embedding Model Directory", width=26)
        self.select_embedding_model_button.pack(pady=5)

        self.choose_documents_button = tk.Button(left_frame, text="Choose Documents for Database", width=26)
        self.choose_documents_button.pack(pady=5)

        self.create_chromadb_button = tk.Button(left_frame, text="Create Vector Database", width=26)
        self.create_chromadb_button.pack(pady=5)

        create_table(left_frame)
        
        # GPU label
        self.gpu_info_label = tk.Label(left_frame, font=("Segoe UI Semibold", 16), foreground='green')
        self.gpu_info_label.pack(pady=1)
        
        # VRAM label
        self.vram_info_label = tk.Label(left_frame, font=("Segoe UI Semibold", 16), foreground='blue')
        self.vram_info_label.pack(pady=1)

        # Adjust CudaVramLogic initialization:
        self.cuda_logic = CudaVramLogic(self.vram_info_label, self.gpu_info_label, self.root)

        main_pane.add(left_frame)

        right_frame = tk.Frame(main_pane)
        main_pane.add(right_frame)

        middle_frame = tk.Frame(right_frame)
        middle_frame.pack(pady=5, fill=tk.BOTH, expand=1)

        self.text_input = tk.Text(middle_frame, wrap=tk.WORD, height=5)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.text_input.configure(font=("Segoe UI Historic", 10))

        scroll1 = tk.Scrollbar(middle_frame, command=self.text_input.yview)
        scroll1.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_input.config(yscrollcommand=scroll1.set)

        self.submit_query_button = tk.Button(right_frame, text="Submit Question", width=15)
        self.submit_query_button.pack(pady=5, side=tk.TOP)

        bottom_frame = tk.Frame(right_frame)
        bottom_frame.pack(pady=5, fill=tk.BOTH, expand=1)

        self.read_only_text = tk.Text(bottom_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.read_only_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.read_only_text.configure(font=("Segoe UI Historic", 12))

        scroll2 = tk.Scrollbar(bottom_frame, command=self.read_only_text.yview)
        scroll2.pack(side=tk.RIGHT, fill=tk.Y)
        self.read_only_text.config(yscrollcommand=scroll2.set)

        self.center_window(root)

    def center_window(self, root):
        root.withdraw()
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        root.deiconify()

if __name__ == "__main__":
    determine_compute_device()
    root = tk.Tk()
    root.title("Welcome to the LM Studio ChromaDB Plugin!")
    root.geometry("800x800")
    app = DocQA_GUI(root)
    from gui_logic import DocQA_Logic
    logic = DocQA_Logic(app)
    root.protocol("WM_DELETE_WINDOW", app.cuda_logic.stop_and_exit)
    root.mainloop()
