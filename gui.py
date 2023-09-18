import tkinter as tk
from tkinter import font as tkfont
import threading
import torch
import yaml

from gui_table import create_table, create_pro_tip
from metrics_gpu import GPU_Monitor
from metrics_system import SystemMonitor

import platform

LABEL_FONT = ("Segoe UI Semibold", 12)

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
    
    return COMPUTE_DEVICE

def is_nvidia_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "nvidia" in gpu_name.lower()
    return False

class DocQA_GUI:
    def __init__(self, root):
        self.root = root

        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=1)

        # LEFT FRAME
        left_frame = tk.Frame(main_pane)
        
        # 1. BUTTONS
        self.download_embedding_model_button = tk.Button(left_frame, text="Download Embedding Model", width=26)
        self.download_embedding_model_button.pack(pady=5)

        self.select_embedding_model_button = tk.Button(left_frame, text="Select Embedding Model Directory", width=26)
        self.select_embedding_model_button.pack(pady=5)

        self.choose_documents_button = tk.Button(left_frame, text="Choose Documents for Database", width=26)
        self.choose_documents_button.pack(pady=5)

        self.create_chromadb_button = tk.Button(left_frame, text="Create Vector Database", width=26)
        self.create_chromadb_button.pack(pady=5)

        # 2. TABLE
        create_table(left_frame)

        # 3. PRO TIP
        create_pro_tip(left_frame)

        # 4. METRICS
        self.gpu_info_label = tk.Label(left_frame, font=LABEL_FONT, foreground='green')
        self.gpu_info_label.pack(pady=0, padx=1)

        self.vram_info_label = tk.Label(left_frame, font=LABEL_FONT, foreground='green')
        self.vram_info_label.pack(pady=0, padx=1)

        self.ram_usage_label = tk.Label(left_frame, font=LABEL_FONT, foreground='medium blue')
        self.ram_usage_label.pack(pady=0, padx=1)

        self.ram_used_label = tk.Label(left_frame, font=LABEL_FONT, foreground='medium blue') 
        self.ram_used_label.pack(pady=0, padx=1)

        self.cpu_usage_label = tk.Label(left_frame, font=LABEL_FONT, foreground='violet red')
        self.cpu_usage_label.pack(pady=0, padx=1)
        
        compute_device = determine_compute_device()
        os_name = platform.system().lower()

        if compute_device != "mps" and os_name == "windows" and is_nvidia_gpu():
            self.cuda_logic = GPU_Monitor(self.vram_info_label, self.gpu_info_label, self.root)
            self.system_monitor = SystemMonitor(self.cpu_usage_label, self.ram_used_label, self.ram_usage_label, self.root)
        else:
            self.cuda_logic = None
            self.system_monitor = None

        main_pane.add(left_frame)

        # RIGHT FRAME
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

    def center_window(self, root):
        root.withdraw()
        root.update_idletasks()

        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)

        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        root.deiconify()
  
    def stop_and_exit(self):
        if self.cuda_logic:
            self.cuda_logic.stop_and_exit_gpu_monitor()
        if self.system_monitor:
            self.system_monitor.stop_and_exit_system_monitor()
        self.root.quit()
        self.root.destroy()

    def start_up(self):
        self.center_window(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.stop_and_exit)
        self.root.mainloop()

if __name__ == "__main__":
    from gui_logic import DocQA_Logic
    root = tk.Tk()
    root.title("Version 1.4.2 - www.chintellalaw.com")
    root.geometry("825x870")
    root.minsize(825, 870)

    app = DocQA_GUI(root)
    logic = DocQA_Logic(app)
    app.start_up()
