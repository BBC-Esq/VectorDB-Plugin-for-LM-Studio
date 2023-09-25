import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Radiobutton, Button
import os
import shutil
from glob import glob
import yaml
from gui import DocQA_GUI
from server_connector import interact_with_chat
import subprocess
import server_connector
import threading
import multiprocessing

def load_config():
    with open("config.yaml", 'r') as stream:
        return yaml.safe_load(stream)

class DownloadModelDialog(simpledialog.Dialog):
    def body(self, master):
        self.model_var = tk.StringVar(value="none_selected")
        
        config_data = load_config()
        self.models = config_data["AVAILABLE_MODELS"]
        
        downloaded_models = [f for f in os.listdir('Embedding_Models') if os.path.isdir(os.path.join('Embedding_Models', f))]
        
        for model in self.models:
            is_downloaded = model in downloaded_models
            Radiobutton(master, text=model, variable=self.model_var, value=model, state=tk.DISABLED if is_downloaded else tk.NORMAL).pack(anchor=tk.W)
        
        return master

    def buttons(self):
        Button(self, text="Download", command=self.ok).pack(side=tk.LEFT)
        Button(self, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)
            
def run_create_chromadb(embedding_model_path):
    os.system(f'python ingest_improved.py "{embedding_model_path}"')

class DocQA_Logic:
    def __init__(self, gui: DocQA_GUI):
        self.gui = gui
        
        config_data = load_config()
        self.embed_model_name = config_data.get("EMBEDDING_MODEL_NAME", "")
        
        self.gui.download_embedding_model_button.config(command=self.download_embedding_model)
        self.gui.select_embedding_model_button.config(command=self.select_embedding_model_directory)
        self.gui.choose_documents_button.config(command=self.choose_documents)
        self.gui.create_chromadb_button.config(command=self.create_chromadb)
        self.gui.submit_query_button.config(command=self.submit_query)

    def download_embedding_model(self):
        if not os.path.exists('Embedding_Models'):
            os.makedirs('Embedding_Models')
    
        dialog = DownloadModelDialog(self.gui.root)
        selected_model = dialog.model_var.get()
        
        if selected_model:
            model_url = f"https://huggingface.co/{selected_model}"
            target_directory = os.path.join("Embedding_Models", selected_model.replace("/", "--"))
            
            def download_model():
                subprocess.run(["git", "clone", model_url, target_directory])
                
            download_thread = threading.Thread(target=download_model)
            download_thread.start()

    def select_embedding_model_directory(self):
        initial_dir = 'Embedding_Models' if os.path.exists('Embedding_Models') else os.path.expanduser("~")
        chosen_directory = filedialog.askdirectory(initialdir=initial_dir, title="Select Embedding Model Directory")
    
        if chosen_directory:
            self.embedding_model_directory = chosen_directory
            self.embed_model_name = chosen_directory
            
            # Update the config.yaml file with the chosen model directory
            config_data = load_config()
            config_data["EMBEDDING_MODEL_NAME"] = chosen_directory
            with open("config.yaml", 'w') as file:
                yaml.dump(config_data, file)

            print(f"Selected directory: {chosen_directory}")

    def choose_documents(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        docs_folder = os.path.join(current_dir, "Docs_for_DB")
        file_paths = filedialog.askopenfilenames(initialdir=current_dir)

        if file_paths:
            if not os.path.exists(docs_folder):
                os.mkdir(docs_folder)

            for file_path in file_paths:
                shutil.copy(file_path, docs_folder)

    def create_chromadb(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        vector_db_folder = os.path.join(current_dir, "Vector_DB")

        if not os.path.exists(vector_db_folder):
            os.mkdir(vector_db_folder)

        response = messagebox.askokcancel(
            "Create Vector Database?",
            "This will overwrite any current databases!"
        )
        
        if response:
            embedding_model_path = getattr(self, "embedding_model_directory", "")
            
            create_chromadb_process = multiprocessing.Process(target=run_create_chromadb, args=(embedding_model_path,))
            create_chromadb_process.start()

    def submit_query(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        vector_db_folder = os.path.join(current_dir, "Vector_DB")
        docs_folder = os.path.join(current_dir, "Docs_for_DB")

        valid_extensions = ['.pdf', '.docx', '.txt', '.json', '.enex', '.eml', '.msg', '.csv', '.xls', '.xlsx']
        files_present = any(glob(os.path.join(docs_folder, f'*{ext}')) for ext in valid_extensions)

        if (not os.path.exists(vector_db_folder) or
            not os.path.exists(docs_folder) or
            not files_present or
            len(glob(os.path.join(vector_db_folder, '*.parquet'))) < 2):

            messagebox.showerror("Error", "Must choose documents and create the database first!")
            return

        query = self.gui.text_input.get("1.0", tk.END).strip()

        def interact_with_chat_and_update_gui(query):
            answer = interact_with_chat(query)
            self.gui.read_only_text.config(state=tk.NORMAL)
            self.gui.read_only_text.delete("1.0", tk.END)
            self.gui.read_only_text.insert(tk.END, answer)
            self.gui.read_only_text.config(state=tk.DISABLED)

        chat_thread = threading.Thread(target=interact_with_chat_and_update_gui, args=(query,))
        chat_thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = DocQA_GUI(root)
    logic = DocQA_Logic(app)
    root.mainloop()