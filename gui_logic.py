import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Radiobutton, Button
import os
import shutil
from glob import glob
from gui import DocQA_GUI
from server_connector import interact_with_chat
import subprocess
import server_connector

class DownloadModelDialog(simpledialog.Dialog):
    def body(self, master):
        self.model_var = tk.StringVar(value="none_selected")
        self.models = [
            "BAAI/bge-large-en", "BAAI/bge-base-en", "BAAI/bge-small-en", "thenlper/gte-large",
            "thenlper/gte-base", "thenlper/gte-small", "intfloat/e5-large-v2", "intfloat/e5-base-v2",
            "intfloat/e5-small-v2", "hkunlp/instructor-xl", "hkunlp/instructor-large", "hkunlp/instructor-base",
            "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L6-v2"
]
        downloaded_models = [f for f in os.listdir('Embedding_Models') if os.path.isdir(os.path.join('Embedding_Models', f))]
        
        for model in self.models:
            is_downloaded = model in downloaded_models
            Radiobutton(master, text=model, variable=self.model_var, value=model, state=tk.DISABLED if is_downloaded else tk.NORMAL).pack(anchor=tk.W)
        
        return master

    def buttons(self):
        Button(self, text="Download", command=self.ok).pack(side=tk.LEFT)
        Button(self, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)

class DocQA_Logic:
    def __init__(self, gui: DocQA_GUI):
        self.gui = gui
        self.embed_model_name = ""  # Store the selected embedding model name

        # Connect the buttons to their respective actions
        self.gui.download_embedding_model_button.config(command=self.download_embedding_model)
        self.gui.select_embedding_model_button.config(command=self.select_embedding_model_directory)
        self.gui.choose_documents_button.config(command=self.choose_documents)
        self.gui.create_chromadb_button.config(command=self.create_chromadb)
        self.gui.submit_query_button.config(command=self.submit_query)

    def download_embedding_model(self):
        # Creating the "Embedding_Models" folder if it doesn't exist
        if not os.path.exists('Embedding_Models'):
            os.makedirs('Embedding_Models')
    
        # Opening the dialog window
        dialog = DownloadModelDialog(self.gui.root)
        selected_model = dialog.model_var.get()  # this gets the selected model's name
        
        if selected_model:
            # Construct the URL for the Hugging Face model repository
            model_url = f"https://huggingface.co/{selected_model}"
            
            # Define the target directory for the download
            target_directory = os.path.join("Embedding_Models", selected_model.replace("/", "--"))
            
            # Clone the repository using the subprocess module
            subprocess.run(["git", "clone", model_url, target_directory])

    def select_embedding_model_directory(self):
        initial_dir = 'Embedding_Models' if os.path.exists('Embedding_Models') else os.path.expanduser("~")
        chosen_directory = filedialog.askdirectory(initialdir=initial_dir, title="Select Embedding Model Directory")
    
        # Store the chosen directory locally
        if chosen_directory:
            self.embedding_model_directory = chosen_directory

            # Also update the global variable in server_connector.py
            server_connector.EMBEDDING_MODEL_NAME = chosen_directory
        
            # Optionally, you can print or display a confirmation to the user
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
                # Add any additional logic to handle the selected files

    def create_chromadb(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        vector_db_folder = os.path.join(current_dir, "Vector_DB")

        # Check if the "Vector_DB" folder exists, and create it if not
        if not os.path.exists(vector_db_folder):
            os.mkdir(vector_db_folder)

        response = messagebox.askokcancel(
            "Create New Vector Database?",
            "Proceeding will:\n\n" 
            "(1) Delete the current database\n"
            "(2) Create a new ChromaDB vector database.\n\n"
            "If GPU acceleration is properly set up, you will see CUDA being utilized when the database is created. "
            "Check CUDA usage by going to Task Manager, select your GPU, and choosing "
            "the 'CUDA' graph from one of the pull-down menus.\n\n"
            "CUDA usage stops once the vector database is created and then you can ask questions of your docs!"
        )
        
        if response:
            embedding_model_path = getattr(self, "embedding_model_directory", "")
            os.system(f'python ingest_improved.py "{embedding_model_path}"')

    def submit_query(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        vector_db_folder = os.path.join(current_dir, "Vector_DB")
        docs_folder = os.path.join(current_dir, "Docs_for_DB")

        if (not os.path.exists(vector_db_folder) or
            not os.path.exists(docs_folder) or
            not glob(os.path.join(docs_folder, '*.pdf')) or
            len(glob(os.path.join(vector_db_folder, '*.parquet'))) < 2):

            messagebox.showerror("Error", "Must choose documents and create the database first!")
            return

        query = self.gui.text_input.get("1.0", tk.END).strip()
        answer = interact_with_chat(query)
        self.gui.read_only_text.config(state=tk.NORMAL)
        self.gui.read_only_text.delete("1.0", tk.END)
        self.gui.read_only_text.insert(tk.END, answer)
        self.gui.read_only_text.config(state=tk.DISABLED)
