import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
from glob import glob
from gui import DocQA_GUI
from server_connector import interact_with_chat

class DocQA_Logic:
    def __init__(self, gui: DocQA_GUI):
        self.gui = gui

        # Connect the buttons to their respective actions
        self.gui.choose_documents_button.config(command=self.choose_documents)
        self.gui.create_chromadb_button.config(command=self.create_chromadb)
        self.gui.submit_query_button.config(command=self.submit_query)

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

        response = messagebox.askokcancel("Proceed?", "Proceeding will (1) delete all documents in the current database folder and then (2) create the ChromaDB vector database. If GPU acceleration is properly set up, you should see CUDA being utilized after the PDFs are processed. You can doublecheck this by going to Task Manager, selecting your GPU, and choosing the 'CUDA' graph from one of the pull-down menus. Once the vector database is created, you can type a query and ask questions of your document(s).")
        if response:
            os.system('python ingest_improved.py')

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
