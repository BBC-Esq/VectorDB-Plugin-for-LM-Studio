import tkinter as tk

class DocQA_GUI:
    def __init__(self, root):
        self.root = root  # Store the root window for later access
        self.file_path = tk.StringVar()

        # Use a PanedWindow to manage the left buttons and the right text frames
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=1)

        # Left Section: Buttons
        left_frame = tk.Frame(main_pane)

        self.download_embedding_model_button = tk.Button(left_frame, text="Download Embedding Model", width=26)
        self.download_embedding_model_button.pack(pady=5)

        self.select_embedding_model_button = tk.Button(left_frame, text="Select Embedding Model Directory", width=26)
        self.select_embedding_model_button.pack(pady=5)

        self.choose_documents_button = tk.Button(left_frame, text="Choose Documents for Database", width=26)
        self.choose_documents_button.pack(pady=5)

        self.create_chromadb_button = tk.Button(left_frame, text="Create Vector Database", width=26)
        self.create_chromadb_button.pack(pady=5)

        # Create table below the buttons
        self.create_table(left_frame)

        main_pane.add(left_frame)

        # Middle and Bottom Sections: Text Input and Output
        right_frame = tk.Frame(main_pane)
        main_pane.add(right_frame)

        # Middle Section: Text Input and Control
        middle_frame = tk.Frame(right_frame)
        middle_frame.pack(pady=5, fill=tk.BOTH, expand=1)

        self.text_input = tk.Text(middle_frame, wrap=tk.WORD, height=5)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.text_input.configure(font=("Segoe UI Historic", 10))
        self.set_placeholder()  # Set the initial placeholder text

        scroll1 = tk.Scrollbar(middle_frame, command=self.text_input.yview)
        scroll1.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_input.config(yscrollcommand=scroll1.set)

        # Button between Middle and Bottom
        self.submit_query_button = tk.Button(right_frame, text="Submit Question", width=15)
        self.submit_query_button.pack(pady=5, side=tk.TOP)

        # Bottom Section: Text Output and Actions
        bottom_frame = tk.Frame(right_frame)
        bottom_frame.pack(pady=5, fill=tk.BOTH, expand=1)

        self.read_only_text = tk.Text(bottom_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.read_only_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.read_only_text.configure(font=("Segoe UI Historic", 12))

        scroll2 = tk.Scrollbar(bottom_frame, command=self.read_only_text.yview)
        scroll2.pack(side=tk.RIGHT, fill=tk.Y)
        self.read_only_text.config(yscrollcommand=scroll2.set)

        # Center the window and display it
        self.center_window(root)

    def set_placeholder(self):
        self.text_input.insert(tk.END, 'Enter question here...')

    def center_window(self, root):
        root.withdraw()  # Hide the window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        root.deiconify()  # Show the window

    def create_table(self, parent_frame):
        # Define the models and their corresponding VRAM values
        models = ["BAAI/bge-large-en", "BAAI/bge-base-en", "BAAI/bge-small-en", "thenlper/gte-large",
                  "thenlper/gte-base", "thenlper/gte-small", "intfloat/e5-large-v2", "intfloat/e5-base-v2",
                  "intfloat/e5-small-v2", "hkunlp/instructor-xl", "hkunlp/instructor-large", "hkunlp/instructor-base",
                  "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L6-v2"]
        vram_values = ["5.3GB", "3.7GB", "2.9GB", "5.3GB", "3.7GB", "3GB", "5.2GB", "3.7GB", "2.9GB",
                       "18.1GB", "6.8GB", "4.6GB", "2.7GB", "1.6GB", "1.6GB"]  # Placeholder values

        # Table frame
        table_frame = tk.Frame(parent_frame)
        table_frame.pack(pady=5, fill=tk.BOTH, expand=1)

        # Header
        tk.Label(table_frame, text="Embedding Model", borderwidth=1, relief="solid").grid(row=0, column=0, sticky="nsew")
        tk.Label(table_frame, text="Estimated VRAM", borderwidth=1, relief="solid").grid(row=0, column=1, sticky="nsew")

        # Content
        for i, (model, vram) in enumerate(zip(models, vram_values), start=1):
            tk.Label(table_frame, text=model, borderwidth=1, relief="solid").grid(row=i, column=0, sticky="nsew")
            tk.Label(table_frame, text=vram, borderwidth=1, relief="solid").grid(row=i, column=1, sticky="nsew")

        # Adjusting column weights so they expand equally
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_columnconfigure(1, weight=1)

        # Add Pro Tip and accompanying text
        pro_tip_label = tk.Label(parent_frame, text="Pro tip:", font=("Segoe UI Historic", 12, "bold"))
        pro_tip_label.pack(pady=(20, 0), anchor="w", padx=5, side=tk.TOP)

        pro_tip_text = ("DO NOT have LM Studio running when creating the vector database.  The VRAM numbers above refer to when creating the database. "
                        "After it's created, run LM Studio and load your LLM (remember only Llama2-based models work currently when querying the database). "
                        "To query the database, the embedding model will use about half the VRAM it used when creating it.  Use the LARGEST embedding "
                        "model you can possibly fit into VRAM while the LLM is loaded into LM Studio (remembering the half rule above).  The quality of the "
                        "embedding model is ACTUALLY MORE important that the size of the LLM.  Experiment with low-quality LLMs and high-quality embedding models. "
                        "EXAMPLE: q3_k_3 model + instructor-xl worked just fine together.")

        pro_tip_description = tk.Label(parent_frame, text=pro_tip_text, wraplength=400, justify="left")
        pro_tip_description.pack(anchor="w", padx=5, side=tk.TOP)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Welcome to the LM Studio ChromaDB Plugin!")
    root.geometry("800x700")  # Adjust the size slightly for the paned layout
    app = DocQA_GUI(root)
    from gui_logic import DocQA_Logic
    logic = DocQA_Logic(app)
    root.mainloop()
