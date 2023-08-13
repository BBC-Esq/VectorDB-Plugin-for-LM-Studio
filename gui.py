import tkinter as tk

class DocQA_GUI:
    def __init__(self, root):
        self.file_path = tk.StringVar()

        # Top Section: Document Selection and Actions
        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)

        self.choose_documents_button = tk.Button(top_frame, text="Choose Documents", width=15)
        self.choose_documents_button.pack(side=tk.LEFT, padx=5)

        self.create_chromadb_button = tk.Button(top_frame, text="Create ChromaDB", width=15)
        self.create_chromadb_button.pack(side=tk.LEFT, padx=5)

        self.submit_query_button = tk.Button(top_frame, text="Submit Query", width=15)
        self.submit_query_button.pack(side=tk.LEFT, padx=5)

        # Middle Section: Text Input and Control
        middle_frame = tk.Frame(root)
        middle_frame.pack(pady=5)

        self.text_input = tk.Text(middle_frame, wrap=tk.WORD, height=5)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH)
        self.text_input.configure(font=("Segoe UI Historic", 10))
        self.set_placeholder() # Set the initial placeholder text

        scroll1 = tk.Scrollbar(middle_frame, command=self.text_input.yview)
        scroll1.pack(side=tk.LEFT, fill=tk.Y)
        self.text_input.config(yscrollcommand=scroll1.set)

        # Bottom Section: Text Output and Actions
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        self.read_only_text = tk.Text(bottom_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.read_only_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.read_only_text.configure(font=("Segoe UI Historic", 12))

        scroll2 = tk.Scrollbar(bottom_frame, command=self.read_only_text.yview)
        scroll2.pack(side=tk.LEFT, fill=tk.Y)
        self.read_only_text.config(yscrollcommand=scroll2.set)

        # Center the window and display it
        self.center_window(root)

    def set_placeholder(self):
        self.text_input.insert(tk.END, 'Enter text here...')

    def center_window(self, root):
        root.withdraw()  # Hide the window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        root.deiconify()  # Show the window

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Welcome to the LM Studio ChromaDB Plugin!")
    root.geometry("415x500")
    app = DocQA_GUI(root)
    from gui_logic import DocQA_Logic
    logic = DocQA_Logic(app)
    root.mainloop()
