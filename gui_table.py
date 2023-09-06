import tkinter as tk

def create_table(parent_frame):
    models = [
        "BAAI/bge-large-en", "BAAI/bge-base-en", "BAAI/bge-small-en",
        "thenlper/gte-large", "thenlper/gte-base", "thenlper/gte-small",
        "intfloat/e5-large-v2", "intfloat/e5-base-v2", "intfloat/e5-small-v2",
        "hkunlp/instructor-xl", "hkunlp/instructor-large", "hkunlp/instructor-base",
        "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L6-v2"
    ]
    vram_values = [
        "5.3GB", "3.7GB", "2.9GB", "5.3GB", "3.7GB", "3GB", "5.2GB", "3.7GB", "2.9GB",
        "18.1GB", "6.8GB", "4.6GB", "2.7GB", "1.6GB", "1.6GB"
    ]

    table_frame = tk.Frame(parent_frame)
    table_frame.pack(pady=5, fill=tk.BOTH, expand=1)

    tk.Label(table_frame, text="Embedding Model", borderwidth=1, relief="solid").grid(row=0, column=0, sticky="nsew")
    tk.Label(table_frame, text="Estimated VRAM", borderwidth=1, relief="solid").grid(row=0, column=1, sticky="nsew")

    for i, (model, vram) in enumerate(zip(models, vram_values), start=1):
        tk.Label(table_frame, text=model, borderwidth=1, relief="solid").grid(row=i, column=0, sticky="nsew")
        tk.Label(table_frame, text=vram, borderwidth=1, relief="solid").grid(row=i, column=1, sticky="nsew")

    table_frame.grid_columnconfigure(0, weight=1)
    table_frame.grid_columnconfigure(1, weight=1)

    pro_tip_label = tk.Label(parent_frame, text="Pro tip:", font=("Segoe UI Historic", 12, "bold"))
    pro_tip_label.pack(pady=(20, 0), anchor="w", padx=5, side=tk.TOP)

    pro_tip_text = (
        "DO NOT have LM Studio running when creating the vector database. The VRAM numbers above refer to when creating "
        "the database. After it's created, run LM Studio and load your LLM (remember only Llama2-based models work "
        "currently when querying the database). To query the database, the embedding model will use about half the VRAM "
        "it used when creating it. Use the LARGEST embedding model you can possibly fit into VRAM while the LLM is loaded "
        "into LM Studio (remembering the half rule above). The quality of the embedding model is ACTUALLY MORE important "
        "than the size of the LLM. Experiment with low-quality LLMs and high-quality embedding models. EXAMPLE: q3_k_3 "
        "model + instructor-xl worked just fine together."
    )

    pro_tip_description = tk.Label(parent_frame, text=pro_tip_text, wraplength=400, justify="left")
    pro_tip_description.pack(anchor="w", padx=5, side=tk.TOP)
