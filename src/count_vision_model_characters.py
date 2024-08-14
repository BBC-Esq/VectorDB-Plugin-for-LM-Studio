import tkinter as tk
from tkinter import scrolledtext
import re


def analyze_text_chunks(text):
    # Split the text into chunks using a more robust pattern
    chunks = re.split(r'\n+IMG_\d+\.(?:JPG|jpg)\n+', text)[1:]  # Ignore the first empty split
    
    # Process each chunk
    chunk_lengths = []
    for chunk in chunks:
        # Remove any leading/trailing whitespace
        chunk = chunk.strip()
        # Calculate the length of the chunk
        chunk_lengths.append(len(chunk))
    
    # Find the shortest and longest chunks
    shortest_length = min(chunk_lengths)
    longest_length = max(chunk_lengths)
    
    # Find the indices of the shortest and longest chunks
    shortest_index = chunk_lengths.index(shortest_length)
    longest_index = chunk_lengths.index(longest_length)
    
    return {
        'shortest': {
            'length': shortest_length,
            'text': chunks[shortest_index][:100] + '...' if len(chunks[shortest_index]) > 100 else chunks[shortest_index]
        },
        'longest': {
            'length': longest_length,
            'text': chunks[longest_index][:100] + '...' if len(chunks[longest_index]) > 100 else chunks[longest_index]
        }
    }

def process_text():
    input_text = input_area.get("1.0", tk.END)
    result = analyze_text_chunks(input_text)
    
    output_text = f"Shortest chunk ({result['shortest']['length']} characters):\n"
    output_text += result['shortest']['text'] + "\n\n"
    output_text += f"Longest chunk ({result['longest']['length']} characters):\n"
    output_text += result['longest']['text']
    
    output_area.delete("1.0", tk.END)
    output_area.insert(tk.END, output_text)

# Create the main window
root = tk.Tk()
root.title("Text Chunk Analyzer")
root.geometry("800x600")

# Create and pack the input area
input_label = tk.Label(root, text="Paste your text here:")
input_label.pack(pady=5)
input_area = scrolledtext.ScrolledText(root, height=15)
input_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Create and pack the process button
process_button = tk.Button(root, text="Process", command=process_text)
process_button.pack(pady=10)

# Create and pack the output area
output_label = tk.Label(root, text="Results:")
output_label.pack(pady=5)
output_area = scrolledtext.ScrolledText(root, height=10)
output_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Start the GUI event loop
root.mainloop()