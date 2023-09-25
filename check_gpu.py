import torch
import tkinter as tk
from tkinter import messagebox

# Create a function to display the information in a popup window
def display_info():
    info_message = ""

    if torch.cuda.is_available():
        info_message += "CUDA is available!\n"
        info_message += "CUDA version: {}\n\n".format(torch.version.cuda)
    else:
        info_message += "CUDA is not available.\n\n"

    if torch.backends.mps.is_available():
        info_message += "Metal/MPS is available!\n\n"
    else:
        info_message += "Metal/MPS is not available.\n\n"

    info_message += "If you want to check the version of Metal and MPS on your macOS device, you can go to \"About This Mac\" -> \"System Report\" -> \"Graphics/Displays\" and look for information related to Metal and MPS.\n\n"

    if torch.version.hip is not None:
        info_message += "ROCm is available!\n"
        info_message += "ROCm version: {}\n".format(torch.version.hip)
    else:
        info_message += "ROCm is not available.\n"

    # Create a small window to display the information
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("System Information", info_message)
    root.destroy()  # Close the hidden main window when the popup is closed

# Call the display_info function to show the information in a window
display_info()
