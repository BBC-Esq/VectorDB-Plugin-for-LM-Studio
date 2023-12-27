from pathlib import Path
from PySide6.QtWidgets import QApplication

def validate_symbolic_links(source_directory):
    source_path = Path(source_directory)
    symbolic_links = [entry for entry in source_path.iterdir() if entry.is_symlink()]

    for symlink in symbolic_links:
        target_path = symlink.resolve(strict=False)

        if not target_path.exists():
            print(f"Warning: Symbolic link {symlink.name} points to a missing file. It will be skipped.")
            symlink.unlink()

def load_stylesheet(filename):
    script_dir = Path(__file__).parent
    stylesheet_path = script_dir / 'CSS' / filename
    with stylesheet_path.open('r') as file:
        stylesheet = file.read()
    return stylesheet

def list_theme_files():
    script_dir = Path(__file__).parent
    theme_dir = script_dir / 'CSS'
    return [f.name for f in theme_dir.iterdir() if f.suffix == '.css']

def make_theme_changer(theme_name):
    def change_theme():
        stylesheet = load_stylesheet(theme_name)
        QApplication.instance().setStyleSheet(stylesheet)
    return change_theme

if __name__ == '__main__':
    source_directory = "Docs_for_DB"
    validate_symbolic_links(source_directory)

    
'''
# Print GPU memory stats in script
def print_cuda_memory_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # NVML memory information
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Memory Total: {memory_info.total / 1024**2} MB") 
        print(f"Memory Used: {memory_info.used / 1024**2} MB")
        print(f"Memory Free: {memory_info.free / 1024**2} MB")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        pynvml.nvmlShutdown()

# Check for references to an object when trying to clear memory
script_dir = os.path.dirname(__file__)
referrers_file_path = os.path.join(script_dir, "references.txt")

with open(referrers_file_path, "w") as file:
    referrers = gc.get_referrers(model)
    file.write(f"Number of references found: {len(referrers)}\n")
    for ref in referrers:
        file.write(str(ref) + "\n")
'''