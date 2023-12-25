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
Print GPU memory stats in script
def print_cuda_memory():
    if ENABLE_CUDA_PRINT:
        max_allocated_memory = torch.cuda.max_memory_allocated()
        memory_allocated = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()

        my_cprint(f"Max CUDA memory allocated: {max_allocated_memory / (1024**2):.2f} MB", "green")
        my_cprint(f"Total CUDA memory allocated: {memory_allocated / (1024**2):.2f} MB", "yellow")
        my_cprint(f"Total CUDA memory reserved: {reserved_memory / (1024**2):.2f} MB", "yellow")
        print_cuda_memory()
'''