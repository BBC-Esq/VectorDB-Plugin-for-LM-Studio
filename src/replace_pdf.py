import shutil
from pathlib import Path
import hashlib

def find_all_target_directories_with_file(base_path, target_folder, target_file):
    found_directories = []
    
    for entry in base_path.rglob(target_folder):  # Search for all 'parsers' directories
        if entry.is_dir() and entry.name.lower() == target_folder.lower():
            for file in entry.iterdir():
                if file.is_file() and file.name.lower() == target_file.lower():
                    found_directories.append(entry)
                    break
    
    return found_directories

def get_directory_depth(directory, base_directory):
    return len(directory.relative_to(base_directory).parts)

def find_closest_directory(directories, base_directory):
    depths = [(dir, get_directory_depth(dir, base_directory)) for dir in directories]
    return min(depths, key=lambda x: x[1])[0]

def hash_file(filepath):
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def replace_pdf_in_parsers():
    script_dir = Path(__file__).parent
    user_manual_pdf_path = script_dir / "User_Manual" / "PDF.py"

    if not user_manual_pdf_path.exists():
        print("No 'pdf.py' file found in 'User_Manual' directory.")
        return

    base_dir = script_dir.parent  # Move up one level from the script's location
    target_folder = "parsers"
    target_file = "pdf.py"
    found_paths = find_all_target_directories_with_file(base_dir, target_folder, target_file)

    if not found_paths:
        print("No suitable 'parsers' directory found.")
        return

    if len(found_paths) == 1:
        chosen_pdf_path = found_paths[0] / target_file
    else:
        closest_parsers_path = find_closest_directory(found_paths, base_dir)
        print(f"Chosen 'parsers' directory based on path depth: {closest_parsers_path}")
        chosen_pdf_path = closest_parsers_path / target_file

    # Hash comparison and replacement
    user_manual_pdf_hash = hash_file(user_manual_pdf_path)
    chosen_pdf_hash = hash_file(chosen_pdf_path)

    if user_manual_pdf_hash != chosen_pdf_hash:
        print("Replacing the existing pdf.py with the new one...")
        shutil.copy(user_manual_pdf_path, chosen_pdf_path)
        print(f"PDF.py replaced at: {chosen_pdf_path}")
    else:
        print("No replacement needed. The files are identical.")

if __name__ == "__main__":
    replace_pdf_in_parsers()
