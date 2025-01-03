import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

class DependencyUpdater:
    def __init__(self):
        self.site_packages_path = self.get_site_packages_path()

    def get_site_packages_path(self):
        paths = sys.path
        site_packages_paths = [Path(path) for path in paths if 'site-packages' in path.lower()]
        return site_packages_paths[0] if site_packages_paths else None

    def find_dependency_path(self, dependency_path_segments):
        current_path = self.site_packages_path
        if current_path and current_path.exists():
            for segment in dependency_path_segments:
                next_path = next((current_path / child for child in current_path.iterdir() if child.name.lower() == segment.lower()), None)
                if next_path is None:
                    return None
                current_path = next_path
            return current_path
        return None

    @staticmethod
    def hash_file(filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        return hasher.hexdigest()

    @staticmethod
    def copy_and_overwrite_if_necessary(source_path, target_path):
        if not target_path.exists() or DependencyUpdater.hash_file(source_path) != DependencyUpdater.hash_file(target_path):
            shutil.copy(source_path, target_path)
            print(f"{source_path} has been successfully copied to {target_path}.")
        else:
            print(f"{target_path} is already up to date.")

    def update_file_in_dependency(self, source_folder, file_name, dependency_path_segments):
        target_path = self.find_dependency_path(dependency_path_segments)
        if target_path is None:
            print("Target dependency path not found.")
            return
        
        source_path = Path(__file__).parent / source_folder / file_name
        if not source_path.exists():
            print(f"{file_name} not found in {source_folder}.")
            return

        target_file = None
        for child in target_path.iterdir():
            if child.is_file() and child.name.lower() == file_name.lower():
                target_file = child
                break
        
        if target_file:
            target_file_path = target_file
        else:
            target_file_path = target_path / file_name
        self.copy_and_overwrite_if_necessary(source_path, target_file_path)

def replace_pdf_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("Assets", "pdf.py", ["langchain_community", "document_loaders", "parsers"])

def replace_instructor_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("Assets", "instructor.py", ["InstructorEmbedding"])

def replace_sentence_transformer_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("Assets", "SentenceTransformer.py", ["sentence_transformers"])

def replace_chattts_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("Assets", "core.py", ["ChatTTS"])

def add_cuda_files():
    updater = DependencyUpdater()
    
    # First part: Copy ptxas.exe
    source_path = updater.find_dependency_path(["nvidia", "cuda_nvcc", "bin"])
    if source_path is None:
        print("Source path for ptxas.exe not found.")
        return
    
    source_file = source_path / "ptxas.exe"
    if not source_file.exists():
        print("ptxas.exe not found in the source directory.")
        return
    
    target_path = updater.find_dependency_path(["nvidia", "cuda_runtime", "bin"])
    if target_path is None:
        print("Target path (cuda_runtime) not found.")
        return
    
    target_file = target_path / "ptxas.exe"
    updater.copy_and_overwrite_if_necessary(source_file, target_file)

    # Second part: Extract zip file
    zip_path = Path(__file__).parent / "Assets" / "cudart_lib.zip"
    if not zip_path.exists():
        print("cudart_lib.zip not found.")
        return
    
    # Ensure that target_bin_path has a parent (which should be cuda_runtime)
    cuda_lib_runtime_path = target_path.parent
    if target_path is None or not target_path.exists():
        print("Parent directory of cuda_runtime/bin not found.")
        return
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cuda_lib_runtime_path)
            print(f"Extracted cudart_lib.zip to {cuda_lib_runtime_path}")
    except zipfile.BadZipFile:
        print("Error: cudart_lib.zip is corrupted or not a zip file.")
    except PermissionError:
        print("Error: Permission denied when extracting cudart_lib.zip.")
    except Exception as e:
        print(f"Unexpected error during extraction: {str(e)}")

def setup_vector_db():
    zip_path = Path(__file__).parent / "Assets" / "user_manual_db.zip"
    if not zip_path.exists():
        print("user_manual_db.zip not found in Assets folder.")
        return

    vector_db_path = Path(__file__).parent / "Vector_DB"
    vector_db_backup_path = Path(__file__).parent / "Vector_DB_Backup"

    try:
        vector_db_path.mkdir(exist_ok=True)
        vector_db_backup_path.mkdir(exist_ok=True)
    except PermissionError:
        print("Error: Insufficient permissions to create directories.")
        return
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return

    user_manual_paths = [
        vector_db_path / "user_manual",
        vector_db_backup_path / "user_manual"
    ]

    for path in user_manual_paths:
        if path.exists():
            try:
                shutil.rmtree(path, ignore_errors=False)
                print(f"Removed existing user_manual folder from {path.parent}")
            except PermissionError:
                print(f"Error: Permission denied when trying to remove {path}")
                return
            except Exception as e:
                print(f"Error removing {path}: {str(e)}")
                return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if zip_ref.testzip() is not None:
                print("Error: Zip file is corrupted")
                return
            zip_ref.extractall(vector_db_path)
            zip_ref.extractall(vector_db_backup_path)
        print(f"Successfully extracted user_manual_db.zip to {vector_db_path} and {vector_db_backup_path}")
    except PermissionError:
        print("Error: Permission denied when extracting zip file")
    except Exception as e:
        print(f"Error extracting zip file: {str(e)}")

if __name__ == "__main__":
    replace_pdf_file()
    replace_instructor_file()
    replace_sentence_transformer_file()
    replace_chattts_file()
    add_cuda_files()
    setup_vector_db()