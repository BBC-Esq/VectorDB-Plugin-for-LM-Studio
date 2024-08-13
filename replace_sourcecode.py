import hashlib
from pathlib import Path
import shutil
import sys

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
    updater.update_file_in_dependency("user_manual", "pdf.py", ["langchain_community", "document_loaders", "parsers"])

def replace_instructor_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("user_manual", "instructor.py", ["InstructorEmbedding"])

def replace_sentence_transformer_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("user_manual", "SentenceTransformer.py", ["sentence_transformers"])

if __name__ == "__main__":
    replace_pdf_file()
    replace_instructor_file()
    replace_sentence_transformer_file()