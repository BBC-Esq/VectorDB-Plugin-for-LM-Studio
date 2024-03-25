import pickle
from pathlib import Path
import yaml
from langchain_community.docstore.document import Document
from utilities import my_cprint

ROOT_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"

def load_audio_documents(source_dir: Path = SOURCE_DIRECTORY) -> list:
    pkl_paths = [f for f in source_dir.iterdir() if f.suffix.lower() == '.pkl']
    docs = []

    for pkl_path in pkl_paths:
        try:
            with open(pkl_path, 'rb') as pkl_file:
                doc = pickle.load(pkl_file)
                docs.append(doc)
        except Exception as e:
            my_cprint(f"Error loading {pkl_path}: {e}", "red")

    return docs
