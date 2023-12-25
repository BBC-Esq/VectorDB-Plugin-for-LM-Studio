import yaml
from PySide6.QtWidgets import QApplication, QFileDialog
from pathlib import Path

def load_config():
    with open(Path("config.yaml"), 'r') as stream:
        return yaml.safe_load(stream)

def select_embedding_model_directory():
    initial_dir = Path('Embedding_Models') if Path('Embedding_Models').exists() else Path.home()
    chosen_directory = QFileDialog.getExistingDirectory(None, "Select Embedding Model Directory", str(initial_dir))
    
    if chosen_directory:
        config_data = load_config()
        config_data["EMBEDDING_MODEL_NAME"] = chosen_directory
        with open(Path("config.yaml"), 'w') as file:
            yaml.dump(config_data, file)

        print(f"Selected directory: {chosen_directory}")

if __name__ == '__main__':
    app = QApplication([])
    select_embedding_model_directory()
    app.exec()
