import os
import subprocess
from pathlib import Path
from PySide6.QtCore import QObject, Signal

class ModelDownloadedSignal(QObject):
    downloaded = Signal(str, str)

model_downloaded_signal = ModelDownloadedSignal()

MODEL_DIRECTORIES = {
    "vector": "Vector",
    "chat": "Chat"
}

class ModelDownloader:
    def __init__(self, model_info, model_type):
        self.model_info = model_info
        self.model_type = model_type
        self._model_directory = None

    def get_model_directory_name(self):
        if isinstance(self.model_info, dict):
            return self.model_info['cache_dir']
        else:
            return self.model_info.replace("/", "--")

    def get_model_directory(self):
        if not self._model_directory:
            model_type_dir = MODEL_DIRECTORIES.get(self.model_type, "")
            self._model_directory = Path("Models") / model_type_dir / self.get_model_directory_name()
        return self._model_directory

    def get_model_url(self):
        if isinstance(self.model_info, dict):
            return f"https://huggingface.co/{self.model_info['repo_id']}"
        else:
            return f"https://huggingface.co/{self.model_info}"

    def download_model(self):
        model_url = self.get_model_url()
        target_directory = self.get_model_directory()
        print(f"Downloading {self.get_model_directory_name()}...")
        
        env = os.environ.copy()
        env["GIT_CLONE_PROTECTION_ACTIVE"] = "false"
        
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", model_url, str(target_directory)],
                check=True,
                env=env
            )
            print("\033[92mModel downloaded and ready to use.\033[0m")
            print(f"Emitting signal: {self.get_model_directory_name()}, {self.model_type}")
            model_downloaded_signal.downloaded.emit(self.get_model_directory_name(), self.model_type)
        except subprocess.CalledProcessError as e:
            print(f"Command 'git clone' returned non-zero exit status {e.returncode}.")