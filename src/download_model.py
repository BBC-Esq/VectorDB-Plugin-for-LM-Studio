from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import disable_progress_bars, RepositoryNotFoundError, GatedRepoError
from huggingface_hub.hf_api import RepoFile
from PySide6.QtCore import QObject, Signal
import fnmatch
import humanfriendly

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
        self.api = HfApi()
        disable_progress_bars()
    
    def get_model_url(self):
        if isinstance(self.model_info, dict):
            return self.model_info['repo_id']
        else:
            return self.model_info

    def check_repo_type(self, repo_id):
        try:
            repo_info = self.api.repo_info(repo_id)
            if repo_info.private:
                return "private"
            elif getattr(repo_info, 'gated', False):
                return "gated"
            else:
                return "public"
        except GatedRepoError:
            return "gated"
        except RepositoryNotFoundError:
            return "not_found"
        except Exception as e:
            return f"error: {str(e)}"

    def get_model_directory_name(self):
        if isinstance(self.model_info, dict):
            return self.model_info['cache_dir']
        else:
            return self.model_info.replace("/", "--")

    def get_model_directory(self):
        model_type_dir = MODEL_DIRECTORIES.get(self.model_type, "")
        return Path("Models") / model_type_dir / self.get_model_directory_name()

    def download_model(self):
        repo_id = self.get_model_url()
        
        # only download if repo is public
        # https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user
        # https://huggingface.co/docs/hub/en/enterprise-hub-tokens-management
        repo_type = self.check_repo_type(repo_id)
        if repo_type != "public":
            if repo_type == "private":
                print(f"Repository {repo_id} is private and requires a token. Aborting download.")
            elif repo_type == "gated":
                print(f"Repository {repo_id} is gated. Please request access through the web interface. Aborting download.")
            elif repo_type == "not_found":
                print(f"Repository {repo_id} not found. Aborting download.")
            else:
                print(f"Error checking repository {repo_id}: {repo_type}. Aborting download.")
            return

        local_dir = self.get_model_directory()
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            # use list_repo_tree from huggingface_hub
            repo_files = list(self.api.list_repo_tree(repo_id, recursive=True))
            
            safetensors_files = [file for file in repo_files if file.path.endswith('.safetensors')]
            bin_files = [file for file in repo_files if file.path.endswith('.bin')]
            
            ignore_patterns = [".gitattributes", "*.ckpt", "*.gguf", "*.h5", "*.ot", "*.md", "README*", "onnx/**", "coreml/**"]
            
            if safetensors_files and bin_files:
                ignore_patterns.append("*.bin")
            
            # Only add "*consolidated*" to ignore_patterns if there are other .safetensors or .bin files
            if safetensors_files or bin_files:
                ignore_patterns.append("*consolidated*")
            
            total_size = 0
            included_files = []
            ignored_files = []
            
            for file in repo_files:
                if isinstance(file, RepoFile):
                    if not any(fnmatch.fnmatch(file.path, pattern) for pattern in ignore_patterns):
                        included_files.append(file)
                        total_size += file.size
                    else:
                        ignored_files.append(file)
            
            readable_total_size = humanfriendly.format_size(total_size)
            print(f"\nTotal size to be downloaded: {readable_total_size}")
            print(f"\nDownloading to {local_dir}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                max_workers=4,
                ignore_patterns=ignore_patterns
            )
            print("\033[92mModel downloaded and ready to use.\033[0m")
            model_downloaded_signal.downloaded.emit(self.get_model_directory_name(), self.model_type)
        
        except Exception as e:
            print(f"An error occurred during download: {str(e)}")
            # remove directory if download fails
            if local_dir.exists():
                import shutil
                shutil.rmtree(local_dir)