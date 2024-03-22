import warnings
import threading
import queue
from transformers import AutoProcessor, BarkModel
import torch
import numpy as np
import re
import pyaudio
import gc
import yaml
import platform
from tqdm import tqdm
from utilities import my_cprint

warnings.filterwarnings("ignore", message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

class BarkAudio:
    def __init__(self):
        self.load_config()
        self.initialize_model_and_processor()
        self.sentence_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.start_time = None
        self.running = True
        self.lock = threading.Lock()

    def load_config(self):
        with open('config.yaml', 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)['bark']

    def initialize_model_and_processor(self):
        os_name = platform.system().lower()
        if torch.cuda.is_available():
            if torch.version.hip and os_name == 'linux':
                self.device = "cuda:0"
            elif torch.version.cuda:
                self.device = "cuda:0"
            elif torch.version.hip and os_name == 'windows':
                self.device = "cpu"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif os_name == 'darwin':
            self.device = "cpu"
        else:
            self.device = "cpu"
        
        my_cprint(f"Selected compute device: {self.device}", "white")
        
        # load processor
        if self.config['size'] == 'small':
            self.processor = AutoProcessor.from_pretrained("suno/bark-small")
            my_cprint("Bark processor loaded.", "green")
        else:
            self.processor = AutoProcessor.from_pretrained("suno/bark")
            my_cprint("Bark processor loaded.", "green")
        
        # load bark model
        if self.config['size'] == 'small' and self.config['model_precision'] == 'float16':
            self.model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(self.device)
            print("Size = Small")
            print("Quant = float16")
        elif self.config['size'] == 'small' and self.config['model_precision'] != 'float16':
            self.model = BarkModel.from_pretrained("suno/bark-small").to(self.device)
            print("Size = Small")
            print("Quant = float32")
        elif self.config['size'] != 'small' and self.config['model_precision'] == 'float16':
            self.model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(self.device)
            print("Size = Normal")
            print("Quant = float16")
        else:
            self.model = BarkModel.from_pretrained("suno/bark").to(self.device)
            print("Size = Normal")
            print("Quant = float32")
        
        my_cprint("Bark model loaded.", "green")
        
        self.model = self.model.to_bettertransformer()

        if self.config['enable_cpu_offload']:
            self.model.enable_cpu_offload()
            my_print("Offload to CPU selected.", "cyan")

    def play_audio_thread(self):
        my_cprint("Creating audio.", "white")
        while self.running:
            queue_item = self.sentence_queue.get()
            if queue_item is None:
                my_cprint("All audio finished playing.", "white")
                break

            audio_array, sampling_rate, _ = queue_item

            try:
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, output=True)
                stream.write(audio_array.tobytes())
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            finally:
                p.terminate()

        self.release_resources()

    def process_text_thread(self):
        while self.running:
            text_prompt = self.processing_queue.get()
            if text_prompt is None:
                break

            sentences = re.split(r'[.!?;]+', text_prompt)
            for sentence in tqdm(sentences, desc="Processing Sentences"):
                if sentence.strip():
                    voice_preset = self.config['speaker']
                    inputs = self.processor(text=sentence, voice_preset=voice_preset, return_tensors="pt")

                    try:
                        speech_output = self.model.generate(**inputs.to(self.device), pad_token_id=0, do_sample=True)
                    except Exception:
                        continue

                    audio_array = speech_output[0].cpu().numpy()
                    audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
                    sampling_rate = self.model.generation_config.sample_rate

                    self.sentence_queue.put((audio_array, sampling_rate, len(sentences) - 1))

            self.sentence_queue.put(None)

    def run(self):
        self.load_config()
        with open('chat_history.txt', 'r', encoding='utf-8') as file:
            llm_response = file.read()
            self.processing_queue.put(llm_response)

        processing_thread = threading.Thread(target=self.process_text_thread)
        playback_thread = threading.Thread(target=self.play_audio_thread)
        
        processing_thread.daemon = True
        playback_thread.daemon = True
        
        processing_thread.start()
        playback_thread.start()

        processing_thread.join()
        playback_thread.join()

    def release_resources(self):
        if hasattr(self.model, 'semantic'):
            del self.model.semantic
        if hasattr(self.model, 'coarse_acoustics'):
            del self.model.coarse_acoustics
        if hasattr(self.model, 'fine_acoustics'):
            del self.model.fine_acoustics
        
        del self.processor
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Bark models removed from memory.", "red")
