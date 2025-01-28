import queue
import re
import threading
from pathlib import Path

import io
import numpy as np
import sounddevice as sd
# print(sd.query_devices()) # DEBUG
import torch
import yaml
from tqdm import tqdm
import ChatTTS
from transformers import AutoProcessor, BarkModel
from whisperspeech.pipeline import Pipeline
import soundfile as sf
from gtts import gTTS
from gtts.tokenizer import pre_processors, tokenizer_cases

from utilities import my_cprint
from constants import WHISPER_SPEECH_MODELS

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "models" / "tts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class BaseAudio:
    def __init__(self):
        self.sentence_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.config = {}
        self.processing_thread = None

    def load_config(self, config_file, section):
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if section in config_data:
                self.config = config_data[section]
            else:
                print(f"Warning: Section '{section}' not found in config file.")
                self.config = {}

    def initialize_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise RuntimeError("CUDA is not available, but it's required for this program.")

    def play_audio_from_queue(self):
        while not self.stop_event.is_set():
            try:
                queue_item = self.audio_queue.get(timeout=5)
                if queue_item is None or self.stop_event.is_set():
                    break
                audio_array, sampling_rate = queue_item
                try:
                    if len(audio_array.shape) == 1:
                        audio_array = np.expand_dims(audio_array, axis=1)
                    elif len(audio_array.shape) == 2 and audio_array.shape[1] != 1:
                        audio_array = audio_array.T
                    sd.play(audio_array, samplerate=sampling_rate)
                    sd.wait()
                except Exception as e:
                    print(f"Error playing audio: {e}")
            except queue.Empty:
                if self.processing_thread is None or not self.processing_thread.is_alive():
                    break

    def run(self, input_text_file):
        try:
            with open(input_text_file, 'r', encoding='utf-8') as file:
                input_text = file.read()
                sentences = re.split(r'[.!?;]+\s*', input_text)
        except Exception as e:
            print(f"Error reading {input_text_file}: {e}")
            return

        self.processing_thread = threading.Thread(target=self.process_text_to_audio, args=(sentences,)) # thread 1
        playback_thread = threading.Thread(target=self.play_audio_from_queue) # thread 2

        self.processing_thread.daemon = True
        playback_thread.daemon = True

        self.processing_thread.start()
        playback_thread.start()

        self.processing_thread.join()
        playback_thread.join()

    def stop(self):
        self.stop_event.set()
        self.audio_queue.put(None)


class BarkAudio(BaseAudio):
    def __init__(self):
        super().__init__()
        self.load_config('config.yaml', 'bark')
        self.initialize_device()
        self.initialize_model_and_processor()

    def initialize_model_and_processor(self):
        repository_id = "suno/bark" if self.config['size'] == 'normal' else f"suno/bark-{self.config['size']}"
        
        self.processor = AutoProcessor.from_pretrained(repository_id, cache_dir=CACHE_DIR)
        
        self.model = BarkModel.from_pretrained(
            repository_id,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
            # attn_implementation="flash_attention_2"
        ).to(self.device)

        self.model.eval()
        
        my_cprint("Bark model loaded (float16)", "green")

        # self.model = self.model.to_bettertransformer()

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        for sentence in tqdm(sentences, desc="Processing Sentences"):
            if sentence.strip():
                print(f"Processing sentence: {sentence}")
                try:
                    inputs = self.processor(text=sentence, voice_preset=self.config['speaker'], return_tensors="pt")
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                            for k, v in inputs.items()}

                    speech_output = self.model.generate(
                        **inputs,
                        use_cache=True,
                        do_sample=True,
                        # temperature=0.2,
                        # top_k=50,
                        # top_p=0.95,
                        pad_token_id=0,
                    )

                    audio_array = speech_output[0].cpu().numpy()
                    audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
                    self.audio_queue.put((audio_array, self.model.generation_config.sample_rate))
                except Exception as e:
                    print(f"Exception during audio generation: {str(e)}")
                    continue
        self.audio_queue.put(None)


class WhisperSpeechAudio(BaseAudio):
    def __init__(self):
        super().__init__()
        self.load_config('config.yaml', 'tts')
        self.pipe = None
        self.initialize_model()

    def get_whisper_speech_models(self):
        s2a_model = self.config.get('s2a', 's2a-q4-hq-fast-en+pl.model')
        s2a = f"collabora/whisperspeech:{s2a_model}"
        
        t2s_model = self.config.get('t2s', 't2s-base-en+pl.model')
        t2s = f"collabora/whisperspeech:{t2s_model}"

        return s2a, t2s

    def initialize_model(self):
        s2a, t2s = self.get_whisper_speech_models()

        try:
            self.pipe = Pipeline(
                s2a_ref=s2a,
                t2s_ref=t2s,
                cache_dir=CACHE_DIR
            )
            my_cprint(f"{s2a.split(':')[-1]} loaded\n{t2s.split(':')[-1]} loaded.", "green")
        except Exception as e:
            my_cprint(f"Error initializing WhisperSpeech models: {str(e)}", "red")
            self.pipe = None

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        for sentence in tqdm(sentences, desc="Processing Sentences"):
            if sentence and not self.stop_event.is_set():
                try:
                    audio_tensor = self.pipe.generate(sentence)
                    audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
                    if len(audio_np.shape) == 1:
                        audio_np = np.expand_dims(audio_np, axis=1)
                    else:
                        audio_np = audio_np.T
                    self.audio_queue.put((audio_np, 24000))
                except Exception as e:
                    my_cprint(f"Error processing sentence: {str(e)}", "red")
        self.audio_queue.put(None)

    def run(self, input_text_file):
        self.initialize_device()
        super().run(input_text_file)


class ChatTTSAudio(BaseAudio):
    """
    +----------------+------------------+------------------------+---------------+
    | Source         | Parameters       | Behavior             | Repo Structure? |
    +----------------+------------------+------------------------+---------------+
    | "huggingface"  | (default)        | ~/.cache/huggingface | No              |
    | "huggingface"  | cache_dir="path" | Downloads to path    | No              |
    | "huggingface"  | local_dir="path" | Downloads to path    | Yes             |
    +----------------+------------------+----------------------+-----------------+
    | "local"        | (default)        | Current directory    | No              |
    | "local"        | cache_dir="path" | Downloads to path    | No              |
    | "local"        | local_dir="path" | Downloads to path    | Yes             |
    +----------------+------------------+----------------------+-----------------+
    | "custom"       | custom_path      | Uses existing files  | n/a             |
    | "custom"       | +cache_dir       | cache_dir ignored    | n/a             |
    | "custom"       | +local_dir       | local_dir ignored    | n/a             |
    +----------------+------------------+----------------------+-----------------+

    1. `local_dir` takes precedence over everything else:
    ```python
    chat.load(source="huggingface", local_dir="path/to/dir", cache_dir="path/to/cache")  # local_dir wins
    ```

    2. If `local_dir` is not specified but `cache_dir` is, cache_dir is used:
    ```python
    chat.load(source="huggingface", cache_dir="path/to/cache")  # cache_dir used
    ```

    3. If neither is specified, the default location is used:
    ```python
    chat.load(source="huggingface")  # Uses ~/.cache/huggingface
    chat.load(source="local")        # Uses current directory
    ```
    """
    def __init__(self):
        super().__init__()
        
        print("Initializing ChatTTSAudio...")
        
        self.initialize_device()
        self.chat = ChatTTS.Chat()

        chattts_dir = CACHE_DIR / "2Noise--ChatTTS"
        chattts_dir.mkdir(parents=True, exist_ok=True)

        self.chat.load(
            source="huggingface",
            device=self.device,
            compile=False,
            use_flash_attn=False,
            local_dir=str(chattts_dir)
        )

        torch.manual_seed(11)
        self.rand_spk = self.chat.sample_random_speaker()

        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.rand_spk,
            temperature=0.7,
            top_P=1,
            top_K=40,
            prompt='[speed_5]'
        )

        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_0][laugh_0][break_0]',
            temperature=0.7,
            top_P=0.7,
            top_K=20
        )

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        print(f"Starting text processing... ({len(sentences)} sentences)")
        for sentence in sentences:
            if not sentence or not sentence.strip():
                continue

            print(f"Processing sentence: {sentence}")
            try:
                wavs = self.chat.infer(
                    sentence,
                    params_refine_text=self.params_refine_text,
                    params_infer_code=self.params_infer_code,
                    split_text=False # new in version 0.2.2
                )

                if wavs is not None and len(wavs) > 0:
                    audio_data = wavs[0]
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.cpu().numpy()
                    audio_data = audio_data.squeeze()

                    if np.prod(audio_data.shape) > 0:
                        print(f"Audio data shape: {audio_data.shape}")
                        if np.abs(audio_data).max() > 1.0:
                            audio_data = audio_data / np.abs(audio_data).max()
                        print(f"Audio range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
                        self.audio_queue.put((audio_data, 24000))
                        print("Audio data queued")
            except Exception as e:
                print(f"Error processing sentence: {str(e)}\n{type(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        print("Text processing complete, sending end signal")
        self.audio_queue.put(None)


class GoogleTTSAudio:
    """
    WWW.CHINTELLALAW.COM
    
    GoogleTTSAudio: A class for processing text into speech with advanced audio handling.

    This class provides functionality to convert text to speech using Google's Text-to-Speech 
    (gTTS) API, with additional features for audio processing and text segmentation.

    Key features:
    1. Text Preprocessing: Handles abbreviations, end-of-line characters, and tone marks.
    2. Intelligent Text Segmentation: Splits text into sentences and chunks of up to 100 
       characters, preserving sentence integrity where possible.
    3. Continuation Marking: Adds "<continue>" markers to chunks split mid-sentence to 
       potentially preserve prosody and intonation.
    4. Audio Generation: Converts text chunks to speech using gTTS.
    5. Silence Trimming: Removes excessive silence from the generated audio, with 
       configurable threshold and maximum silence duration.
    6. Audio Concatenation: Combines all generated audio segments into a single, 
       continuous audio stream.
    7. Playback: Provides functionality to play the processed audio.

    The class allows for customization of language, speech rate, and TLD for gTTS, 
    as well as silence threshold and maximum silence duration for audio processing.

    Usage:
    - Initialize with desired parameters (language, speech rate, TLD, silence threshold, 
      max silence duration).
    - Call the 'run' method with a text file path to process and play the audio.

    Note: This implementation focuses on improving text segmentation and audio quality. 
    Further enhancements for prosody and intonation preservation may require additional 
    libraries or post-processing techniques.
    """
    
    def __init__(self, lang='en', slow=False, tld='com', silence_threshold=0.01, max_silence_ms=100):
        self.lang = lang
        self.slow = slow
        self.tld = tld
        self.silence_threshold = silence_threshold
        self.max_silence_ms = max_silence_ms

    def run(self, input_text_file):
        try:
            with open(input_text_file, 'r', encoding='utf-8') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Error: File not found at {input_text_file}")
            return
        except IOError:
            print(f"Error: Unable to read file at {input_text_file}")
            return

        processed_text = self.preprocess_text(text)
        tokens = self.tokenize_and_minimize(processed_text)

        all_audio_data = []
        samplerate = None
        for token in tokens:
            if token.strip():
                print(f"Processing token: '{token}'")
                fp = io.BytesIO()

                if token.startswith("<continue>"):
                    token = token[10:].strip()

                tts = gTTS(text=token, lang=self.lang, slow=self.slow, tld=self.tld)
                tts.write_to_fp(fp)
                fp.seek(0)
                data, samplerate = sf.read(fp)
                all_audio_data.append(data)

        if all_audio_data:
            combined_audio = np.concatenate(all_audio_data)
            processed_audio = self.trim_silence(combined_audio, samplerate)
            sd.play(processed_audio, samplerate)
            sd.wait()
        else:
            print("No audio data generated.")

    @staticmethod
    def preprocess_text(text):
        text = pre_processors.abbreviations(text)
        text = pre_processors.end_of_line(text)
        text = pre_processors.tone_marks(text)
        return text

    @staticmethod
    def tokenize_and_minimize(text):
        sentences = re.split('(?<=[.!?])\s+', text)
        
        minimized_tokens = []
        for sentence in sentences:
            if len(sentence) <= 100:
                minimized_tokens.append(sentence)
            else:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > 100:
                        if current_chunk:
                            minimized_tokens.append(current_chunk.strip())
                            current_chunk = "<continue> " + word
                        else:
                            minimized_tokens.append(word)
                    else:
                        current_chunk += " " + word

                if current_chunk:
                    minimized_tokens.append(current_chunk.strip())

        return minimized_tokens

    def trim_silence(self, audio, samplerate):
        max_silence_samples = int(self.max_silence_ms * samplerate / 1000)

        is_silent = np.abs(audio) < self.silence_threshold

        # Find the boundaries of silent regions
        silent_regions = np.where(np.diff(is_silent.astype(int)))[0]

        if len(silent_regions) < 2:
            return audio

        processed_chunks = []
        start = 0

        for i in range(0, len(silent_regions) - 1, 2):
            silence_start, silence_end = silent_regions[i], silent_regions[i + 1]

            # Trim silence at the beginning of the chunk
            chunk_start = max(start, silence_start - max_silence_samples)

            # Trim silence at the end of the chunk
            chunk_end = min(silence_end, silence_start + max_silence_samples)
            
            processed_chunks.append(audio[chunk_start:chunk_end])
            start = silence_end

        processed_chunks.append(audio[start:])

        return np.concatenate(processed_chunks)


def run_tts(config_path, input_text_file):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        tts_model = config.get('tts', {}).get('model', 'bark')

    if tts_model == 'bark':
        audio_class = BarkAudio()
    elif tts_model == 'whisperspeech':
        audio_class = WhisperSpeechAudio()
    elif tts_model == 'chattts':
        audio_class = ChatTTSAudio()
    elif tts_model == 'googletts':
        audio_class = GoogleTTSAudio()
    elif tts_model == 'kokoro':
        audio_class = KokoroAudio()
    else:
        raise ValueError(f"Invalid TTS model specified in config.yaml: {tts_model}")

    audio_class.run(input_text_file)