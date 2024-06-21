import os
import gc
import torch
import pyaudio
import wave
import tempfile
from pathlib import Path
import whisper_s2t
from PySide6.QtCore import QThread, Signal
from utilities import my_cprint

class TranscriptionThread(QThread):
    transcription_complete = Signal(str)

    def __init__(self, audio_file, voice_recorder):
        super().__init__()
        self.audio_file = audio_file
        self.voice_recorder = voice_recorder

    def run(self):
        device = "cpu"
        compute_type = "float32"
        model_identifier = "ctranslate2-4you/whisper-small.en-ct2-float32"
        cpu_threads = max(4, os.cpu_count() - 4)
        model_kwargs = {
            'compute_type': compute_type,
            'model_identifier': model_identifier,
            'backend': 'CTranslate2',
            "device": device,
            "cpu_threads": cpu_threads,
        }
        self.model = whisper_s2t.load_model(**model_kwargs)

        out = self.model.transcribe_with_vad([self.audio_file],
                                             lang_codes=['en'],
                                             tasks=['transcribe'],
                                             initial_prompts=[None],
                                             batch_size=16)

        transcription_text = " ".join([_['text'] for _ in out[0]]).strip()

        my_cprint("Transcription completed.", 'white')
        self.transcription_complete.emit(transcription_text)
        Path(self.audio_file).unlink()
        self.voice_recorder.ReleaseTranscriber()

class RecordingThread(QThread):
    def __init__(self, voice_recorder):
        super().__init__()
        self.voice_recorder = voice_recorder

    def run(self):
        self.voice_recorder.record_audio()

class VoiceRecorder:
    def __init__(self, gui_instance, format=pyaudio.paInt16, channels=1, rate=16000, chunk=1024):
        self.gui_instance = gui_instance
        self.format, self.channels, self.rate, self.chunk = format, channels, rate, chunk
        self.is_recording, self.frames = False, []
        self.recording_thread = None
        self.transcription_thread = None

    def record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        self.frames = []
        while self.is_recording:
            data = stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def save_audio(self):
        self.is_recording = False
        temp_file = Path(tempfile.mktemp(suffix=".wav"))
        with wave.open(str(temp_file), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))
        self.frames.clear()

        self.transcription_thread = TranscriptionThread(str(temp_file), self)
        self.transcription_thread.transcription_complete.connect(self.gui_instance.update_transcription)
        self.transcription_thread.start()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = RecordingThread(self)
            self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.wait()
            self.save_audio()

    def ReleaseTranscriber(self):
        if hasattr(self, 'model'):
            if hasattr(self.model, 'model'):
                del self.model.model
            if hasattr(self.model, 'feature_extractor'):
                del self.model.feature_extractor
            if hasattr(self.model, 'hf_tokenizer'):
                del self.model.hf_tokenizer
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Whisper model removed from memory.", 'red')
