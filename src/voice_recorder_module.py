import pyaudio
import wave
import os
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel
import torch
import gc
import yaml
from PySide6.QtCore import QThread, Signal
from utilities import my_cprint

class TranscriptionThread(QThread):
    transcription_complete = Signal(str)

    def __init__(self, audio_file, model):  # Can add additional parameters from the transcribe method of WhisperModel
        super().__init__()
        self.audio_file = audio_file
        self.model = model

    def run(self):
        segments, _ = self.model.transcribe(self.audio_file)
        transcription_text = "\n".join([segment.text for segment in segments])
        my_cprint("Transcription completed.", 'white')
        self.transcription_complete.emit(transcription_text)
        Path(self.audio_file).unlink()

class RecordingThread(QThread):
    def __init__(self, voice_recorder):
        super().__init__()
        self.voice_recorder = voice_recorder

    def run(self):
        self.voice_recorder.record_audio()

class VoiceRecorder:
    def __init__(self, gui_instance, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
        self.gui_instance = gui_instance
        self.format, self.channels, self.rate, self.chunk = format, channels, rate, chunk
        self.is_recording, self.frames = False, []
        self.recording_thread = None
        self.transcription_thread = None

    def record_audio(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
            while self.is_recording:
                self.frames.append(stream.read(self.chunk))
            stream.stop_stream()
            stream.close()
        finally:
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

        self.transcription_thread = TranscriptionThread(str(temp_file), self.model)
        self.transcription_thread.transcription_complete.connect(self.gui_instance.update_transcription)
        self.transcription_thread.transcription_complete.connect(self.ReleaseTranscriber)
        self.transcription_thread.start()

    def start_recording(self):
        if not self.is_recording:
            with open("config.yaml", 'r') as stream:
                config_data = yaml.safe_load(stream)
            
            transcriber_config = config_data['transcriber']
            model_string = f"ctranslate2-4you/{transcriber_config['model']}-ct2-{transcriber_config['quant']}"
            
            cpu_threads = max(4, os.cpu_count() - 6)
            
            print(f"Device = {transcriber_config['device']}")
            print(f"Model = {transcriber_config['model']}")
            print(f"Quant = {transcriber_config['quant']}")
            
            self.model = WhisperModel(
                model_string,
                device=transcriber_config['device'],
                compute_type=transcriber_config['quant'],
                cpu_threads=cpu_threads
            )
            my_cprint("Whisper model loaded.", 'green')
            
            self.is_recording = True
            self.recording_thread = RecordingThread(self)
            self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.wait()
            self.save_audio()

    def ReleaseTranscriber(self):
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
