import pyaudio
import wave
import os
import tempfile
from faster_whisper import WhisperModel
import torch
import gc
import yaml
from termcolor import cprint
from PySide6.QtCore import QThread, Signal

ENABLE_PRINT = True

def my_cprint(*args, **kwargs):
    if ENABLE_PRINT:
        filename = "voice_recorder_module.py"
        modified_message = f"{filename}: {args[0]}"
        cprint(modified_message, *args[1:], **kwargs)

class TranscriptionThread(QThread):
    transcription_complete = Signal(str)

    def __init__(self, audio_file, model):
        super().__init__()
        self.audio_file = audio_file
        self.model = model

    def run(self):
        segments, _ = self.model.transcribe(self.audio_file)
        transcription_text = "\n".join([segment.text for segment in segments])
        my_cprint("Transcription completed.", 'green')
        self.transcription_complete.emit(transcription_text)
        os.remove(self.audio_file)

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
        temp_filename = tempfile.mktemp(suffix=".wav")
        with wave.open(temp_filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))
        self.frames.clear()

        self.transcription_thread = TranscriptionThread(temp_filename, self.model)
        self.transcription_thread.transcription_complete.connect(self.gui_instance.update_transcription)
        self.transcription_thread.transcription_complete.connect(self.ReleaseTranscriber)
        self.transcription_thread.start()

    def start_recording(self):
        if not self.is_recording:
            with open("config.yaml", 'r') as stream:
                config_data = yaml.safe_load(stream)
            
            transcriber_config = config_data['transcriber']
            model_string = f"ctranslate2-4you/whisper-{transcriber_config['model']}-ct2-{transcriber_config['quant']}"
            
            cpu_threads = max(4, os.cpu_count() - 6)
            
            print(f"Loaded device: {transcriber_config['device']}")
            print(f"Loaded model: {transcriber_config['model']}")
            print(f"Loaded quant: {transcriber_config['quant']}")
            
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
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        my_cprint("Whisper model removed from memory.", 'red')
