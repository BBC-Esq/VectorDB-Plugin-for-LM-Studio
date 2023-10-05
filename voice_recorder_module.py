import pyaudio
import wave
import os
import tempfile
import threading
import pyperclip
from faster_whisper import WhisperModel

class VoiceRecorder:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
        self.format, self.channels, self.rate, self.chunk = format, channels, rate, chunk
        self.is_recording, self.frames = False, []
                
        current_directory = os.getcwd()

        model_folder_path = os.path.join(
            current_directory,
            "whisper-small.en-ct2-int8_float32"
        )

        self.model = WhisperModel(
            model_folder_path,
            device="auto",
            compute_type="int8_float32"
        )

    def transcribe_audio(self, audio_file):
        segments, _ = self.model.transcribe(audio_file)
        pyperclip.copy("\n".join([segment.text for segment in segments]))

    def record_audio(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
            [self.frames.append(stream.read(self.chunk)) for _ in iter(lambda: self.is_recording, False)]
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

        def transcribe_and_cleanup():
            self.transcribe_audio(temp_filename)
            os.remove(temp_filename)
            self.frames.clear()

        threading.Thread(target=transcribe_and_cleanup).start()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.save_audio()
