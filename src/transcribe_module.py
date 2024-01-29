import os
import yaml
from multiprocessing import Process
from faster_whisper import WhisperModel
from utilities import my_cprint

class TranscribeFile:
    def __init__(self, audio_file, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)['transcribe_file']

        self.audio_file = audio_file
        self.include_timestamps = config.get('timestamps', False)
        self.cpu_threads = max(4, os.cpu_count() - 4)

        self.model_config = {
            'model_name': f"ctranslate2-4you/{config['model']}-ct2-{config['quant']}",
            'device': config['device'],
            'compute_type': config['quant'],
            'cpu_threads': self.cpu_threads
        }
        
        my_cprint(f"Loaded device: {config['device']}", 'white')
        my_cprint(f"Loaded model: {config['model']}", 'white')
        my_cprint(f"Loaded quant: {config['quant']}", 'white')

        if config['device'].lower() == 'cpu':
            my_cprint(f"CPU Threads: {self.cpu_threads}", 'white')

    def start_transcription(self):
        my_cprint("Starting transcription process...", 'white')
        self.transcription_process = Process(target=self.transcribe)
        self.transcription_process.start()

    def transcribe(self):
        model = WhisperModel(
            self.model_config['model_name'],
            device=self.model_config['device'],
            compute_type=self.model_config['compute_type'],
            cpu_threads=self.model_config.get('cpu_threads')
        )
        my_cprint("Whisper model loaded.", 'green')

        segments_generator, _ = model.transcribe(self.audio_file, beam_size=5)
        segments = []

        for segment in segments_generator:
            segments.append(segment)
            my_cprint(segment.text, 'blue')

        transcription = TranscribeFile.format_transcription(segments, self.include_timestamps)
        TranscribeFile.save_transcription(self.audio_file, transcription)
        my_cprint("Transcription completed. Whisper model removed from memory.", 'red')

    @staticmethod
    def format_transcription(segments, include_timestamps):
        transcription = []
        for segment in segments:
            formatted_segment = segment.text
            if include_timestamps:
                start_time = TranscribeFile.format_time(segment.start)
                end_time = TranscribeFile.format_time(segment.end)
                formatted_segment = f"{start_time} - {end_time} {formatted_segment}"
            transcription.append(formatted_segment)
        
        return "\n".join(transcription)

    @staticmethod
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def save_transcription(audio_file, transcription_text):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        docs_dir = os.path.join(script_dir, "Docs_for_DB")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)

        base_audio_file_name = os.path.basename(audio_file)
        base_file_name_without_ext = os.path.splitext(base_audio_file_name)[0]
        transcribed_file_name = f"transcribed_{base_file_name_without_ext}.txt"
        output_file = os.path.join(docs_dir, transcribed_file_name)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(transcription_text)
        my_cprint(f"Transcription saved to {output_file}.", 'white')
