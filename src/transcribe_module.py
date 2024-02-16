import os
import yaml
import pickle
from multiprocessing import Process
from faster_whisper import WhisperModel
from utilities import my_cprint
from extract_metadata import extract_audio_metadata
from pathlib import Path

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata
        self.type = "Document"

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
        self.create_document_object(transcription)
        my_cprint("Transcription completed and document object created.", 'red')

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

    def create_document_object(self, transcription_text):
        metadata = extract_audio_metadata(self.audio_file)
        
        doc = Document(
            page_content=transcription_text,
            metadata=metadata
        )
        
        script_dir = Path(__file__).parent
        docs_dir = script_dir / "Docs_for_DB"
        docs_dir.mkdir(exist_ok=True)
        
        audio_file_name = Path(self.audio_file).stem
        audio_file_extension = Path(self.audio_file).suffix
        pickle_file_path = docs_dir / f"{audio_file_name}{audio_file_extension}.pkl"
        
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(doc, file)

        my_cprint(f"Document object created with transcription and metadata, and saved to {pickle_file_path}.", 'green')
