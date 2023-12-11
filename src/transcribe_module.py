import os
import yaml
from multiprocessing import Process, Pipe
from faster_whisper import WhisperModel
from termcolor import cprint
from pydub import AudioSegment
import os

PRINT_ENABLED = True

def my_cprint(message, color='white'):
    if PRINT_ENABLED:
        cprint(f"TranscribeFile: {message}", color, flush=True)

class TranscribeFile:
    def __init__(self, audio_file, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['transcribe_file']

        self.audio_file = audio_file
        self.include_timestamps = config.get('timestamps', False)
        
        self.cpu_threads = max(4, os.cpu_count() - 4)

        self.model_config = {
            'model_name': f"ctranslate2-4you/whisper-{config['model']}-ct2-{config['quant']}",
            'device': config['device'],
            'compute_type': config['quant'],
            'cpu_threads': self.cpu_threads
        }
        
        my_cprint(f"Loaded device: {config['device']}", 'white')
        my_cprint(f"Loaded model: {config['model']}", 'white')
        my_cprint(f"Loaded quant: {config['quant']}", 'white')

        if config['device'].lower() == 'cpu':
            my_cprint(f"CPU Threads: {self.cpu_threads}", 'white')

    def start_transcription_thread(self, sender_pipe):
        my_cprint("Starting transcription process...", 'white')
        self.transcription_process = Process(
            target=self.transcribe_worker, 
            args=(self.audio_file, self.include_timestamps, self.model_config, sender_pipe))
        self.transcription_process.start()

    @staticmethod
    def transcribe_worker(audio_file, include_timestamps, model_config, sender_pipe):
        model = WhisperModel(
            model_config['model_name'],
            device=model_config['device'],
            compute_type=model_config['compute_type'],
            cpu_threads=model_config.get('cpu_threads')
        )
        my_cprint("Whisper model loaded.", 'green')

        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio)
        my_cprint(f"Total audio duration: {total_duration / 1000} seconds", 'yellow')

        segments_generator, _ = model.transcribe(audio_file, beam_size=1)
        segments = []

        accumulated_duration = 0

        # Iterate over the generator and process each segment
        for segment in segments_generator:
            segments.append(segment)  # Add the segment to the list

            segment_duration = segment.end - segment.start
            accumulated_duration += segment_duration
            progress_percentage = (accumulated_duration / (total_duration / 1000)) * 100
            sender_pipe.send(progress_percentage)
            my_cprint(f"Processing segment: {segment.text[:50]}... ({progress_percentage:.2f}% complete)", 'blue')

        sender_pipe.send(100)
        
        transcription = TranscribeFile.format_transcription(segments, include_timestamps)

        TranscribeFile.save_transcription(audio_file, transcription)
        del model
        my_cprint("Whisper model removed from memory.", 'red')

    @staticmethod
    def format_transcription(segments, include_timestamps):
        transcription = []
        for i, segment in enumerate(segments):
            if include_timestamps:
                start_time = TranscribeFile.format_time(segment.start)
                end_time = TranscribeFile.format_time(segment.end)
                formatted_segment = f"{start_time} - {end_time} {segment.text}"
                transcription.append(formatted_segment)
            else:
                transcription.append(segment.text)
        
        # Join the transcribed segments
        transcription_text = "\n".join(transcription)

        return transcription_text

    @staticmethod
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
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
        
        """
                                      def TranscribeFile
                          (Initialize transcription configuration)
                                         │
                                         │
                                         │
                                         ▼
                              def start_transcription_thread
                          (Starts a new transcription process)
                                         │
                                         │
                                         │
                                         ▼
                         ┌───────────────────────────────────────┐
                         │     def transcribe_worker             │
                         │ (Runs in a separate process for each  │
                         │  audio file, utilizing multiprocessing│
                         │  with specified CPU threads)          │
                         └───────────────────────────────────────┘
                                         │
                                         │
                                         │
                                         ▼
                       ┌───────────────────────────────────────┐
                       │   WhisperModel.transcribe             │
                       │ (Transcribes audio segments and sends │
                       │  progress updates through a pipe)     │
                       └───────────────────────────────────────┘
                                         │
                                         │
                                         │
                                         ▼
                          def format_transcription
                       (Formats transcription with timestamps)
                                         │
                                         │
                                         │
                                         ▼
                             def save_transcription
                      (Saves the transcribed text to a file)
"""
