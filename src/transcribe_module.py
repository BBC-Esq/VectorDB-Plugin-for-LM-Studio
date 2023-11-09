import os
import yaml
from faster_whisper import WhisperModel
import time
from termcolor import cprint

class TranscribeFile:
    def __init__(self, config_path='config.yaml'):
        # Read the configuration from config.yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['transcribe_file']

        model_name = f"ctranslate2-4you/whisper-{config['model']}-ct2-{config['quant']}"
        self.audio_file = config['file']
        self.include_timestamps = config['timestamps']
        self.model = WhisperModel(model_name, device=config['device'], compute_type=config['quant'])
        self.enable_print = True
        self.my_cprint("Whisper model loaded", "green")

    def my_cprint(self, *args, **kwargs):
        if self.enable_print:
            filename = "transcribe_module.py"
            modified_message = f"{filename}: {args[0]}"
            cprint(modified_message, *args[1:], **kwargs)

    @staticmethod
    def format_time(seconds):
        # Converts seconds to 'hours:minutes:seconds' format if more than 59m59s
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def transcribe(self, audio_file, output_file):
        segments, _ = self.model.transcribe(audio_file)
        transcription = []
        
        for segment in segments:
            if self.include_timestamps:
                start_time = self.format_time(segment.start)
                end_time = self.format_time(segment.end)
                transcription.append(f"{start_time} - {end_time} {segment.text}")
            else:
                transcription.append(segment.text)
        
        transcription_text = "\n".join(transcription)
            
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(transcription_text)
        
        return transcription_text

    def transcribe_to_file(self):
        if not os.path.isfile(self.audio_file):
            raise FileNotFoundError(f"Error: {self.audio_file} does not exist.")
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        docs_dir = os.path.join(script_dir, "Docs_for_DB")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)

        base_audio_file_name = os.path.basename(self.audio_file)
        base_file_name_without_ext = os.path.splitext(base_audio_file_name)[0]
        transcribed_file_name = f"transcribed_{base_file_name_without_ext}.txt"
        output_file = os.path.join(docs_dir, transcribed_file_name)

        start_time = time.time()
        transcription_text = self.transcribe(self.audio_file, output_file)
        end_time = time.time()
        
        duration = end_time - start_time
        self.my_cprint(f"Transcription took {duration:.2f} seconds.", "yellow")

        return transcription_text

# For standalone run
if __name__ == "__main__":
    transcriber = TranscribeFile()
    try:
        transcription = transcriber.transcribe_to_file()
        print("Transcription completed.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

