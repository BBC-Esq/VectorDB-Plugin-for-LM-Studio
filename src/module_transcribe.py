import os
import pickle
import subprocess
from multiprocessing import Process
from pathlib import Path
import warnings

import torch
import av
from langchain_community.docstore.document import Document

import whisper_s2t
from whisper_s2t.backends.ctranslate2.hf_utils import download_model
from extract_metadata import extract_audio_metadata
from constants import WHISPER_MODELS

warnings.filterwarnings("ignore")

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "Models" / "whisper"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class WhisperTranscriber:
    def __init__(self, model_key, batch_size):
        model_info = WHISPER_MODELS[model_key]
        self.model_identifier = model_info['repo_id']
        self.compute_type = model_info['precision']
        self.batch_size = batch_size
        self.cache_dir = str(CACHE_DIR)

        script_dir = Path(__file__).parent
        self.model_dir = script_dir / "Models" / "whisper"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_kwargs = {
            'compute_type': self.compute_type,
            'asr_options': {
                "beam_size": 5,
                "best_of": 1,
                "patience": 2,
                "length_penalty": 1,
                "repetition_penalty": 1.01,
                "no_repeat_ngram_size": 0,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.5,
                "prefix": None,
                "suppress_blank": True,
                "suppress_tokens": [-1],
                "without_timestamps": True,
                "max_initial_timestamp": 1.0,
                "word_timestamps": False,
                "sampling_temperature": 1.0,
                "return_scores": True,
                "return_no_speech_prob": True,
                "word_aligner_model": 'tiny',
            },
            'model_identifier': self.model_identifier,
            'backend': 'CTranslate2',
        }

        if 'large-v3' in self.model_identifier:
            self.model_kwargs['n_mels'] = 128

    def start_transcription_process(self, audio_file):
        self.audio_file = audio_file
        process = Process(target=self.transcribe_and_create_document)
        process.start()
        process.join()


    @torch.inference_mode()
    def transcribe_and_create_document(self):
        audio_file_str = str(self.audio_file)
        converted_audio_file = self.convert_to_wav(audio_file_str)
        
        try:
            downloaded_path = download_model(
                size_or_id=self.model_identifier,
                cache_dir=str(CACHE_DIR)
            )
            
            model_kwargs = self.model_kwargs.copy()
            model_kwargs.pop('model_identifier', None)
            model_kwargs.pop('cache_dir', None)
            
            model = whisper_s2t.load_model(
                model_identifier=downloaded_path,
                **model_kwargs
            )
            
        except Exception as e:
            print(f"Error loading model {self.model_identifier}: {e}")
            raise

        transcription = self.transcribe(model, [str(converted_audio_file)])
        self.create_document_object(transcription, audio_file_str)

        script_dir = Path(__file__).parent
        converted_audio_file_name = f"{Path(audio_file_str).stem}_converted.wav"
        converted_audio_file_full_path = script_dir / converted_audio_file_name

        if converted_audio_file_full_path.exists():
            try:
                converted_audio_file_full_path.unlink()
            except Exception as e:
                print(f"Error deleting file {converted_audio_file_full_path}: {e}")
        else:
            print(f"File does not exist: {converted_audio_file_full_path}")

    def convert_to_wav(self, audio_file):
        output_file = f"{Path(audio_file).stem}_converted.wav"
        output_path = Path(__file__).parent / output_file
        
        with av.open(audio_file) as container:
            stream = next(s for s in container.streams if s.type == 'audio')
            
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=16000,
            )
            
            output_container = av.open(str(output_path), mode='w')
            output_stream = output_container.add_stream('pcm_s16le', rate=16000)
            output_stream.layout = 'mono'
            
            for frame in container.decode(audio=0):
                frame.pts = None
                resampled_frames = resampler.resample(frame)
                if resampled_frames is not None:
                    for resampled_frame in resampled_frames:
                        for packet in output_stream.encode(resampled_frame):
                            output_container.mux(packet)
            
            for packet in output_stream.encode(None):
                output_container.mux(packet)
            
            output_container.close()
        
        return str(output_path)

    def transcribe(self, model, files, lang_codes=['en'], tasks=['transcribe'], initial_prompts=[None]):
        out = model.transcribe_with_vad(files,
                                        lang_codes=lang_codes,
                                        tasks=tasks,
                                        initial_prompts=initial_prompts,
                                        batch_size=self.batch_size)
        transcription = " ".join([_['text'] for _ in out[0]]).strip()
        return transcription

    def create_document_object(self, transcription_text, audio_file_path):
        metadata = extract_audio_metadata(audio_file_path)
        
        doc = Document(page_content=transcription_text, metadata=metadata)
        
        script_dir = Path(__file__).parent
        docs_dir = script_dir / "Docs_for_DB"
        docs_dir.mkdir(exist_ok=True)
        
        audio_file_name = Path(audio_file_path).stem
        json_file_path = docs_dir / f"{audio_file_name}.json"
        
        json_file_path.write_text(doc.json(indent=4), encoding='utf-8')
            
        script_dir = Path(__file__).parent
        converted_audio_file_name = f"{Path(audio_file_path).stem}_converted.wav"
        converted_audio_file_full_path = script_dir / converted_audio_file_name

        if converted_audio_file_full_path.exists():
            try:
                converted_audio_file_full_path.unlink()
            except Exception as e:
                print(f"Error deleting file {converted_audio_file_full_path}: {e}")
        else:
            print(f"File does not exist: {converted_audio_file_full_path}")