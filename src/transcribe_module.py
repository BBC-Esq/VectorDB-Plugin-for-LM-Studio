import whisper_s2t
import pickle
from multiprocessing import Process
from pathlib import Path
from extract_metadata import extract_audio_metadata
import subprocess
import av

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

class WhisperTranscriber:
    def __init__(self, model_identifier="ctranslate2-4you/whisper-large-v2-ct2-float16", batch_size=48, compute_type='float16'):
        self.model_identifier = model_identifier
        self.batch_size = batch_size
        self.compute_type = compute_type
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
            'model_identifier': model_identifier,
            'backend': 'CTranslate2',
        }

    def start_transcription_process(self, audio_file):
        self.audio_file = audio_file
        process = Process(target=self.transcribe_and_create_document)
        process.start()
        process.join()

    def transcribe_and_create_document(self):
        audio_file_str = str(self.audio_file)
        converted_audio_file = self.convert_to_wav(audio_file_str)
        self.model_kwargs['model_identifier'] = self.model_identifier
        model = whisper_s2t.load_model(**self.model_kwargs)
        transcription = self.transcribe(model, [str(converted_audio_file)])
        self.create_document_object(transcription, audio_file_str)

    def convert_to_wav(self, audio_file):
        output_file = Path(audio_file).stem + "_converted.wav"
        output_path = Path(__file__).parent / output_file
        
        container = av.open(audio_file)
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
        metadata = extract_audio_metadata(audio_file_path) # get metadata
        
        doc = Document(page_content=transcription_text, metadata=metadata)
        
        script_dir = Path(__file__).parent
        docs_dir = script_dir / "Docs_for_DB"
        docs_dir.mkdir(exist_ok=True)
        
        audio_file_name = Path(self.audio_file).stem
        pickle_file_path = docs_dir / f"{audio_file_name}.pkl"
        
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(doc, file)