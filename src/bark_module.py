import threading
import queue
from transformers import AutoProcessor, BarkModel
import torch
import numpy as np
import re
import time
import pyaudio
import gc

class BarkAudio:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(self.device)
        self.model = self.model.to_bettertransformer()
        # self.model.enable_cpu_offload()

        self.sentence_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.start_time = None

    def play_audio_thread(self):
        while True:
            queue_item = self.sentence_queue.get()
            if queue_item is None:
                break

            audio_array, sampling_rate, sentence_num = queue_item
            elapsed_time = time.time() - self.start_time
            print(f"({elapsed_time:.2f} seconds) Playing sentence #{sentence_num}")

            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, output=True)
            stream.write(audio_array.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()

        self.stop()

    def process_text_thread(self):
        sentence_count = 1
        while True:
            text_prompt = self.processing_queue.get()
            if text_prompt is None:
                break

            sentences = re.split(r'[.!?;]+', text_prompt)

            for sentence in sentences:
                if sentence.strip():
                    elapsed_time = time.time() - self.start_time
                    print(f"({elapsed_time:.2f} seconds) Processing sentence #{sentence_count}")
                    voice_preset = "v2/en_speaker_6"
                    
                    inputs = self.processor(text=sentence, voice_preset=voice_preset, return_tensors="pt")
                    
                    with torch.no_grad():
                        speech_output = self.model.generate(**inputs.to(self.device), do_sample=True)

                    audio_array = speech_output[0].cpu().numpy()
                    audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
                    sampling_rate = self.model.generation_config.sample_rate

                    self.sentence_queue.put((audio_array, sampling_rate, sentence_count))
                    sentence_count += 1

    def run(self):
        with open('chat_history.txt', 'r', encoding='utf-8') as file:
            llm_response = file.read()
            self.processing_queue.put(llm_response)

        self.start_time = time.time()

        processing_thread = threading.Thread(target=self.process_text_thread)
        playback_thread = threading.Thread(target=self.play_audio_thread)
        processing_thread.start()
        playback_thread.start()

        processing_thread.join()
        playback_thread.join()

    def stop(self):
        self.sentence_queue.put(None)

        self.release_resources()

    def release_resources(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    bark_audio = BarkAudio()
    bark_audio.run()

'''
INSTRUCTIONS:

(1) Bark consists of 4 models but only one is used at any given moment.  You can uncomment "self.model.enable_cpu_offload()"
    to put 3 models into RAM and only the one being used into VRAM.  This saves VRAM at a significant speed cost.

(2) Delete ", torch_dtype=torch.float16" verbatim to run the model in float32 instead of float16.  You must leave ".to(device)".

(3) You can comment out "model = model.to_bettertransformer()" to NOT use "better transformer," which is a library from Huggingface.
    Only do this if Better Transformers isn't compatible with your system, but it should be, and it provides a 5-20% speedup.

(4) Finally, to use the Bark full-size model remove "-small" on the two lines above; for example, it should read "suno/bark" instead.

*** You can experiment with any combination items (1)-(4) above to get the VRAM/speed/quality you want.  For example, using the
    full-size Bark models but only at float16...or using the "-small" models but at full float32. ***
'''

'''
INSTRUCTIONS:
                    
Go here for examples of different voices:

https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
'''