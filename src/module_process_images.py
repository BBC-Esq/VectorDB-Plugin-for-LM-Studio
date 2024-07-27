import datetime
import gc
import os
import logging
import traceback
import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor, BlipForConditionalGeneration, BlipProcessor,
    LlamaTokenizer, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
)

from langchain_community.docstore.document import Document

from extract_metadata import extract_image_metadata
from utilities import my_cprint
from constants import VISION_MODELS

datasets_logger = logging.getLogger('datasets')
datasets_logger.setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger().setLevel(logging.WARNING)

# warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
# # logging.getLogger("transformers").setLevel(logging.CRITICAL)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("transformers").setLevel(logging.INFO)
# logging.getLogger("transformers").setLevel(logging.DEBUG)

ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "models" / "vision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

current_directory = Path(__file__).parent
VISION_DIR = current_directory / "models" / "vision"
VISION_DIR.mkdir(parents=True, exist_ok=True)

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def check_for_images(image_dir):
    return any(
        os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS
        for file in os.listdir(image_dir)
    )

def run_loader_in_process(loader_func):
    try:
        return loader_func()
    except Exception as e:
        error_message = f"Error processing images: {e}\n\nTraceback:\n{traceback.format_exc()}"
        my_cprint(error_message, "red")
        return []

def choose_image_loader():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    chosen_model = config["vision"]["chosen_model"]

    if chosen_model in ['llava 1.5 - 7b', 'bakllava 1.5 - 7b', 'llava 1.5 - 13b', ]:
        loader_func = loader_llava(config).process_images
    elif chosen_model == 'Cogvlm':
        loader_func = loader_cogvlm(config).process_images
    elif chosen_model == 'Moondream2':
        loader_func = loader_moondream(config).process_images
    elif chosen_model in ["Florence-2-large", "Florence-2-base"]:
        loader_func = loader_florence2(config).process_images
    elif chosen_model == 'Phi-3-vision-128k-instruct':
        loader_func = loader_phi3vision(config).process_images
    elif chosen_model == 'MiniCPM-Llama3-V-2_5-int4':
        loader_func = loader_minicpm_llama3v(config).process_images
    elif chosen_model in ['Llava 1.6 Vicuna - 7b', 'Llava 1.6 Vicuna - 13b']:
        loader_func = loader_llava_next(config).process_images
    else:
        my_cprint("No valid image model specified in config.yaml", "red")
        return []

    script_dir = os.path.dirname(__file__)
    image_dir = os.path.join(script_dir, "Docs_for_DB")

    if not check_for_images(image_dir):
        print("No images selected for processing...")
        return []

    with ProcessPoolExecutor(1) as executor:
        future = executor.submit(run_loader_in_process, loader_func)
        try:
            processed_docs = future.result()
        except Exception as e:
            my_cprint(f"Error occurred during image processing: {e}", "red")
            return []

        if processed_docs is None:
            return []
        return processed_docs


class BaseLoader:
    def __init__(self, config):
        self.config = config
        self.device = get_best_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    def initialize_model_and_tokenizer(self):
        raise NotImplementedError("Subclasses must implement initialize_model_and_tokenizer method")

    def process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Docs_for_DB")
        documents = []
        allowed_extensions = ALLOWED_EXTENSIONS

        image_files = [file for file in os.listdir(image_dir) if os.path.splitext(file)[1].lower() in allowed_extensions]

        self.model, self.tokenizer, self.processor = self.initialize_model_and_tokenizer()

        print("Processing images...")
        
        total_start_time = time.time()

        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for file_name in image_files:
                full_path = os.path.join(image_dir, file_name)
                try:
                    with Image.open(full_path) as raw_image:
                        extracted_text = self.process_single_image(raw_image)
                        extracted_metadata = extract_image_metadata(full_path)
                        document = Document(page_content=extracted_text, metadata=extracted_metadata)
                        documents.append(document)
                        progress_bar.update(1)
                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Loaded {len(documents)} image(s)...")
        print(f"Total image processing time: {total_time_taken:.2f} seconds")

        my_cprint("Vision model removed from memory.", "red")

        return documents

    def process_single_image(self, raw_image):
        raise NotImplementedError("Subclasses must implement process_single_image method")

class loader_cogvlm(BaseLoader):
    def initialize_model_and_tokenizer(self):
        model_name = 'THUDM/cogvlm-chat-hf'
        TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=TORCH_TYPE)

        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=TORCH_TYPE,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        my_cprint(f"Cogvlm vision model loaded into memory...", "green")
        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        prompt = "Describe this image in as much detail as possible while still trying to be succinct and not repeat yourself."
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[raw_image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
        }

        gen_kwargs = {"max_length": 2048, "do_sample": False}
        output = self.model.generate(**inputs, **gen_kwargs)
        output = output[:, inputs['input_ids'].shape[1]:]
        model_response = self.tokenizer.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[-1]
        return model_response

class loader_llava(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        precision = model_info['precision']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        
        my_cprint(f"{chosen_model} vision model loaded into memory...", "green")
        
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = "USER: <image>\nDescribe this image in as much detail as possible while still trying to be succinct and not repeat yourself.\nASSISTANT:"
        inputs = self.processor(prompt, raw_image, return_tensors='pt').to(self.device)
        inputs = inputs.to(torch.float32)

        output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        full_response = self.processor.decode(output[0][2:], skip_special_tokens=True, do_sample=False)
        model_response = full_response.split("ASSISTANT: ")[-1]
        return model_response


class loader_llava_next(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        precision = model_info['precision']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        
        my_cprint(f"{chosen_model} vision model loaded into memory...", "green")
        
        processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        
        return model, None, processor

    @ torch.inference_mode()
    def process_single_image(self, raw_image):
        user_prompt = "Describe this image in as much detail as possible while still trying to be succinct and not repeat yourself."
        prompt = f"USER: <image>\n{user_prompt} ASSISTANT:"
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)
        
        output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        response = self.processor.decode(output[0], skip_special_tokens=True) # possibly adjust to "full_response = self.processor.decode(output[0][2:], skip_special_tokens=True)" or something similar if output is preceded by special tokens inexplicatly
        model_response = response.split("ASSISTANT:")[-1].strip()
        
        return model_response


class loader_moondream(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_id = VISION_MODELS[chosen_model]['repo_id']
        cache_dir=VISION_DIR
        
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     trust_remote_code=True, 
                                                     revision="2024-07-23",
                                                     torch_dtype=torch.float16,
                                                     cache_dir=cache_dir,
                                                     low_cpu_mem_usage=True).to(self.device)

        my_cprint(f"Moondream2 vision model loaded into memory...", "green")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2024-07-23", cache_dir=cache_dir)
        
        return model, tokenizer, None
    
    @torch.inference_mode()
    def process_single_image(self, raw_image):
        enc_image = self.model.encode_image(raw_image)
        summary = self.model.answer_question(enc_image, "Describe what this image depicts in as much detail as possible.", self.tokenizer)
        return summary


class loader_florence2(BaseLoader):
    def __init__(self, config):
        super().__init__(config)
        from utilities import my_cprint, get_device_and_precision
        self.my_cprint = my_cprint
        self.get_device_and_precision = get_device_and_precision
        warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        repo_id = VISION_MODELS[chosen_model]["repo_id"]
        save_dir = VISION_MODELS[chosen_model]["cache_dir"]
        
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True, cache_dir=cache_dir)

        device_type, precision_type = self.get_device_and_precision()
        
        if device_type == "cuda":
            self.device = torch.device("cuda")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")
        
        if precision_type == "float16":
            model = model.half()
        elif precision_type == "bfloat16":
            model = model.bfloat16()
        
        self.my_cprint(f"{chosen_model} loaded with {precision_type}.", color="green")
        
        self.precision_type = precision_type
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt")
        
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.precision_type != "float32":
            inputs["pixel_values"] = inputs["pixel_values"].to(getattr(torch, self.precision_type))
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            early_stopping=False
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(raw_image.width, raw_image.height))
        
        return parsed_answer['<MORE_DETAILED_CAPTION>']


class loader_phi3vision(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        repo_id = VISION_MODELS[chosen_model]["repo_id"]
        save_dir = VISION_MODELS[chosen_model]["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_quant_type="nf4"
        )
        
        # microsoft/Phi-3-vision-128k-instruct
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation='flash_attention_2',
            quantization_config=quantization_config,
            cache_dir=cache_dir
        )
        
        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True, cache_dir=cache_dir)
        
        my_cprint(f"Microsoft-Phi-3-vision model loaded into memory...", "green")
        
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = f"""<|user|>
<|image_1|>
Describe this image in as much detail as possible while still trying to be succinct and not repeat yourself.<|end|>
<|assistant|>
"""
        inputs = self.processor(prompt, [raw_image], return_tensors="pt").to(self.device)
        
        generation_args = {
            "max_new_tokens": 500,
            "temperature": None,
            "do_sample": False,
        }
        
        generate_ids = self.model.generate(
            **inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args
        )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response


class loader_minicpm_llama3v(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        repo_id = VISION_MODELS[chosen_model]["repo_id"]
        save_dir = VISION_MODELS[chosen_model]["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # openbmb/MiniCPM-Llama3-V-2_5-int4
        model = AutoModel.from_pretrained(
            repo_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        model.eval()
        
        my_cprint(f"MiniCPM-Llama3-V vision model loaded into memory...", "green")
        
        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        question = 'Describe this image in as much detail as possible while still trying to be succinct and not repeat yourself.'
        msgs = [{'role': 'user', 'content': question}]
        
        response = self.model.chat(
            image=raw_image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=False,
            temperature=None
        )
        
        if isinstance(response, tuple) and len(response) == 3:
            res, context, _ = response
        else:
            res = response
        
        return res

'''
class loader_bunny(BaseLoader):
    def initialize_model_and_tokenizer(self):
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        
        #BAAI/Bunny-v1_1-4B
        # BAAI/Bunny-v1_1-Llama-3-8B-V
        
        chosen_model = self.config['vision']['chosen_model']
        model_path = VISION_MODELS[chosen_model]["model_path"]
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        my_cprint(f"Bunny vision model loaded into memory...", "green")
        
        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = "Describe what this image depicts in as much detail as possible."
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
        
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(self.device)
        
        image_tensor = self.model.process_images([raw_image], self.model.config).to(dtype=self.model.dtype, device=self.device)
        
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_length=4096,
            use_cache=True,
            repetition_penalty=1.0
        )[0].to(self.device)
        
        result = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return result
'''