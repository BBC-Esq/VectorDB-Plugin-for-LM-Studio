import datetime
import gc
import os
import platform
import time
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    LlamaTokenizer,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from langchain_community.docstore.document import Document
from extract_metadata import extract_image_metadata
from utilities import my_cprint
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")
warnings.filterwarnings("ignore", message="No module named 'triton'")
warnings.filterwarnings("ignore", module="xformers.*")
warnings.filterwarnings("ignore", module=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", module=".*transformers.models.llama.modeling_llama.*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", module=".*transformers.models.mistral.modeling_mistral.*")
        
def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    elif hasattr(torch.version, 'hip') and torch.version.hip and platform.system() == 'Linux':
        return 'cuda'
    else:
        return 'cpu'

def run_loader_in_process(loader_func):
    try:
        return loader_func()
    except Exception as e:
        my_cprint(f"Error processing images: {e}", "red")
        return []
        
class loader_cogvlm:
    def initialize_model_and_tokenizer(self):
        model_name = 'THUDM/cogvlm-chat-hf'
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model_settings = {
            'torch_dtype': torch.bfloat16,
            'resume_download': True,
            'low_cpu_mem_usage': True,
            'trust_remote_code': True,
        }

        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, **model_settings)

        my_cprint(f"Cogvlm model using 4-bit loaded into memory...", "green")
        return model, tokenizer

    def cogvlm_process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Docs_for_DB")
        documents = []
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

        image_files = [file for file in os.listdir(image_dir) if os.path.splitext(file)[1].lower() in allowed_extensions]

        if not image_files:
            return []

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        device = get_best_device()
        
        # initialize model and tokenizer
        model, tokenizer = self.initialize_model_and_tokenizer()

        print("Processing images...")
        
        total_start_time = time.time()

        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for file_name in image_files:
                full_path = os.path.join(image_dir, file_name)
                prompt = "Describe what this image depicts in as much detail as possible."

                try:
                    with Image.open(full_path).convert('RGB') as raw_image:
                        inputs = model.build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[raw_image])
                        if config['vision']['chosen_quant'] == '4-bit':
                            inputs = {
                                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                                'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                            }
                        elif config['vision']['chosen_quant'] == '8-bit':
                            inputs = {
                                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                                'images': [[inputs['images'][0].to(device).to(torch.float16)]],
                            }

                        gen_kwargs = {"max_length": 2048, "do_sample": False}
                        with torch.no_grad():
                            output = model.generate(**inputs, **gen_kwargs)
                            output = output[:, inputs['input_ids'].shape[1]:]
                            model_response = tokenizer.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[-1]

                            extracted_text = model_response
                            extracted_metadata = extract_image_metadata(full_path)
                            document = Document(page_content=extracted_text, metadata=extracted_metadata)
                            documents.append(document)

                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")

                progress_bar.update(1)

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Loaded {len(documents)} image(s)...")
        print(f"Total image processing time: {total_time_taken:.2f} seconds")

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        my_cprint("Cogvlm model removed from memory.", "red")

        return documents

class loader_llava:
    def initialize_model_and_tokenizer(self, config):
        chosen_model = config['vision']['chosen_model']
        chosen_size = config['vision']['chosen_size']
        chosen_quant = config['vision']['chosen_quant']
        
        model_id = ""
        if chosen_model == 'llava' and chosen_size == '7b':
            model_id = "llava-hf/llava-1.5-7b-hf"
        elif chosen_model == 'bakllava':
            model_id = "llava-hf/bakLlava-v1-hf"
        elif chosen_model == 'llava' and chosen_size == '13b':
            model_id = "llava-hf/llava-1.5-13b-hf"

        fp16_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        bf16_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        fp16_model_config = {
            'torch_dtype': torch.float16,
            'resume_download': True,
            'low_cpu_mem_usage': True,
        }
        
        bf16_model_config = {
            'torch_dtype': torch.bfloat16,
            'resume_download': True,
            'low_cpu_mem_usage': True,
        }
        
        device = get_best_device()

        if chosen_model == 'llava' and chosen_quant == 'float16':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                resume_download=True
            ).to(device)

        elif chosen_model == 'llava' and chosen_quant == '4-bit':
            model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=fp16_quant_config, **fp16_model_config)
            
        elif chosen_model == 'bakllava' and chosen_quant == 'float16':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                resume_download=True
            ).to(device)

        elif chosen_model == 'bakllava' and chosen_quant == '4-bit':
            model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=bf16_quant_config, **bf16_model_config)
        
        my_cprint(f"{chosen_model} {chosen_size} model using {chosen_quant} loaded into memory...", "green")
        
        processor = AutoProcessor.from_pretrained(model_id, resume_download=True)
        
        return model, processor, device, chosen_quant

    def llava_process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Docs_for_DB")
        documents = []
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

        image_files = [file for file in os.listdir(image_dir) if os.path.splitext(file)[1].lower() in allowed_extensions]

        if not image_files:
            return []

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        model, processor, device, chosen_quant = self.initialize_model_and_tokenizer(config)

        print("Processing images...")
        
        total_start_time = time.time()

        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for file_name in image_files:
                full_path = os.path.join(image_dir, file_name)
                prompt = "USER: <image>\nDescribe what this image depicts in as much detail as possible.\nASSISTANT:"

                try:
                    with Image.open(full_path) as raw_image:
                        inputs = processor(prompt, raw_image, return_tensors='pt').to(device)

                        if chosen_quant == 'bfloat16' and chosen_model == 'bakllava':
                            inputs = inputs.to(torch.bfloat16)
                        elif chosen_quant == 'float16':
                            inputs = inputs.to(torch.float16)
                        elif chosen_quant == '8-bit':
                            inputs = inputs.to(torch.float16)
                        elif chosen_quant == '4-bit':
                            inputs = inputs.to(torch.float32)

                        output = model.generate(**inputs, max_new_tokens=200, do_sample=True)
                        full_response = processor.decode(output[0][2:], skip_special_tokens=True, do_sample=True)
                        model_response = full_response.split("ASSISTANT: ")[-1]
                        extracted_text = model_response
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

        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        my_cprint("Llava/Bakllava model removed from memory.", "red")

        return documents

class loader_salesforce:
    def initialize_model_and_processor(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        chosen_quant = config['vision']['chosen_quant']
        device = get_best_device()
        
        if chosen_quant == 'float32':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        elif chosen_quant == 'float16':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
        
        my_cprint(f"Salesforce using {chosen_quant} loaded into memory...", "green")
        
        return model, processor, device

    def salesforce_process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Docs_for_DB")
        documents = []
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

        image_files = [file for file in os.listdir(image_dir) if os.path.splitext(file)[1].lower() in allowed_extensions]

        if not image_files:
            return []

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        model, processor, device = self.initialize_model_and_processor()
        
        print("Processing images...")
        
        total_start_time = time.time()

        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for file_name in image_files:
                full_path = os.path.join(image_dir, file_name)
                try:
                    with Image.open(full_path) as raw_image:
                        text = "an image of"
                        if config['vision']['chosen_quant'] == 'float32':
                            inputs = processor(raw_image, text, return_tensors="pt").to(device)
                        elif config['vision']['chosen_quant'] == 'float16':
                            inputs = processor(raw_image, text, return_tensors="pt").to(device, torch.float16)
                        output = model.generate(**inputs, max_new_tokens=100)
                        caption = processor.decode(output[0], skip_special_tokens=True)
                        extracted_metadata = extract_image_metadata(full_path)
                        document = Document(page_content=caption, metadata=extracted_metadata)
                        documents.append(document)
                        progress_bar.update(1)

                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Loaded {len(documents)} image(s)...")
        print(f"Total image processing time: {total_time_taken:.2f} seconds")

        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        my_cprint("Salesforce model removed from memory.", "red")

        return documents

class loader_moondream:
    def initialize_model_and_tokenizer(self):
        device = get_best_device()
        
        fp16_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        bf16_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        fp16_model_config = {
            'torch_dtype': torch.float16,
            'resume_download': True,
            'low_cpu_mem_usage': True,
        }
        
        bf16_model_config = {
            'torch_dtype': torch.bfloat16,
            'resume_download': True,
            'low_cpu_mem_usage': True,
        }

        model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", 
                                             trust_remote_code=True, 
                                             revision="2024-03-05", 
                                             torch_dtype=torch.float16, 
                                             low_cpu_mem_usage=True,
                                             resume_download=True).to(device)

        my_cprint(f"Moondream2 model using float16 loaded into memory...", "green")
        
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-03-05")
        
        return model, tokenizer, device
    
    def moondream_process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Docs_for_DB")
        documents = []
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']
        
        image_files = [file for file in os.listdir(image_dir) if os.path.splitext(file)[1].lower() in allowed_extensions]
        
        if not image_files:
            return []
            
        model, tokenizer, device = self.initialize_model_and_tokenizer()
        
        print("Processing images...")
        
        total_start_time = time.time()
        
        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for file_name in image_files:
                full_path = os.path.join(image_dir, file_name)
                try:
                    with Image.open(full_path) as raw_image:
                        enc_image = model.encode_image(raw_image)
                        summary = model.answer_question(enc_image, "Describe what this image depicts in as much detail as possible.", tokenizer)
                        extracted_metadata = extract_image_metadata(full_path)
                        document = Document(page_content=summary, metadata=extracted_metadata)
                        documents.append(document)
                        progress_bar.update(1)
                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")
                    
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Loaded {len(documents)} image(s)...")
        print(f"Total image processing time: {total_time_taken:.2f} seconds")

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        my_cprint("Moondream2 model removed from memory.", "red")
        
        return documents

def specify_image_loader():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    chosen_model = config["vision"]["chosen_model"]

    if chosen_model in ['llava', 'bakllava']:
        loader_func = loader_llava().llava_process_images
    elif chosen_model == 'cogvlm':
        loader_func = loader_cogvlm().cogvlm_process_images
    elif chosen_model == 'salesforce':
        loader_func = loader_salesforce().salesforce_process_images
    elif chosen_model == 'moondream2':
        loader_func = loader_moondream().moondream_process_images
    else:
        my_cprint("No valid image model specified in config.yaml", "red")
        return []

    with ProcessPoolExecutor(1) as executor:
        future = executor.submit(run_loader_in_process, loader_func)
        processed_docs = future.result()
        if processed_docs is None:
            return []
        return processed_docs
