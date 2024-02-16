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
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    LlamaTokenizer,
    LlavaForConditionalGeneration
)
from langchain.docstore.document import Document
from extract_metadata import extract_image_metadata
from utilities import my_cprint

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    elif hasattr(torch.version, 'hip') and torch.version.hip and platform.system() == 'Linux':
        return 'cuda'
    else:
        return 'cpu'

class loader_cogvlm:
    def initialize_model_and_tokenizer(self, config):
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

        if config['vision']['chosen_model'] == 'cogvlm' and config['vision']['chosen_quant'] == '4-bit':
            model = AutoModelForCausalLM.from_pretrained(
                'THUDM/cogvlm-chat-hf',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                load_in_4bit=True,
                resume_download=True
            )
            chosen_quant = "4-bit"
            
        elif config['vision']['chosen_model'] == 'cogvlm' and config['vision']['chosen_quant'] == '8-bit':
            model = AutoModelForCausalLM.from_pretrained(
                'THUDM/cogvlm-chat-hf',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                load_in_8bit=True,
                resume_download=True
            )
            chosen_quant = "8-bit"
            
        print("Selected model: cogvlm")
        print(f"Selected quant: {chosen_quant}")
        my_cprint("Vision model loaded.", "green")    
    
        return model, tokenizer

    def cogvlm_process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Images_for_DB")
        documents = []

        if not os.listdir(image_dir):
            print("No files detected in the 'Images_for_DB' directory.")
            return

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        device = get_best_device()
        print(f"Using device: {device}")
        model, tokenizer = self.initialize_model_and_tokenizer(config)

        total_start_time = time.time()

        with tqdm(total=len(os.listdir(image_dir)), unit="image") as progress_bar:
            for file_name in os.listdir(image_dir):
                full_path = os.path.join(image_dir, file_name)
                prompt = "Describe in detail what this image depicts in as much detail as possible."

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
                    print(f"{file_name}: Error processing image. Details: {e}")

                progress_bar.update(1)

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Total image processing time: {total_time_taken:.2f} seconds")

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        my_cprint("Vision model removed from memory.", "red")

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

        print(f"Selected model: {chosen_model}")
        print(f"Selected size: {chosen_size}")
        print(f"Selected quant: {chosen_quant}")

        device = get_best_device()
        print(f"Using device: {device}")

        if chosen_model == 'llava' and chosen_quant == 'float16':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                resume_download=True
            ).to(device)
        elif chosen_model == 'llava' and chosen_quant == '8-bit':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                resume_download=True
            )
        elif chosen_model == 'llava' and chosen_quant == '4-bit':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                resume_download=True
            )
        elif chosen_model == 'bakllava' and chosen_quant == 'float16':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                resume_download=True
            ).to(device)
        elif chosen_model == 'bakllava' and chosen_quant == '8-bit':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                resume_download=True
            )
        elif chosen_model == 'bakllava' and chosen_quant == '4-bit':
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                resume_download=True
            )

        processor = AutoProcessor.from_pretrained(model_id, resume_download=True)

        my_cprint("Vision model loaded.", "green")
        return model, processor, device, chosen_quant

    def llava_process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Images_for_DB")
        documents = []

        if not os.listdir(image_dir):
            print("No files detected in the 'Images_for_DB' directory.")
            return

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        model, processor, device, chosen_quant = self.initialize_model_and_tokenizer(config)

        total_start_time = time.time()
        total_tokens = 0

        with tqdm(total=len(os.listdir(image_dir)), unit="image") as progress_bar:
            for file_name in os.listdir(image_dir):
                full_path = os.path.join(image_dir, file_name)
                prompt = "USER: <image>\nDescribe in detail what this image depicts in as much detail as possible.\nASSISTANT:"

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

                        total_tokens += output[0].shape[0]
                        progress_bar.update(1)

                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Total image processing time: {total_time_taken:.2f} seconds")
        print(f"Tokens per second: {total_tokens / total_time_taken:.2f}")

        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        my_cprint("Vision model removed from memory.", "red")

        return documents

class loader_salesforce:
    def initialize_model_and_processor(self):
        device = get_best_device()
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

        return model, processor, device

    def salesforce_process_images(self):
        script_dir = os.path.dirname(__file__)
        image_dir = os.path.join(script_dir, "Images_for_DB")
        documents = []

        if not os.listdir(image_dir):
            print("No files detected in the 'Images_for_DB' directory.")
            return

        model, processor, device = self.initialize_model_and_processor()

        total_tokens = 0
        total_start_time = time.time()

        with tqdm(total=len(os.listdir(image_dir)), unit="image") as progress_bar:
            for file_name in os.listdir(image_dir):
                full_path = os.path.join(image_dir, file_name)
                try:
                    with Image.open(full_path) as raw_image:
                        inputs = processor(raw_image, return_tensors="pt").to(device)
                        output = model.generate(**inputs, max_new_tokens=50)
                        caption = processor.decode(output[0], skip_special_tokens=True)
                        total_tokens += output[0].shape[0]

                        extracted_metadata = extract_image_metadata(full_path)
                        document = Document(page_content=caption, metadata=extracted_metadata)
                        documents.append(document)

                        progress_bar.update(1)

                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Total image processing time: {total_time_taken:.2f} seconds")
        print(f"Tokens per second: {total_tokens / total_time_taken:.2f}")

        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        my_cprint("Vision model removed from memory.", "red")

        return documents