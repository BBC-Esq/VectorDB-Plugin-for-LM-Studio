from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import time
import os
import yaml
from tqdm import tqdm
import datetime
from langchain.docstore.document import Document
import gc
import platform
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

def llava_process_images():
    script_dir = os.path.dirname(__file__)
    image_dir = os.path.join(script_dir, "Images_for_DB")
    documents = []

    if not os.listdir(image_dir):
        print("No files detected in the 'Images_for_DB' directory.")
        return

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

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

    # Load the model
    if chosen_model == 'llava' and chosen_quant == 'float16':
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
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

    my_cprint(f"Vision model loaded.", "green")

    processor = AutoProcessor.from_pretrained(model_id, resume_download=True)

    total_start_time = time.time()
    total_tokens = 0

    with tqdm(total=len(os.listdir(image_dir)), unit="image") as progress_bar:
        for file_name in os.listdir(image_dir):
            full_path = os.path.join(image_dir, file_name)
            prompt = "USER: <image>\nDescribe in detail what this image depicts in as much detail as possible.\nASSISTANT:"

            try:
                with Image.open(full_path) as raw_image:
                    if chosen_quant == 'bfloat16' and chosen_model == 'bakllava':
                        inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.bfloat16)
                    elif chosen_quant == 'float16':
                        inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16)
                    elif chosen_quant == '8-bit':
                        if chosen_model == 'llava':
                            inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16)
                        elif chosen_model == 'bakllava':
                            inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.bfloat16)
                    elif chosen_quant == '4-bit':
                        inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float32)

                    output = model.generate(**inputs, max_new_tokens=200, do_sample=True)
                    full_response = processor.decode(output[0][2:], skip_special_tokens=True, do_sample=True)  # can add num_beams=5
                    model_response = full_response.split("ASSISTANT: ")[-1]
                    
                    # Create a Document object
                    extracted_text = model_response
                    extracted_metadata = extract_image_metadata(full_path, file_name) # get metadata
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
    
    my_cprint(f"Vision model removed from memory.", "red")
    
    return documents
