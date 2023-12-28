from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import time
import os
import sys
import textwrap
import yaml
import subprocess
import platform

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'has_mps') and torch.has_mps():
        return 'mps'
    elif hasattr(torch.version, 'hip') and torch.version.hip and platform.system() == 'Linux':
        return 'cuda'
    else:
        return 'cpu'

def llava_process_images():
    device = get_best_device()
    print(f"Selected device: {device}")
    
    script_dir = os.path.dirname(__file__)

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    test_image_path = config['vision']['test_image']

    if not test_image_path or test_image_path == '':
        print("You must choose a file to test process first.")
        return

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

    output_file_path = os.path.join(script_dir, "processed_images.txt")

    # Load the model
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

    print(f"Model: {chosen_model} {chosen_size}")
    print(f"Quant: {chosen_quant}")

    processor = AutoProcessor.from_pretrained(model_id, resume_download=True)

    total_start_time = time.time()
    total_tokens = 0

    try:
        with Image.open(test_image_path) as raw_image, open(output_file_path, "w", encoding="utf-8") as output_file:
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
            
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
            full_response = processor.decode(output[0][2:], skip_special_tokens=True, do_sample=True)# can specify num_beams
            model_response = full_response.split("ASSISTANT: ")[-1]
            wrapped_response = wrap_text(model_response, 130)
            output_file.write(f"{test_image_path}\n\n{wrapped_response}\n\n")

            total_tokens += output[0].shape[0]

    except Exception as e:
        print(f"Error processing file {test_image_path}: {e}")

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    tokens_per_second = total_tokens / total_time_taken
    print(f"Total time taken to process the image: {total_time_taken:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    try:
        if sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', output_file_path])
        elif sys.platform.startswith('darwin'):
            subprocess.Popen(['open', output_file_path])
        elif sys.platform.startswith('win32'):
            os.startfile(output_file_path)
    except Exception as e:
        print(f"Error opening file: {e}")

def wrap_text(text, width):
    return textwrap.fill(text, width=width, break_long_words=False, replace_whitespace=False)
