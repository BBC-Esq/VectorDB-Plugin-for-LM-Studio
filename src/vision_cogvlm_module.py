from PIL import Image
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
import time
import os
import textwrap
import yaml
import gc
from tqdm import tqdm
import subprocess
import sys
import platform

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif 'mps' in torch.backends.backends and torch.backends.mps.is_available():
        return 'mps'
    elif hasattr(torch.version, 'hip') and torch.version.hip and platform.system() == 'Linux':
        return 'cuda'
    else:
        return 'cpu'

def initialize_model_and_tokenizer(config):
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

    if config['vision']['chosen_model'] == 'cogvlm' and config['vision']['chosen_quant'] == '4-bit':
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True,
            resume_download=True
        )
        
    elif config['vision']['chosen_model'] == 'cogvlm' and config['vision']['chosen_quant'] == '8-bit':
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_8bit=True,
            resume_download=True
        )
    return model, tokenizer

def cogvlm_process_images():
    device = get_best_device()
    print(f"Selected device: {device}")
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    script_dir = os.path.dirname(__file__)
    test_image_path = config['vision']['test_image']

    if not test_image_path or test_image_path == '':
        print("You must choose a file to test process first.")
        return

    model, tokenizer = initialize_model_and_tokenizer(config)

    print(f"Model: {config['vision']['chosen_model']} {config['vision']['chosen_size']}")
    print(f"Quant: {config['vision']['chosen_quant']}")

    output_file_path = os.path.join(script_dir, "processed_images.txt")
    
    total_start_time = time.time()
    total_tokens = 0

    try:
        with Image.open(test_image_path).convert('RGB') as raw_image, open(output_file_path, "w", encoding="utf-8") as output_file:
            prompt = "Describe in detail what this image depicts."

            inputs = model.build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[raw_image])
            if config['vision']['chosen_quant'] == '4-bit':
                inputs = {
                    'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                    'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                    'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
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
                wrapped_response = wrap_text(model_response, 130)

            output_file.write(f"{test_image_path}\n\n{wrapped_response}\n\n")
            total_tokens += len(output[0])

    except Exception as e:
        print(f"Error processing file {test_image_path}: {e}")

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    tokens_per_second = total_tokens / total_time_taken
    print(f"Total time taken to process the image: {total_time_taken:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    try:
        if os.name == 'nt':
            os.startfile(output_file_path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', output_file_path])
        elif sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', output_file_path])
        else:
            raise NotImplementedError("Unsupported operating system")
    except Exception as e:
        print(f"Error opening file: {e}")

def wrap_text(text, width):
    return textwrap.fill(text, width=width, break_long_words=False, replace_whitespace=False)
