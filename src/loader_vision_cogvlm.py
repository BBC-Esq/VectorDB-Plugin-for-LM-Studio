from PIL import Image
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
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

def initialize_model_and_tokenizer(config):
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
    my_cprint(f"Vision model loaded.", "green")    
    
    return model, tokenizer

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.version, 'hip') and torch.version.hip and platform.system() == 'Linux':
        return 'cuda'
    else:
        return 'cpu'

def cogvlm_process_images():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    script_dir = os.path.dirname(__file__)
    image_dir = os.path.join(script_dir, "Images_for_DB")
    documents = []

    if not os.listdir(image_dir):
        print("No files detected in the 'Images_for_DB' directory.")
        return

    device = get_best_device()
    print(f"Using device: {device}")
    model, tokenizer = initialize_model_and_tokenizer(config)

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

                        # Creating a Document object
                        extracted_text = model_response
                        extracted_metadata = extract_image_metadata(full_path, file_name) # get metadata
                        document = Document(page_content=extracted_text, metadata=extracted_metadata)
                        documents.append(document)

            except Exception as e:
                print(f"{file_name}: Error processing image. Details: {e}")

            progress_bar.update(1)

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    print(f"Total image processing time: {total_time_taken:.2f} seconds")

    # cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    my_cprint(f"Vision model removed from memory.", "red")

    return documents
