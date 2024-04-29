import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import time
import threading
from pathlib import Path
import logging

"""
# Set the logging level for transformers_modules logger
logging.getLogger("transformers_modules.microsoft--Phi-3-mini-4k-instruct.modeling_phi3").setLevel(logging.ERROR)

# Set the logging level for sentence_transformers logger
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tiledb.cloud.config", lineno=96)
"""

system_message = """
                     You are a helpful assistant who clearly and directly answers questions in a succinct fashion based on contexts provided to you.  Here are one or more contexts to solely base your answer off of.  If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer.  However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer.
                 """

bnb_bfloat16_settings = {
    'tokenizer_settings': {
        'torch_dtype': torch.bfloat16,
    },
    'model_settings': {
        'torch_dtype': torch.bfloat16,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        'resume_download': True,
        'low_cpu_mem_usage': True,
    }
}

bnb_float16_settings = {
    'tokenizer_settings': {
        'torch_dtype': torch.float16,
    },
    'model_settings': {
        'torch_dtype': torch.float16,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        'resume_download': True,
        'low_cpu_mem_usage': True,
    }
}

common_generate_settings = {
    'max_length': 4095,
    #'max_new_tokens': 500,
    'do_sample': False,
    'num_beams': 1, # 1 means no beam search.
    #'num_beam_groups': # number of groups to divide beams into to ensure diversity
    'use_cache': True, # whether to use past kv values to speedup, default true
    'temperature': None, # creativity
    #'top_k': 50, # Number of highest probability tokens to consider
    'top_p': None, # only consider tokens totaling at least this percentage probability
    #'num_return_sequences': 1, #Number of independently computed returned sequences
    #'pad_token_id':,
    #'bos_token_id':,
    #'eos_token_id':, #Optionally, use a list to set multiple
    #'generation_kwargs': Will be forwarded to the model's "generate" function.
}

@torch.inference_mode()
def StableLM_2_Zephyr_1_6B(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "stabilityai--stablelm-2-zephyr-1_6b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_float16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_float16_settings['model_settings'])
        return model, tokenizer

    # Use the provided model and tokenizer instances
    prompt = f"<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{user_message}<|endoftext|>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings, pad_token_id=tokenizer.eos_token_id)

    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind("<|assistant|>") + len("<|assistant|>")
    full_response = model_response[start_idx:].strip()

    return full_response

@torch.inference_mode()
def StableLM_Zephyr_3B(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "stabilityai--stablelm-2-zephyr-3b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_bfloat16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_bfloat16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{user_message}<|endoftext|>\n<|assistant|>"
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings, pad_token_id=tokenizer.eos_token_id)
    
    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind("<|assistant|>") + len("<|assistant|>")
    full_response = model_response[start_idx:].strip()
    
    return full_response

@torch.inference_mode()
def stablelm_2_12b_chat(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "stabilityai--stablelm-2-12b-chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_bfloat16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_bfloat16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{user_message}<|endoftext|>\n<|assistant|>"
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings, pad_token_id=tokenizer.eos_token_id)
    
    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind("<|assistant|>") + len("<|assistant|>")
    full_response = model_response[start_idx:].strip()
    
    return full_response

@torch.inference_mode()
def Llama_2_7b_chat_hf(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "meta-llama--Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_float16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_float16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings)
    
    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind("[/INST]") + len("[/INST]")
    full_response = model_response[start_idx:].strip()
    
    return full_response

@torch.inference_mode()
def Neural_Chat_7b_v3_3(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "Intel--neural-chat-7b-v3-3"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_float16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_float16_settings['model_settings'])
        return model, tokenizer

    prompt = f"### System:\n{system_message}\n### User:\n{user_message}\n### Assistant: "
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings, pad_token_id=tokenizer.eos_token_id)
    
    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.find("### Assistant: ") + len("### Assistant: ")
    full_response = model_response[start_idx:].strip()
    
    return full_response

@torch.inference_mode()
def Phi_3_mini_4k_instruct(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "microsoft--Phi-3-mini-4k-instruct"

        modified_bnb_bfloat16_settings = {
            **bnb_bfloat16_settings,
            'model_settings': {
                **bnb_bfloat16_settings['model_settings'],
                'trust_remote_code': True
            }
        }

        tokenizer = AutoTokenizer.from_pretrained(model_name, **modified_bnb_bfloat16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **modified_bnb_bfloat16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<s><|system|>{system_message}<|end|>\n<|user|>\n{user_message} *****<|end|>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

    all_settings = {**inputs, **common_generate_settings}

    generated_text = model.generate(**all_settings, pad_token_id=tokenizer.eos_token_id)

    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind("*****") + len("*****")
    full_response = model_response[start_idx:].strip()

    return full_response

@torch.inference_mode()
def Qwen1_5_0_5_chat(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "Qwen--Qwen1.5-0.5B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_bfloat16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_bfloat16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings, pad_token_id=tokenizer.eos_token_id)

    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind('assistant') + len('assistant')
    full_response = model_response[start_idx:].strip()

    return full_response

@torch.inference_mode()
def Qwen1_5_1_8b_chat(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "Qwen--Qwen1.5-1.8B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_bfloat16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_bfloat16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings, pad_token_id=tokenizer.eos_token_id)

    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind('assistant') + len('assistant')
    full_response = model_response[start_idx:].strip()

    return full_response

@torch.inference_mode()
def SOLAR_10_7B_Instruct_v1_0(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "upstage--SOLAR-10.7B-Instruct-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_float16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_float16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<s>###System:\n{system_message}\n\n### User:\n\n{user_message}\n\n### Assistant:\n"

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings)

    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.find("### Assistant:\n") + len("### Assistant:\n")
    full_response = model_response[start_idx:].strip()

    return full_response

@torch.inference_mode()
def Dolphin_Llama_3_8B_Instruct(user_message, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        script_dir = Path(__file__).resolve().parent
        model_name = script_dir / "Models" / "cognitivecomputations--dolphin-2.9-llama3-8b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, **bnb_bfloat16_settings['tokenizer_settings'])
        model = AutoModelForCausalLM.from_pretrained(model_name, **bnb_bfloat16_settings['model_settings'])
        return model, tokenizer

    prompt = f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    all_settings = {**inputs, **common_generate_settings}
    generated_text = model.generate(**all_settings, eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))

    model_response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    start_idx = model_response.rfind("<|im_start|>assistant") + len("<|im_start|>assistant")
    full_response = model_response[start_idx:].strip()

    return full_response

def cleanup_resources(model, tokenizer):
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()