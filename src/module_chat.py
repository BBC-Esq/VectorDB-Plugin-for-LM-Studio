import gc
import logging
import warnings
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

from constants import CHAT_MODELS
from utilities import my_cprint

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().setLevel(logging.WARNING)

system_message = """You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. Here are one or more contexts to solely base your answer off of. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."""

bnb_bfloat16_settings = {
    'tokenizer_settings': {
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
    },
    'model_settings': {
        'torch_dtype': torch.bfloat16,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
    }
}

bnb_float16_settings = {
    'tokenizer_settings': {
        'torch_dtype': torch.float16,
        'trust_remote_code': True,
    },
    'model_settings': {
        'torch_dtype': torch.float16,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
    }
}


common_generate_settings = {
    'max_length': 4096,
    'do_sample': False,
    'num_beams': 1,
    'use_cache': True,
    'temperature': None,
    #'top_k': 50,
    'top_p': None,
}

def extract_response_decorator(start_text):
    def decorator(func):
        def wrapper(self, model_response):
            start_idx = model_response.rfind(start_text) + len(start_text)
            return model_response[start_idx:].strip()
        return wrapper
    return decorator

class BaseModel:
    def __init__(self, model_info, settings):
        self.model_info = model_info
        self.settings = settings
        self.model_name = model_info['model']
        
        script_dir = Path(__file__).resolve().parent
        cache_dir = script_dir / "Models" / "chat" / model_info['cache_dir']
        
        tokenizer_settings = {**settings['tokenizer_settings'], 'cache_dir': str(cache_dir)}
        model_settings = {**settings['model_settings'], 'cache_dir': str(cache_dir)}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_info['repo_id'], **tokenizer_settings)
        self.model = AutoModelForCausalLM.from_pretrained(model_info['repo_id'], **model_settings)
        my_cprint(f"Loaded {model_info['model']}", "green")

    def get_model_name(self):
        return self.model_name

    def create_inputs(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, pad_token_id=self.tokenizer.eos_token_id)
        return generated_text

    def decode_response(self, generated_text):
        return self.tokenizer.decode(generated_text[0], skip_special_tokens=True)

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def switch_model(self, new_model_class):
        self.cleanup()
        return new_model_class()

def cleanup_resources(model, tokenizer):
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
# FACTORY

class Dolphin_Llama3_8B_Instruct(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Llama 3 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text

    def decode_response(self, generated_text):
        return self.tokenizer.decode(generated_text[0], skip_special_tokens=True)

class Dolphin_Mistral_Nemo(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Mistral-Nemo - 12b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        mistral_generate_settings = {**common_generate_settings, 'max_length': 8192}
        all_settings = {**inputs, **mistral_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text

    def decode_response(self, generated_text):
        return self.tokenizer.decode(generated_text[0], skip_special_tokens=True)

# set pad token id to zero
class Dolphin_Phi3_Medium(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Phi 3 - Medium']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text

    def decode_response(self, generated_text):
        return self.tokenizer.decode(generated_text[0], skip_special_tokens=True)


# set pad token id to zero
class Dolphin_Qwen2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text


class Dolphin_Qwen2_0_5b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - .5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text


class Dolphin_Qwen2_1_5b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - 1.5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text


# set pad token id to zero
class Dolphin_Yi_1_5_9b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Yi 1.5 - 9b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text


# also uses an different common parameter
class InternLM2_20b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2 - 20b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text


class InternLM2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text


class InternLM2_5_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2_5 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text


class InternLM2_1_8b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2 - 1.8b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return generated_text
        

class Llama2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Llama 2 - 7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"

    @extract_response_decorator("[/INST]")
    def extract_response(self, model_response):
        pass


class Llama2_13b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Llama 2 - 13b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"

    @extract_response_decorator("[/INST]")
    def extract_response(self, model_response):
        pass


class Llama3_8B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Llama 3 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|start_header_id|>system\n{system_message}<|eot_id|><|start_header_id|>user\n{user_message}<|eot_id|><|start_header_id|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        return generated_text


class Mistral7B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Mistral 0.3 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_info['repo_id'], add_prefix_space=None, **self.settings['tokenizer_settings'])

    def create_prompt(self, user_message):
        return f"[INST] {user_message} [/INST]*****"

    @extract_response_decorator("*****")
    def extract_response(self, model_response):
        pass

    def generate_response(self, inputs):
        all_settings = {**inputs, **common_generate_settings}
        generated_text = self.model.generate(**all_settings, pad_token_id=self.tokenizer.eos_token_id)
        return generated_text

    def decode_response(self, generated_text):
        return self.tokenizer.decode(generated_text[0], skip_special_tokens=True)


class Neural_Chat_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Neural-Chat - 7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"### System:\n{system_message}\n### User:\n{user_message}\n### Assistant: "

    @extract_response_decorator("### Assistant: ")
    def extract_response(self, model_response):
        pass


class Nous_Llama2_13b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Nous-Llama 2 - 13b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"

    @extract_response_decorator("[/INST]")
    def extract_response(self, model_response):
        pass

# set top p to none
class Orca2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Orca 2 - 7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass


class Orca2_13b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Orca 2 - 13b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass


# class Phi3_mini_4k(BaseModel):
    # def __init__(self):
        # model_info = CHAT_MODELS['Phi-3 Mini 4k - 3.8B']
        # super().__init__(model_info, bnb_bfloat16_settings)

    # def create_prompt(self, user_message):
        # return f"<s><|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|> *****\n"

    # @extract_response_decorator("*****")
    # def extract_response(self, model_response):
        # pass


# class Phi3_medium_4k(BaseModel):
    # def __init__(self):
        # model_info = CHAT_MODELS['Phi-3 Medium 4k - 14b']
        # super().__init__(model_info, bnb_bfloat16_settings)

    # def create_prompt(self, user_message):
        # return f"<s><|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|> *****\n"

    # @extract_response_decorator("*****")
    # def extract_response(self, model_response):
        # pass


class Qwen1_5_0_5(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 1.5 - 0.5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    @extract_response_decorator('assistant')
    def extract_response(self, model_response):
        pass


class Qwen1_5_1_8b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 1.5 - 1.8B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    @extract_response_decorator('assistant')
    def extract_response(self, model_response):
        pass


class Qwen1_5_4b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 1.5 - 4B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    @extract_response_decorator('assistant')
    def extract_response(self, model_response):
        pass


class Qwen2_0_5b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 2 - 0.5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    @extract_response_decorator('assistant')
    def extract_response(self, model_response):
        pass


class Qwen2_1_5b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 2 - 1.5B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    @extract_response_decorator('assistant')
    def extract_response(self, model_response):
        pass


class Qwen2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 2 - 7B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"

    @extract_response_decorator('assistant')
    def extract_response(self, model_response):
        pass


class SOLAR_10_7B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['SOLAR - 10.7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<s>###System:\n{system_message}\n\n### User:\n\n{user_message}\n\n### Assistant:\n"

    @extract_response_decorator("### Assistant:\n")
    def extract_response(self, model_response):
        pass


class Zephyr_1_6B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Zephyr - 1.6b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{user_message}<|endoftext|>\n<|assistant|>"

    @extract_response_decorator("<|assistant|>")
    def extract_response(self, model_response):
        pass


class Zephyr_3B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Zephyr - 3b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{user_message}<|endoftext|>\n<|assistant|>"

    @extract_response_decorator("<|assistant|>")
    def extract_response(self, model_response):
        pass


class Stablelm_2_12b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Stablelm 2 - 12b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    @extract_response_decorator("assistant")
    def extract_response(self, model_response):
        pass


# requires top p is none
# class Yi_6B(BaseModel):
    # def __init__(self):
        # model_info = CHAT_MODELS['Yi 1.5 - 6B']
        # super().__init__(model_info, bnb_bfloat16_settings)

    # def create_prompt(self, user_message):
        # return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant"

    # @extract_response_decorator("assistant")
    # def extract_response(self, model_response):
        # pass


# class Yi_9B(BaseModel):
    # def __init__(self):
        # model_info = CHAT_MODELS['Yi 1.5 - 9B']
        # super().__init__(model_info, bnb_bfloat16_settings)

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **self.settings['tokenizer_settings'])

    # def create_prompt(self, user_message):
        # return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant"

    # @extract_response_decorator("assistant")
    # def extract_response(self, model_response):
        # pass


def cleanup_resources(model, tokenizer):
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


@torch.inference_mode()
def generate_response(model_instance, user_message):
    prompt = model_instance.create_prompt(user_message)
    inputs = model_instance.create_inputs(prompt)
    generated_text = model_instance.generate_response(inputs)
    response = model_instance.decode_response(generated_text)
    final_response = model_instance.extract_response(response)
    return final_response


def choose_model(model_name):
    if model_name in CHAT_MODELS:
        model_class_name = CHAT_MODELS[model_name]['function']
        model_class = globals()[model_class_name]
        return model_class()
    else:
        raise ValueError(f"Unknown model: {model_name}")