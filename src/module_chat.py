import yaml
import logging
import gc
import copy
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import threading
from abc import ABC, abstractmethod

from constants import CHAT_MODELS, system_message, MODEL_MAX_TOKENS, MODEL_MAX_NEW_TOKENS
from utilities import my_cprint, has_bfloat16_support

# logging.getLogger("transformers").setLevel(logging.WARNING) # adjust to see deprecation and other non-fatal errors
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_model_settings(base_settings, attn_implementation):
    settings = copy.deepcopy(base_settings)
    settings['model_settings']['attn_implementation'] = attn_implementation
    return settings
    
def get_max_length(model_name):
    return MODEL_MAX_TOKENS.get(model_name, 8192)

def get_max_new_tokens(model_name):
    return MODEL_MAX_NEW_TOKENS.get(model_name, 1024)

def get_generation_settings(max_length, max_new_tokens):
    return {
        'max_length': max_length,
        'max_new_tokens': max_new_tokens,
        'do_sample': False,
        'num_beams': 1,
        'use_cache': True,
        'temperature': None,
        'top_p': None,
        'top_k': None,
    }

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
            bnb_4bit_quant_type="nf4",
        ),
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
        'attn_implementation': "sdpa"
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
            bnb_4bit_quant_type="nf4",
        ),
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
        'attn_implementation': "sdpa"
    }
}

def get_hf_token():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config.get('hf_access_token')
    return None

class BaseModel(ABC):
    def __init__(self, model_info, settings, generation_settings, attn_implementation=None, tokenizer_kwargs=None, model_kwargs=None):
        if attn_implementation:
            settings = get_model_settings(settings, attn_implementation)
        self.model_info = model_info
        self.settings = settings
        self.model_name = model_info['model']
        self.generation_settings = generation_settings
        self.max_length = generation_settings['max_length']
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        script_dir = Path(__file__).resolve().parent
        cache_dir = script_dir / "Models" / "chat" / model_info['cache_dir']

        # rewrite bfloat dictionary to float16 if cuda compute 8.6 not supported
        if self.device == "cuda" and not has_bfloat16_support():
            if 'bnb_bfloat16_settings' in settings:
                settings['bnb_float16_settings'] = settings.pop('bnb_bfloat16_settings')
                settings['bnb_float16_settings']['tokenizer_settings']['torch_dtype'] = torch.float16
                settings['bnb_float16_settings']['model_settings']['torch_dtype'] = torch.float16
                settings['bnb_float16_settings']['model_settings']['quantization_config'].bnb_4bit_compute_dtype = torch.float16

        hf_token = get_hf_token()

        tokenizer_settings = {
            **settings.get('tokenizer_settings', {}), 
            'cache_dir': str(cache_dir)
        }
        if tokenizer_kwargs:
            tokenizer_settings.update(tokenizer_kwargs)
        if hf_token:
            tokenizer_settings['use_auth_token'] = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_info['repo_id'], **tokenizer_settings)
        
        if tokenizer_kwargs and 'eos_token' in tokenizer_kwargs:
            self.tokenizer.eos_token = tokenizer_kwargs['eos_token']
        
        model_settings = {
            **settings.get('model_settings', {}), 
            'cache_dir': str(cache_dir)
        }
        if model_kwargs:
            model_settings.update(model_kwargs)

        # if using CPU, remove CUDA-specific settings
        # only applies to Zephyr 1.6b because all other models are not populated in combobox if cuda isn't available
        if self.device == "cpu":
            model_settings.pop('quantization_config', None)
            model_settings.pop('attn_implementation', None)
            model_settings['device_map'] = "cpu"

        if hf_token:
            model_settings['use_auth_token'] = hf_token

        self.model = AutoModelForCausalLM.from_pretrained(model_info['repo_id'], **model_settings)
        self.model.eval()
        
        my_cprint(f"Loaded {model_info['model']} on {self.device}", "green")

    def get_model_name(self):
        return self.model_name

    @abstractmethod
    def create_prompt(self, augmented_query):
        pass

    def create_inputs(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def generate_response(self, inputs):
        """
        Creates a TextIteratorStreamer to stream partial responses.
        """
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        eos_token_id = self.tokenizer.eos_token_id
        
        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        # generation + streamer require two threads to work
        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()

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

class Zephyr_1_6B(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Zephyr - 1.6b']
        settings = bnb_float16_settings if torch.cuda.is_available() else {}
        attn_implementation="eager" if torch.cuda.is_available() else {}
        super().__init__(model_info, settings, generation_settings, attn_implementation=attn_implementation)

    def create_prompt(self, augmented_query):
        return f"""<|system|>
{system_message}<|endoftext|>
<|user|>
{augmented_query}<|endoftext|>
<|assistant|>
"""

class Zephyr_3B(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Zephyr - 3b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|system|>
{system_message}<|endoftext|>
<|user|>
{augmented_query}<|endoftext|>
<|assistant|>
"""

class Qwen2_5_1_5b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 - 1.5b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

class QwenCoder_1_5b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 Coder - 1.5b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def generate_response(self, inputs):
        # Remove token_type_ids if it exists
        inputs.pop('token_type_ids', None)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        eos_token_id = self.tokenizer.eos_token_id

        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()


class Qwen2_5_3b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 - 3b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

class QwenCoder_3b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 Coder - 3b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def generate_response(self, inputs):
        # Remove token_type_ids if it exists
        inputs.pop('token_type_ids', None)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        eos_token_id = self.tokenizer.eos_token_id

        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()


class Llama_3_2_3b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Llama 3.2 - 3b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{augmented_query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""


class Phi3_5_mini_4b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Phi 3.5 Mini - 4b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings, attn_implementation="flash_attention_2")

    def create_prompt(self, augmented_query):
        return f"""<|system|>
{system_message}<|end|>
<|user|>
{augmented_query}<|end|>
<|assistant|>
"""


class MiniCPM3_4b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['MiniCPM3 - 4b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s><|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def create_inputs(self, prompt):
        inputs = super().create_inputs(prompt)
        inputs['pad_token_id'] = self.tokenizer.pad_token_id
        return inputs


class Qwen2_5_7b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""


class QwenCoder_7b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 Coder - 7b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def generate_response(self, inputs):
        # Remove token_type_ids if it exists
        inputs.pop('token_type_ids', None)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        eos_token_id = self.tokenizer.eos_token_id

        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()


class Dolphin_Llama3_1_8B(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Dolphin-Llama 3.1 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""


class Marco_o1_7b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Marco-o1 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system

你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.

## 重要！！！！！
当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。
<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。
<|im_end|>

<|im_start|>user
{augmented_query}<|im_end|>

<|im_start|>assistant
"""

    def generate_response(self, inputs):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        eos_token_id = self.tokenizer.eos_token_id
        
        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}
        
        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        # Buffer to accumulate text until we find <Output>
        buffer = ""
        output_started = False
        
        for partial_response in streamer:
            if not output_started:
                buffer += partial_response
                output_idx = buffer.find('<Output>')
                if output_idx != -1:
                    # Found the <Output> tag
                    output_started = True
                    # Yield everything after <Output>
                    remaining_text = buffer[output_idx + len('<Output>'):]
                    if remaining_text:
                        yield remaining_text
                    buffer = ""  # Clear the buffer
            else:
                # We're past <Output>, yield directly
                yield partial_response

        generation_thread.join()


class QwenCoder_14b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 Coder - 14b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def generate_response(self, inputs):
        # Remove token_type_ids if it exists
        inputs.pop('token_type_ids', None)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        eos_token_id = self.tokenizer.eos_token_id

        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()

class Qwen_2_5_14b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 - 14b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""


class Mistral_Small_22b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Mistral Small - 22b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s>
[INST] {system_message}

{augmented_query}[/INST]"""


class QwenCoder_32b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 Coder - 32b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def generate_response(self, inputs):
        # Remove token_type_ids if it exists
        inputs.pop('token_type_ids', None)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        eos_token_id = self.tokenizer.eos_token_id

        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()


class Qwen_2_5_32b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Qwen 2.5 - 32b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""


@torch.inference_mode()
def generate_response(model_instance, augmented_query):
    prompt = model_instance.create_prompt(augmented_query)
    inputs = model_instance.create_inputs(prompt)
    for partial_response in model_instance.generate_response(inputs):
        yield partial_response

def choose_model(model_name):
    if model_name in CHAT_MODELS:
        model_class_name = CHAT_MODELS[model_name]['function']
        model_class = globals()[model_class_name]
        
        # Get the relevant max_length
        max_length = get_max_length(model_name)

        # Get the relevant max_new_tokens
        max_new_tokens = get_max_new_tokens(model_name)

        # Generate the settings based on max_length and max_new_tokens
        generation_settings = get_generation_settings(max_length, max_new_tokens)
        
        # Pass the generation settings to the model constructor
        return model_class(generation_settings)
    else:
        raise ValueError(f"Unknown model: {model_name}")