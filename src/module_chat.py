import gc
import logging
import warnings
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import threading
from abc import ABC, abstractmethod

from constants import CHAT_MODELS, system_message, MODEL_MAX_TOKENS, MODEL_MAX_NEW_TOKENS
from utilities import my_cprint, has_bfloat16_support

def get_max_length(model_name):
    return MODEL_MAX_TOKENS.get(model_name, 4096)

def get_max_new_tokens(model_name):
    return MODEL_MAX_NEW_TOKENS.get(model_name, 2048)

def get_generation_settings(max_length, max_new_tokens):
    return {
        'max_length': max_length,
        'max_new_tokens': max_new_tokens,
        'do_sample': False,
        'num_beams': 1,
        'use_cache': True,
        'temperature': None,
        'top_p': None,
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
            bnb_4bit_use_double_quant=True,
        ),
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
        'attn_implementation': "flash_attention_2"
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
            bnb_4bit_use_double_quant=True,
        ),
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
        'attn_implementation': "flash_attention_2"
    }
}
class BaseModel(ABC):
    def __init__(self, model_info, settings, generation_settings, tokenizer_kwargs=None, model_kwargs=None):
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

        tokenizer_settings = {
            **settings.get('tokenizer_settings', {}), 
            'cache_dir': str(cache_dir)
        }
        if tokenizer_kwargs:
            tokenizer_settings.update(tokenizer_kwargs)

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

        self.model = AutoModelForCausalLM.from_pretrained(model_info['repo_id'], **model_settings)
        
        my_cprint(f"Loaded {model_info['model']} on {self.device}", "green")

    def get_model_name(self):
        return self.model_name

    @abstractmethod
    def create_prompt(self, augmented_query):
        pass

    def create_inputs(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        if self.device == "cpu":
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


class Phi3_5_mini_4b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Phi 3.5 Mini - 4b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s><|system|>
        {system_message}<|end|>
        <|user|>
        {augmented_query}<|end|>
        <|assistant|>
        """


class Dolphin_Llama3_1_8B(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Dolphin-Llama 3.1 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class LongWriter_Llama_3_1(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['LongWriter Llama 3.1 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"<<SYS>>\n{system_message}\n<</SYS>>\n\n[INST]{augmented_query}[/INST]"

class Danube_3_4b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Danube 3 - 4b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|prompt|>{augmented_query}</s><|answer|>"""

class Hermes_3_Llama_3_1(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Hermes-3-Llama-3.1 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class Dolphin_Mistral_Nemo(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Dolphin-Mistral-Nemo - 12b']
        # tokenizer_kwargs = {'eos_token': "<|im_end|>"}
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class Dolphin_Phi3_Medium(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Dolphin-Phi 3 - Medium']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings, tokenizer_kwargs={'legacy': False})

    def create_prompt(self, augmented_query):
        return f"""<s><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class CodeQwen1_5_7b_chat(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['CodeQwen 1.5 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

    def generate_response(self, inputs):
        """
        Overrides the BaseModel method to handle model-specific kwargs.
        """
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


class Dolphin_Qwen2_7b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class Dolphin_Qwen2_1_5b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - 1.5b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class Dolphin_Yi_1_5_9b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Dolphin-Yi 1.5 - 9b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class Yi_Coder_9b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Yi Coder - 9b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|endoftext|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class InternLM2_5_7b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Internlm2_5 - 7b']
        tokenizer_kwargs = {'eos_token': "<|im_end|>"}
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings, tokenizer_kwargs=tokenizer_kwargs)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class InternLM2_5_20b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Internlm2_5 - 20b']
        tokenizer_kwargs = {'eos_token': "<|im_end|>"}
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings, tokenizer_kwargs=tokenizer_kwargs)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant
        """

class Llama2_13b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Llama 2 - 13b']
        super().__init__(model_info, bnb_float16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s>[INST]<<SYS>>
        {system_message}
        <</SYS>>

        {augmented_query}[/INST]"""


class DeepSeek_Coder_v2_lite(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['DeepSeek Coder v2 - 16b']
        super().__init__(model_info, bnb_float16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<｜begin▁of▁sentence｜>{system_message}
        User: {augmented_query}
        
        Assistant:"""


class Neural_Chat_7b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Neural-Chat - 7b']
        super().__init__(model_info, bnb_float16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""### System:
        {system_message}
        ### User:
        {augmented_query}
        ### Assistant:
        """

class Orca2_7b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Orca 2 - 7b']
        super().__init__(model_info, bnb_float16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant"""


class Orca2_13b(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Orca 2 - 13b']
        super().__init__(model_info, bnb_float16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {augmented_query}<|im_end|>
        <|im_start|>assistant"""

class SOLAR_10_7B(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['SOLAR - 10.7b']
        super().__init__(model_info, bnb_float16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s>### System:
        {system_message}

        ### User:
        {augmented_query}

        ### Assistant:
        """

class Zephyr_1_6B(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Zephyr - 1.6b']
        settings = bnb_float16_settings if torch.cuda.is_available() else {}
        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|system|>
        {system_message}<|endoftext|>
        <|user|>
        {augmented_query}<|endoftext|>
        <|assistant|>"""

class Zephyr_3B(BaseModel):
    def __init__(self, generation_settings):
        model_info = CHAT_MODELS['Zephyr - 3b']
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|system|>
        {system_message}<|endoftext|>
        <|user|>
        {augmented_query}<|endoftext|>
        <|assistant|>"""

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
        
        # Get the max_length for this model
        max_length = get_max_length(model_name)

        # Get the max_new_tokens for this model
        max_new_tokens = get_max_new_tokens(model_name)

        # Generate the settings based on both max_length and max_new_tokens
        generation_settings = get_generation_settings(max_length, max_new_tokens)
        
        # Pass the generation settings to the model constructor
        return model_class(generation_settings)
    else:
        raise ValueError(f"Unknown model: {model_name}")