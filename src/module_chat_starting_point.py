import gc
import logging
import warnings
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

from constants import CHAT_MODELS # system_message
from utilities import my_cprint, bnb_bfloat16_settings, bnb_float16_settings, generate_settings_4096

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

system_message = """Answer my question."""

class BaseModel:
    def __init__(self, model_info, settings, tokenizer_kwargs=None, model_kwargs=None, eos_token=None):
        self.model_info = model_info
        self.settings = settings
        self.model_name = model_info['model']

        script_dir = Path(__file__).resolve().parent
        cache_dir = script_dir / "Models" / "chat" / model_info['cache_dir']

        tokenizer_settings = {**settings['tokenizer_settings'], 'cache_dir': str(cache_dir)}
        if tokenizer_kwargs:
            tokenizer_settings.update(tokenizer_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_info['repo_id'], **tokenizer_settings)
        
        model_settings = {**settings['model_settings'], 'cache_dir': str(cache_dir)}
        self.model = AutoModelForCausalLM.from_pretrained(model_info['repo_id'], **model_settings)
        
        self.eos_token = eos_token

        my_cprint(f"Loaded {model_info['model']}", "green")

    def get_model_name(self):
        return self.model_name

    def create_inputs(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

    def generate_response(self, inputs):
        """
        Creates a TextIteratorStreamer to handle the streaming of generated text.

        Args:
            inputs (dict): A dictionary of inputs prepared for the model.

        Returns:
            str: The full generated response as a string.
        """
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        if self.eos_token:
            eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        else:
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.convert_tokens_to_ids("")
        
        all_settings = {**inputs, **generate_settings_4096, 'streamer': streamer, 'eos_token_id': eos_token_id}

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        full_response = ""
        print(end="", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text

        generation_thread.join()

        return full_response


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



class Dolphin_Llama3_8B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Llama 3 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        # return f"system\n{system_message}\nuser\n{user_message}\nassistant\n"
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


class Stablelm_2_12b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Stablelm 2 - 12b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"system\n{system_message}\nuser\n{user_message}\nassistant\n"


class Dolphin_Mistral_Nemo(BaseModel):
    # prints "begin_of_text|> " at the beginning of the response
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Mistral-Nemo - 12b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


# set pad token id to zero
class Dolphin_Phi3_Medium(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Phi 3 - Medium']
        super().__init__(model_info, bnb_bfloat16_settings, tokenizer_kwargs={'legacy': False})

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


# set pad token id to zero
class Dolphin_Qwen2_7b(BaseModel):
    # Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - 7b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


class Dolphin_Qwen2_0_5b(BaseModel):
    """
    Assistant: Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
    """
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - .5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


class Dolphin_Qwen2_1_5b(BaseModel):
    """
    Assistant: Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
    """
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Qwen 2 - 1.5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


# set pad token id to zero
class Dolphin_Yi_1_5_9b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Dolphin-Yi 1.5 - 9b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


# also uses an different common parameter
class InternLM2_20b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2 - 20b']
        tokenizer_kwargs = {'trust_remote_code': True}
        model_kwargs = {'trust_remote_code': True}
        super().__init__(model_info, bnb_bfloat16_settings, 
                         tokenizer_kwargs=tokenizer_kwargs, 
                         model_kwargs=model_kwargs,
                         eos_token="<|im_end|>")

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


class InternLM2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2 - 7b']
        tokenizer_kwargs = {'trust_remote_code': True}
        model_kwargs = {'trust_remote_code': True}
        super().__init__(model_info, bnb_bfloat16_settings, 
                         tokenizer_kwargs=tokenizer_kwargs, 
                         model_kwargs=model_kwargs,
                         eos_token="<|im_end|>")

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


class InternLM2_5_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2_5 - 7b']
        tokenizer_kwargs = {'trust_remote_code': True}
        model_kwargs = {'trust_remote_code': True}
        super().__init__(model_info, bnb_bfloat16_settings, 
                         tokenizer_kwargs=tokenizer_kwargs, 
                         model_kwargs=model_kwargs,
                         eos_token="<|im_end|>")

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


class InternLM2_1_8b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Internlm2 - 1.8b']
        tokenizer_kwargs = {'trust_remote_code': True}
        model_kwargs = {'trust_remote_code': True}
        super().__init__(model_info, bnb_bfloat16_settings, 
                         tokenizer_kwargs=tokenizer_kwargs, 
                         model_kwargs=model_kwargs,
                         eos_token="<|im_end|>")

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        

class Llama2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Llama 2 - 7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"


class Llama2_13b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Llama 2 - 13b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"


class Llama3_8B(BaseModel):
    # Assistant: Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
    def __init__(self):
        model_info = CHAT_MODELS['Llama 3 - 8b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|begin_of_text|><|start_header_id|>system\n{system_message}<|eot_id|><|start_header_id|>user\n{user_message}<|eot_id|><|start_header_id|>assistant\n"


class Neural_Chat_7b(BaseModel):
    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    def __init__(self):
        model_info = CHAT_MODELS['Neural-Chat - 7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"### System:\n{system_message}\n### User:\n{user_message}\n### Assistant: "


class Orca2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Orca 2 - 7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"


class Orca2_13b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Orca 2 - 13b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"


class Qwen1_5_0_5(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 1.5 - 0.5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"


class Qwen1_5_1_8b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 1.5 - 1.8B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"


class Qwen1_5_4b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 1.5 - 4B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"


class Qwen2_0_5b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 2 - 0.5b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"


class Qwen2_1_5b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 2 - 1.5B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"


class Qwen2_7b(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Qwen 2 - 7B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"


class SOLAR_10_7B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['SOLAR - 10.7b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<s>###System:\n{system_message}\n\n### User:\n\n{user_message}\n\n### Assistant:\n"


class Zephyr_1_6B(BaseModel):
    # Setting `pad_token_id` to `eos_token_id`:100257 for open-end generation.
    def __init__(self):
        model_info = CHAT_MODELS['Zephyr - 1.6b']
        super().__init__(model_info, bnb_float16_settings)

    def create_prompt(self, user_message):
        return f"<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{user_message}<|endoftext|>\n<|assistant|>"


class Zephyr_3B(BaseModel):
    # Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
    def __init__(self):
        model_info = CHAT_MODELS['Zephyr - 3b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{user_message}<|endoftext|>\n<|assistant|>"


class Stablelm_2_12b(BaseModel):
    # unexplained error the first time I tried to load, ran fine the second time.
    def __init__(self):
        model_info = CHAT_MODELS['Stablelm 2 - 12b']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


class H2O_Danube3_4B(BaseModel):
    # 'context_length': 8192,
    def __init__(self):
        model_info = CHAT_MODELS['H2O Danube3 4B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|prompt|>{user_message}</s><|answer|>"


class Yi_6B(BaseModel):
    def __init__(self):
        model_info = CHAT_MODELS['Yi 1.5 - 6B']
        super().__init__(model_info, bnb_bfloat16_settings)

    def create_prompt(self, user_message):
        return f"<|im_start|>system\n{system_message}\n<|im_end|>\n\n<|im_start|>user\n{user_message}\n<|im_end|>\n\n<|im_start|>assistant\n"


@torch.inference_mode()
def generate_response(model_instance, user_message):
    prompt = model_instance.create_prompt(user_message)
    inputs = model_instance.create_inputs(prompt)
    return model_instance.generate_response(inputs)

def choose_model(model_name):
    if model_name in CHAT_MODELS:
        model_class_name = CHAT_MODELS[model_name]['function']
        model_class = globals()[model_class_name]
        return model_class()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def test_model(model_name):
    model_instance = choose_model(model_name)
    print(f"Testing {model_instance.get_model_name()}...")

    user_message = "Tell me a short story about a brave knight."
    print("\nAssistant:\n\n", end="", flush=True)

    response = generate_response(model_instance, user_message)
    # print(response)

    model_instance.cleanup()

def main():
    print("Available models:")
    for i, model in enumerate(CHAT_MODELS.keys(), 1):
        print(f"{i}. {model}")

    choice = int(input("\nChoose a model number to test: ")) - 1
    model_name = list(CHAT_MODELS.keys())[choice]

    test_model(model_name)

if __name__ == "__main__":
    main()
