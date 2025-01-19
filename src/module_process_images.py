import traceback
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor, AutoConfig,
    LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig, GenerationConfig
)

from langchain_community.docstore.document import Document

from extract_metadata import extract_image_metadata
from utilities import my_cprint
from constants import VISION_MODELS

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "models" / "vision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

current_directory = Path(__file__).parent
VISION_DIR = current_directory / "models" / "vision"
VISION_DIR.mkdir(parents=True, exist_ok=True)

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def check_for_images(image_dir):
    return any(
        Path(file).suffix.lower() in ALLOWED_EXTENSIONS
        for file in Path(image_dir).iterdir()
    )

def run_loader_in_process(loader_func):
    try:
        return loader_func()
    except Exception as e:
        error_message = f"Error processing images: {e}\n\nTraceback:\n{traceback.format_exc()}"
        my_cprint(error_message, "red")
        return []

def choose_image_loader():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    chosen_model = config["vision"]["chosen_model"]

    if chosen_model == 'Moondream2 - 1.9b':
        loader_func = loader_moondream(config).process_images
    elif chosen_model in ["Florence-2-large", "Florence-2-base"]:
        loader_func = loader_florence2(config).process_images
    elif chosen_model == 'MiniCPM-V-2_6 - 8b':
        loader_func = loader_minicpm_V_2_6(config).process_images        
    elif chosen_model in ['Llava 1.6 Vicuna - 7b', 'Llava 1.6 Vicuna - 13b']:
        loader_func = loader_llava_next(config).process_images
    elif chosen_model == 'THUDM glm4v - 9b':
        loader_func = loader_glmv4(config).process_images
    elif chosen_model == 'Molmo-D-0924 - 8b':
        loader_func = loader_molmo(config).process_images
    elif chosen_model == 'Mississippi - 2b':
        loader_func = loader_mississippi(config).process_images
    elif chosen_model == 'Ovis1.6-Llama3.2 - 3b':
        loader_func = loader_ovis(config).process_images
    elif chosen_model in ['InternVL2.5 - 1b', 'InternVL2.5 - 4b']:
        loader_func = loader_internvl2_5(config).process_images
    else:
        my_cprint("No valid image model specified in config.yaml", "red")
        return []

    script_dir = Path(__file__).parent
    image_dir = script_dir / "Docs_for_DB"

    if not check_for_images(image_dir):
        # print("No images selected to process...")
        return []

    with ProcessPoolExecutor(1) as executor:
        future = executor.submit(run_loader_in_process, loader_func)
        try:
            processed_docs = future.result()
        except Exception as e:
            my_cprint(f"Error occurred during image processing: {e}", "red")
            return []

        if processed_docs is None:
            return []
        return processed_docs


class BaseLoader:
    def __init__(self, config):
        self.config = config
        self.device = get_best_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    def initialize_model_and_tokenizer(self):
        raise NotImplementedError("Subclasses must implement initialize_model_and_tokenizer method")

    def process_images(self):
        script_dir = Path(__file__).parent
        image_dir = script_dir / "Docs_for_DB"
        documents = []
        allowed_extensions = ALLOWED_EXTENSIONS

        image_files = [file for file in image_dir.iterdir() if file.suffix.lower() in allowed_extensions]

        self.model, self.tokenizer, self.processor = self.initialize_model_and_tokenizer()

        print("Processing images...")
        
        total_start_time = time.time()

        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for file_name in image_files:
                full_path = image_dir / file_name
                try:
                    with Image.open(full_path) as raw_image:
                        extracted_text = self.process_single_image(raw_image)
                        extracted_metadata = extract_image_metadata(full_path)
                        document = Document(page_content=extracted_text, metadata=extracted_metadata)
                        documents.append(document)
                        progress_bar.update(1)
                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Loaded {len(documents)} image(s)...")
        print(f"Total image processing time: {total_time_taken:.2f} seconds")

        my_cprint("Vision model removed from memory.", "red")

        return documents

    def process_single_image(self, raw_image):
        raise NotImplementedError("Subclasses must implement.")


class loader_llava_next(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        model.eval()
        
        my_cprint(f"{chosen_model} vision model loaded into memory...", "green")

        processor = LlavaNextProcessor.from_pretrained(
            model_id, 
            cache_dir=cache_dir
        )

        return model, None, processor

    @ torch.inference_mode()
    def process_single_image(self, raw_image):
        user_prompt = "Describe this image in as much detail as possible but do not repeat yourself."
        prompt = f"USER: <image>\n{user_prompt} ASSISTANT:"
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)
        
        output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        # possibly adjust to "full_response = self.processor.decode(output[0][2:], skip_special_tokens=True)" or something similar if output is preceded by special tokens inexplicatly
        response = self.processor.decode(output[0], skip_special_tokens=True)
        model_response = response.split("ASSISTANT:")[-1].strip()
        
        return model_response


class loader_moondream(BaseLoader):
    """Most classes use CACHE_DIR consistently, but there's a slight inconsistency in the
    loader_moondream class, which uses VISION_DIR instead of CACHE_DIR.
    """
    def initialize_model_and_tokenizer(self):
        # moondream's approach uses the "vision" directory and does not create a nested folder like all other sub-classes
        chosen_model = self.config['vision']['chosen_model']
        model_id = VISION_MODELS[chosen_model]['repo_id']
        cache_dir=VISION_DIR
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            revision="2024-08-26",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        ).to(self.device)
        model.eval()

        my_cprint("Moondream2 vision model loaded into memory...", "green")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            revision="2024-08-26", 
            cache_dir=cache_dir
        )

        return model, tokenizer, None
    
    @torch.inference_mode()
    def process_single_image(self, raw_image):
        enc_image = self.model.encode_image(raw_image)
        summary = self.model.answer_question(enc_image, "Describe what this image depicts in as much detail as possible.", self.tokenizer)
        return summary


class loader_florence2(BaseLoader):
    def __init__(self, config):
        super().__init__(config)
        from utilities import my_cprint, get_device_and_precision
        self.my_cprint = my_cprint
        self.get_device_and_precision = get_device_and_precision

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        repo_id = VISION_MODELS[chosen_model]["repo_id"]
        save_dir = VISION_MODELS[chosen_model]["cache_dir"]
        
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            repo_id, 
            trust_remote_code=True, 
            low_cpu_mem_usage=True, 
            cache_dir=cache_dir
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(
            repo_id, 
            trust_remote_code=True, 
            cache_dir=cache_dir
        )

        device_type, precision_type = self.get_device_and_precision()
        
        if device_type == "cuda":
            self.device = torch.device("cuda")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")
        
        if precision_type == "float16":
            model = model.half()
        elif precision_type == "bfloat16":
            model = model.bfloat16()
        
        self.my_cprint(f"{chosen_model} loaded with {precision_type}.", color="green")
        
        self.precision_type = precision_type
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt")
        
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.precision_type != "float32":
            inputs["pixel_values"] = inputs["pixel_values"].to(getattr(torch, self.precision_type))
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            early_stopping=False,
            top_p=None,
            top_k=None,
            temperature=None,
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(raw_image.width, raw_image.height))
        
        return parsed_answer['<MORE_DETAILED_CAPTION>']


class loader_glmv4(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=cache_dir
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        my_cprint("GLM4V-9B vision model loaded into memory...", "green")

        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        query = "Describe this image in as much detail as possible but do not repeat yourself."
        
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": raw_image, "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = inputs.to(self.device)

        gen_kwargs = {
            "max_length": 512,
            "do_sample": False,
            "top_k": None,
            "top_p": None,
            "temperature": None
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return description


class loader_molmo(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]

        # Use local model path if specified; otherwise, use model ID
        model_path = model_info.get('model_path')
        model_id = model_info.get('repo_id')

        if model_path:
            model_source = model_path
        else:
            model_source = model_id

        cache_dir = CACHE_DIR / model_info.get('cache_dir', '')
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto',
                cache_dir=cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto',
                cache_dir=cache_dir
            )
            self.model.eval()

            my_cprint(f"{chosen_model} vision model loaded into memory...", "green")
        except Exception as e:
            my_cprint(f"Error loading {chosen_model} model: {str(e)}", "red")
            raise

        return self.model, None, self.processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        user_prompt = "Describe this image in detail as possible but be succinct and don't repeat yourself."
        inputs = self.processor.process(images=[raw_image], text=user_prompt)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

        try:
            generation_config = GenerationConfig(
                max_new_tokens=500,
                stop_strings=["<|endoftext|>"]
            )
            output = self.model.generate_from_batch(
                inputs,
                generation_config,
                tokenizer=self.processor.tokenizer
            )

            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except Exception as e:
            my_cprint(f"Error processing image: {str(e)}", "red")
            return ""

        return generated_text


class loader_mississippi(BaseLoader):
    def __init__(self, config):
        super().__init__(config)
        from utilities import my_cprint, get_device_and_precision
        self.my_cprint = my_cprint
        self.get_device_and_precision = get_device_and_precision

    def initialize_model_and_tokenizer(self):
        _, precision_type = self.get_device_and_precision()
        
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        repo_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        config = AutoConfig.from_pretrained(
            repo_id, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        config.hidden_act = "gelu"
        config.patch_size = 14
        config.image_size = 672
        config.min_dynamic_patch = 1
        config.max_dynamic_patch = 6
        
        model = AutoModel.from_pretrained(
            repo_id,
            config=config,
            torch_dtype=torch.bfloat16 if precision_type == "bfloat16" else torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir
        ).eval().cuda()
        
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, 
            trust_remote_code=True, 
            use_fast=False,
            cache_dir=cache_dir
        )
        
        self.my_cprint(f"H2OVL vision model loaded with {precision_type}.", color="green")
        
        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        image_path = raw_image.filename
        
        generation_config = dict(
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        question = '<image>\nDescribe this image in as much detail as possible but do not repeat yourself.'
        response, _ = self.model.chat(
            self.tokenizer, 
            image_path,
            question, 
            generation_config, 
            history=None, 
            return_history=True
        )
        
        return response


class loader_ovis(BaseLoader):
    def __init__(self, config):
        super().__init__(config)
        from utilities import my_cprint
        self.my_cprint = my_cprint

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        repo_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(
            repo_id, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            multimodal_max_length=4096,
            cache_dir=cache_dir
        ).eval().cuda()
        
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        
        return model, text_tokenizer, visual_tokenizer

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        text = "Describe this image in as much detail as possible but do not repeat yourself."
        query = f'<image>\n{text}'
        
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [raw_image])
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.processor.dtype, device=self.processor.device)]
        
        gen_kwargs = {
            'max_new_tokens': 1024,
            'do_sample': False,
            'top_p': None,
            'top_k': None,
            'temperature': None,
            'repetition_penalty': 1.0,
            'eos_token_id': self.model.generation_config.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'use_cache': True
        }
        
        output_ids = self.model.generate(
            input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask, 
            **gen_kwargs
        )[0]
        
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output


class loader_internvl2_5(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # use bnb with 4 model but run 1b model in native bfloat16
        if model_info['size'] == '4b':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModel.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            model.eval() # bnb automatically places model on cuda
        else:
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            model.eval().cuda()  # no bnb; must explicitly place model on cuda

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        my_cprint(f"InternVL2.5 {model_info['size']} vision model loaded into memory...", "green")

        return model, tokenizer, None

    def _build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def _prepare_image(self, raw_image, input_size=448):
        transform = self._build_transform(input_size=input_size)
        pixel_values = transform(raw_image).unsqueeze(0)
        return pixel_values

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        pixel_values = self._prepare_image(raw_image).to(torch.bfloat16).to(self.device)
        
        question = "Describe this image in as much detail as possible but do not repeat yourself."
        
        generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config
        )
        
        return response

"""
+----------------------+-----------------------------------------+-----------+----------------+----------+
| Sub-Class            | Config Details                          | Attention | Precision      | Device   |
+----------------------+-----------------------------------------+-----------+----------------+----------+
| loader_llava_next    | do_sample=False, no temperature control | SDPA      | float16 4-bit  | CUDA     |
| loader_moondream     | No generation config (uses answer_      | SDPA      | float16        | CUDA     |
|                      | question method)                        |           |                |          |
| loader_florence2     | Comprehensive beam settings, no         | SDPA      | autoselect     | CPU/CUDA |
|                      | sampling                                |           |                |          |
| loader_minicpm_V_2_6 | sampling=False, no temperature          | FA2       | bfloat16 4-bit | CUDA     |
| loader_glmv4         | do_sample=False, no temperature         | SDPA      | bfloat16 4-bit | CUDA     |
| loader_molmo         | Uses GenerationConfig class             | SDPA      | float32 4-bit  | CUDA     |
| loader_mississippi   | Uses repetition_penalty=1.1             | SDPA      | autoselect     | CUDA     |
| loader_ovis          | repetition_penalty=1.0, use_cache=True  | SDPA      | autoselect     | CUDA     |
+----------------------+-----------------------------------------+-----------+----------------+----------+
"""