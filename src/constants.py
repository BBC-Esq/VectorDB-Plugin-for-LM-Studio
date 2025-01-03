jeeves_system_message = "You are a helpful British butler who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address a question you answer based on what the contexts say and then briefly summarize the parts of the question that the contexts didn't provide an answer to.  Also, you should be very respectful to the person asking the question and frequently offer traditional butler services like various fancy drinks, snacks, various butler services like shining of shoes, pressing of suites, and stuff like that. Also, if you can't answer the question at all based on the provided contexts, you should apologize profusely and beg to keep your job.  Lastly, it is essential that if there are no contexts actually provided it means that a user's question wasn't relevant and you should state that you can't answer based off of the contexts because there are none.  And it goes without saying you should refuse to answer any questions that are not directly answerable by the provided contexts.  Moreover, some of the contexts might not have relevant information and you shoud simply ignore them and focus on only answering a user's question.  I cannot emphasize enought that you must gear your answer towards using this program and based your response off of the contexts you receive."
system_message = "You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."
rag_string = "Here are the contexts to base your answer on.  However, I need to reiterate that I only want you to base your response on these contexts and do not use outside knowledge that you may have been trained with."

# overrides default max_length parameter of 8192
MODEL_MAX_TOKENS = {
    'Qwen - 1.5b': 4096,
    'Zephyr - 1.6b': 4096,
    'Granite - 2b': 4096,
    'Qwen Coder - 1.5b': 4096,
    'Zephyr - 3b': 4096,
    'Qwen Coder - 3b': 4096,
}

# overrides max_new_tokens parameter of 1024
MODEL_MAX_NEW_TOKENS = {
    'Qwen - 1.5b': 512,
    'Zephyr - 1.6b': 512,
    'Qwen Coder - 1.5b': 512,
}

CHAT_MODELS = {
    'Qwen - 1.5b': {
        'model': 'Qwen - 1.5b',
        'repo_id': 'Qwen/Qwen2.5-1.5B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-1.5B-Instruct',
        'cps': 261.31,
        'context_length': 32768,
        'vram': 1749.97,
        'function': 'Qwen_1_5b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen Coder - 1.5b': {
        'model': 'Qwen Coder - 1.5b',
        'repo_id': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-Coder-1.5B-Instruct',
        'cps': 236.32,
        'context_length': 4096,
        'vram': 1742.12,
        'function': 'QwenCoder_1_5b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Granite - 2b': {
        'model': 'Granite - 2b',
        'repo_id': 'ibm-granite/granite-3.1-2b-instruct',
        'cache_dir': 'ibm-granite--granite-3.1-2b-instruct',
        'cps': 128.11,
        'context_length': 8192,
        'vram': 2292.18,
        'function': 'Granite_2b',
        'precision': 'bfloat16', # have float32 version
        'gated': False,
    },
    'Zephyr - 1.6b': {
        'model': 'Zephyr - 1.6b',
        'repo_id': 'stabilityai/stablelm-2-zephyr-1_6b',
        'cache_dir': 'stabilityai--stablelm-2-zephyr-1_6b',
        'cps': 375.77,
        'context_length': 4096,
        'vram': 2233.45,
        'function': 'Zephyr_1_6B',
        'precision': 'float16', # doesn't have a float32 version
        'gated': False,
    },
    'Zephyr - 3b': {
        'model': 'Zephyr - 3b',
        'repo_id': 'stabilityai/stablelm-zephyr-3b',
        'cache_dir': 'stabilityai--stablelm-zephyr-3b',
        'cps': 293.68,
        'context_length': 4096,
        'vram': 2733.85,
        'function': 'Zephyr_3B',
        'precision': 'bfloat16', # have float32 version
        'gated': False,
    },
    'Exaone - 2.4b': {
        'model': 'Exaone - 2.4b',
        'repo_id': 'LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct',
        'cache_dir': 'LGAI-EXAONE--EXAONE-3.5-2.4B-Instruct',
        'cps': 224.08,
        'context_length': 32768,
        'vram': 2821.06,
        'function': 'Exaone_2_4b',
        # 'precision': 'float32',
        'gated': False,
    },
    'Qwen Coder - 3b': {
        'model': 'Qwen Coder - 3b',
        'repo_id': 'Qwen/Qwen2.5-Coder-3B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-Coder-3B-Instruct',
        'cps': 198.99,
        'context_length': 32768,
        'vram': 2860.01,
        'function': 'QwenCoder_3b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Granite - 8b': {
        'model': 'Granite - 8b',
        'repo_id': 'ibm-granite/granite-3.1-8b-instruct',
        'cache_dir': 'ibm-granite--granite-3.1-8b-instruct',
        'cps': 137.73,
        'context_length': 8192,
        'vram': 5291.93,
        'function': 'Granite_8b',
        'precision': 'bfloat16', # have float32 version
        'gated': False,
    },
    'Exaone - 7.8b': {
        'model': 'Exaone - 7.8b',
        'repo_id': 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct',
        'cache_dir': 'LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct',
        'cps': 187.24,
        'context_length': 32768,
        'vram': 6281.91,
        'function': 'Exaone_7_8b',
        # 'precision': 'float32',
        'gated': False,
    },
    'Qwen Coder - 7b': {
        'model': 'Qwen Coder - 7b',
        'repo_id': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-Coder-7B-Instruct',
        'cps': 219.55,
        'context_length': 4096,
        'vram': 6760.18,
        'function': 'QwenCoder_7b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen Coder - 14b': {
        'model': 'Qwen Coder - 14b',
        'repo_id': 'Qwen/Qwen2.5-Coder-14B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-Coder-14B-Instruct',
        'cps': 144.76,
        'context_length': 32768,
        'vram': 11029.87,
        'function': 'QwenCoder_14b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen - 14b': {
        'model': 'Qwen - 14b',
        'repo_id': 'Qwen/Qwen2.5-14B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-14B-Instruct',
        'cps': 139.26,
        'context_length': 8192,
        'vram': 11168.49,
        'function': 'Qwen_14b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Mistral Small - 22b': {
        'model': 'Mistral Small - 22b',
        'repo_id': 'mistralai/Mistral-Small-Instruct-2409',
        'cache_dir': 'mistralai--Mistral-Small-Instruct-2409',
        'cps': 98.33,
        'context_length': 32768,
        'vram': 13723.65,
        'function': 'Mistral_Small_22b',
        'precision': 'bfloat16',
        'gated': True,
    },
    'Qwen Coder - 32b': {
        'model': 'Qwen Coder - 32b',
        'repo_id': 'Qwen/Qwen2.5-Coder-32B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-Coder-32B-Instruct',
        'cps': 97.47,
        'context_length': 32768,
        'vram': 21120.81,
        'function': 'QwenCoder_32b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen - 32b': {
        'model': 'Qwen - 32b',
        'repo_id': 'Qwen/Qwen2.5-32B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-32B-Instruct',
        'cps': 101.51,
        'context_length': 8192,
        'vram': 21128.30,
        'function': 'Qwen_32b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Exaone - 32b': {
        'model': 'Exaone - 32b',
        'repo_id': 'LGAI-EXAONE/EXAONE-3.5-32B-Instruct',
        'cache_dir': 'LGAI-EXAONE--EXAONE-3.5-32B-Instruct',
        'cps': 100.54,
        'context_length': 32768,
        'vram': 21982.30,
        'function': 'Exaone_32b',
        # 'precision': 'float32',
        'gated': False,
    },
}

WHISPER_SPEECH_MODELS = {
    "s2a": {
        "s2a-q4-tiny": ("s2a-q4-tiny-en+pl.model", 74),
        "s2a-q4-base": ("s2a-q4-base-en+pl.model", 203),
        "s2a-q4-hq-fast": ("s2a-q4-hq-fast-en+pl.model", 380),
        # "s2a-v1.1-small": ("s2a-v1.1-small-en+pl-noyt.model", 437),
        # "s2a-q4-small": ("s2a-q4-small-en+pl.model", 874),
    },
    "t2s": {
        "t2s-tiny": ("t2s-tiny-en+pl.model", 74),
        "t2s-base": ("t2s-base-en+pl.model", 193),
        "t2s-hq-fast": ("t2s-hq-fast-en+pl.model", 743),
        # "t2s-fast-small": ("t2s-fast-small-en+pl.model", 743),
        # "t2s-small": ("t2s-small-en+pl.model", 856),
        # "t2s-v1.1-small": ("t2s-v1.1-small-en+pl.model", 429),
        # "t2s-fast-medium": ("t2s-fast-medium-en+pl+yt.model", 1310)
    }
}

VECTOR_MODELS = {
    'Alibaba-NLP': [
        {
            'name': 'Alibaba-gte-base',
            'dimensions': 768,
            'max_sequence': 8192,
            'size_mb': 547,
            'repo_id': 'Alibaba-NLP/gte-base-en-v1.5',
            'cache_dir': 'Alibaba-NLP--gte-base-en-v1.5',
            'type': 'vector',
            'parameters': '137m',
            'precision': 'float32'
        },
        {
            'name': 'Alibaba-gte-large',
            'dimensions': 1024,
            'max_sequence': 8192,
            'size_mb': 1740,
            'repo_id': 'Alibaba-NLP/gte-large-en-v1.5',
            'cache_dir': 'Alibaba-NLP--gte-large-en-v1.5',
            'type': 'vector',
            'parameters': '434m',
            'precision': 'float32'
        },
    ],
    'BAAI': [
        {
            'name': 'bge-small-en-v1.5',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'BAAI/bge-small-en-v1.5',
            'cache_dir': 'BAAI--bge-small-en-v1.5',
            'type': 'vector',
            'parameters': '33.4m',
            'precision': 'float32'
        },
        {
            'name': 'bge-base-en-v1.5',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438,
            'repo_id': 'BAAI/bge-base-en-v1.5',
            'cache_dir': 'BAAI--bge-base-en-v1.5',
            'type': 'vector',
            'parameters': '109m',
            'precision': 'float32'
        },
        {
            'name': 'bge-large-en-v1.5',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'BAAI/bge-large-en-v1.5',
            'cache_dir': 'BAAI--bge-large-en-v1.5',
            'type': 'vector',
            'parameters': '335m',
            'precision': 'float32'
        },
    ],
    'hkunlp': [
        {
            'name': 'instructor-base',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 439,
            'repo_id': 'hkunlp/instructor-base',
            'cache_dir': 'hkunlp--instructor-base',
            'type': 'vector',
            'parameters': '110m',
            'precision': 'float32'
        },
        {
            'name': 'instructor-large',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'hkunlp/instructor-large',
            'cache_dir': 'hkunlp--instructor-large',
            'type': 'vector',
            'parameters': '335m',
            'precision': 'float32'
        },
        {
            'name': 'instructor-xl',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 4960,
            'repo_id': 'hkunlp/instructor-xl',
            'cache_dir': 'hkunlp--instructor-xl',
            'type': 'vector',
            'parameters': '1.5b',
            'precision': 'float32'
        },
    ],
    'IBM': [
        {
            'name': 'Granite-30m-English',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 61,
            'repo_id': 'ibm-granite/granite-embedding-30m-english',
            'cache_dir': 'ibm-granite--granite-embedding-30m-english',
            'type': 'vector',
            'parameters': '30.3m',
            'precision': 'bfloat16'
        },
        {
            'name': 'Granite-125m-English',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 249,
            'repo_id': 'ibm-granite/granite-embedding-125m-english',
            'cache_dir': 'ibm-granite--granite-embedding-125m-english',
            'type': 'vector',
            'parameters': '125m',
            'precision': 'bfloat16'
        },
    ],
    'intfloat': [
        {
            'name': 'e5-small-v2',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'intfloat/e5-small-v2',
            'cache_dir': 'intfloat--e5-small-v2',
            'type': 'vector',
            'parameters': '33.4m',
            'precision': 'float32'
        },
        {
            'name': 'e5-base-v2',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438,
            'repo_id': 'intfloat/e5-base-v2',
            'cache_dir': 'intfloat--e5-base-v2',
            'type': 'vector',
            'parameters': '109m',
            'precision': 'float32'
        },
        {
            'name': 'e5-large-v2',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'intfloat/e5-large-v2',
            'cache_dir': 'intfloat--e5-large-v2',
            'type': 'vector',
            'parameters': '335m',
            'precision': 'float32'
        },
    ],
    'sentence-transformers': [
        {
            'name': 'sentence-t5-base',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 219,
            'repo_id': 'sentence-transformers/sentence-t5-base',
            'cache_dir': 'sentence-transformers--sentence-t5-base',
            'type': 'vector',
            'parameters': '110m',
            'precision': 'float16'
        },
        {
            'name': 'sentence-t5-large',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 670,
            'repo_id': 'sentence-transformers/sentence-t5-large',
            'cache_dir': 'sentence-transformers--sentence-t5-large',
            'type': 'vector',
            'parameters': '335m',
            'precision': 'float16'
        },
        {
            'name': 'sentence-t5-xl',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 2480,
            'repo_id': 'sentence-transformers/sentence-t5-xl',
            'cache_dir': 'sentence-transformers--sentence-t5-xl',
            'type': 'vector',
            'parameters': '1.24b',
            'precision': 'float16'
        },
        {
            'name': 'sentence-t5-xxl',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 9230,
            'repo_id': 'sentence-transformers/sentence-t5-xxl',
            'cache_dir': 'sentence-transformers--sentence-t5-xxl',
            'type': 'vector',
            'parameters': '4.86b',
            'precision': 'float16'
        },
    ],
    'Snowflake': [
        {
            'name': 'arctic-embed-m-v2.0',
            'dimensions': 768,
            'max_sequence': 8192,
            'size_mb': 1220,
            'repo_id': 'Snowflake/snowflake-arctic-embed-m-v2.0',
            'cache_dir': 'Snowflake--snowflake-arctic-embed-m-v2.0',
            'type': 'vector',
            'parameters': '305m',
            'precision': 'float32'
        },
        {
            'name': 'arctic-embed-l-v2.0',
            'dimensions': 1024,
            'max_sequence': 8192,
            'size_mb': 2270,
            'repo_id': 'Snowflake/snowflake-arctic-embed-l-v2.0',
            'cache_dir': 'Snowflake--snowflake-arctic-embed-l-v2.0',
            'type': 'vector',
            'parameters': '568m',
            'precision': 'float32'
        },
    ],
}

VISION_MODELS = {
    'InternVL2.5 - 1b': {
        'precision': 'bfloat16',
        'quant': 'n/a',
        'size': '1b',
        'repo_id': 'OpenGVLab/InternVL2_5-1B',
        'cache_dir': 'OpenGVLab--InternVL2_5-1B',
        'requires_cuda': True,
        'vram': '2.4 GB',
    },
    'Florence-2-base': {
        'precision': 'autoselect',
        'quant': 'n/a',
        'size': '232m',
        'repo_id': 'microsoft/Florence-2-base',
        'cache_dir': 'microsoft--Florence-2-base',
        'requires_cuda': False,
        'vram': '2.6 GB',
    },
    'InternVL2.5 - 4b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '4b',
        'repo_id': 'OpenGVLab/InternVL2_5-4B',
        'cache_dir': 'OpenGVLab--InternVL2_5-4B',
        'requires_cuda': True,
        'vram': '3.2 GB',
    },
    'Moondream2 - 1.9b': {
        'precision': 'float16',
        'quant': 'n/a',
        'size': '2b',
        'repo_id': 'vikhyatk/moondream2',
        'cache_dir': 'vikhyatk--moondream2',
        'requires_cuda': True,
        'vram': '4.6 GB',
    },
    'Florence-2-large': {
        'precision': 'autoselect',
        'quant': 'n/a',
        'size': '770m',
        'repo_id': 'microsoft/Florence-2-large',
        'cache_dir': 'microsoft--Florence-2-large',
        'requires_cuda': False,
        'vram': '5.3 GB',
    },
    'Mississippi - 2b': {
        'precision': 'autoselect',
        'quant': 'n/a',
        'size': '2b',
        'repo_id': 'h2oai/h2ovl-mississippi-2b',
        'cache_dir': 'h2oai--h2ovl-mississippi-2b',
        'requires_cuda': True,
        'vram': '5.3 GB',
    },
    'Ovis1.6-Llama3.2 - 3b': {
        'precision': 'bfloat16',
        'quant': 'n/a',
        'size': '3b',
        'repo_id': 'AIDC-AI/Ovis1.6-Llama3.2-3B',
        'cache_dir': 'AIDC-AI--Ovis1.6-Llama3.2-3B',
        'requires_cuda': True,
        'vram': '9.6 GB',
    },
    'THUDM glm4v - 9b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '9b',
        'repo_id': 'THUDM/glm-4v-9b',
        'cache_dir': 'THUDM--glm-4v-9b',
        'requires_cuda': True,
        'vram': '10.5 GB',
    },
    'Molmo-D-0924 - 8b': {
        'precision': 'autoselect',
        'quant': '4-bit',
        'size': '8b',
        'repo_id': 'ctranslate2-4you/molmo-7B-O-bnb-4bit',
        'cache_dir': 'ctranslate2-4you--molmo-7B-O-bnb-4bit',
        'requires_cuda': True,
        'vram': '10.5 GB',
    },
    'Llava 1.6 Vicuna - 13b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '13b',
        'repo_id': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'cache_dir': 'llava-hf--llava-v1.6-vicuna-13b-hf',
        'requires_cuda': True,
        'vram': '14.1 GB',
    }
}

WHISPER_MODELS = {
    # LARGE-V3
    'Distil Whisper large-v3 - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float32',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper large-v3 - bfloat16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-bfloat16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper large-v3 - float16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Whisper large-v3 - float32': {
        'name': 'Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float32',
        'cps': 85,
        'optimal_batch_size': 2,
        'vram': '5.5 GB'
    },
    'Whisper large-v3 - bfloat16': {
        'name': 'Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-bfloat16',
        'cps': 95,
        'optimal_batch_size': 3,
        'vram': '3.8 GB'
    },
    'Whisper large-v3 - float16': {
        'name': 'Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float16',
        'cps': 100,
        'optimal_batch_size': 3,
        'vram': '3.3 GB'
    },
    # MEDIUM.EN
    'Distil Whisper medium.en - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float32',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper medium.en - bfloat16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-bfloat16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper medium.en - float16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Whisper medium.en - float32': {
        'name': 'Whisper medium.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float32',
        'cps': 130,
        'optimal_batch_size': 6,
        'vram': '2.5 GB'
    },
    'Whisper medium.en - bfloat16': {
        'name': 'Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-bfloat16',
        'cps': 140,
        'optimal_batch_size': 7,
        'vram': '2.0 GB'
    },
    'Whisper medium.en - float16': {
        'name': 'Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float16',
        'cps': 145,
        'optimal_batch_size': 7,
        'vram': '1.8 GB'
    },
    # SMALL.EN
    'Distil Whisper small.en - float32': {
        'name': 'Distil Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float32',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper small.en - bfloat16': {
        'name': 'Distil Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-bfloat16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper small.en - float16': {
        'name': 'Distil Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Whisper small.en - float32': {
        'name': 'Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float32',
        'cps': 180,
        'optimal_batch_size': 14,
        'vram': '1.5 GB'
    },
    'Whisper small.en - bfloat16': {
        'name': 'Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-bfloat16',
        'cps': 190,
        'optimal_batch_size': 15,
        'vram': '1.2 GB'
    },
    'Whisper small.en - float16': {
        'name': 'Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float16',
        'cps': 195,
        'optimal_batch_size': 15,
        'vram': '1.1 GB'
    },
    # BASE.EN
    'Whisper base.en - float32': {
        'name': 'Whisper base.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float32',
        'cps': 230,
        'optimal_batch_size': 22,
        'vram': '1.0 GB'
    },
    'Whisper base.en - bfloat16': {
        'name': 'Whisper base.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-bfloat16',
        'cps': 240,
        'optimal_batch_size': 23,
        'vram': '0.85 GB'
    },
    'Whisper base.en - float16': {
        'name': 'Whisper base.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float16',
        'cps': 245,
        'optimal_batch_size': 23,
        'vram': '0.8 GB'
    },
    # TINY.EN
    'Whisper tiny.en - float32': {
        'name': 'Whisper tiny.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float32',
        'cps': 280,
        'optimal_batch_size': 30,
        'vram': '0.7 GB'
    },
    'Whisper tiny.en - bfloat16': {
        'name': 'Whisper tiny.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-bfloat16',
        'cps': 290,
        'optimal_batch_size': 31,
        'vram': '0.6 GB'
    },
    'Whisper tiny.en - float16': {
        'name': 'Whisper tiny.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float16',
        'cps': 295,
        'optimal_batch_size': 31,
        'vram': '0.55 GB'
    },
}

JEEVES_MODELS = {
    "EXAONE - 2.4b Q8_0": {
        "filename": "EXAONE-3.5-2.4B-Instruct-Q8_0.gguf",
        "repo_id": "bartowski/EXAONE-3.5-2.4B-Instruct-GGUF",
        "allow_patterns": ["EXAONE-3.5-2.4B-Instruct-Q8_0.gguf"],
        "prompt_template": """[|system|]{jeeves_system_message}[|endofturn|]
[|user|]{user_message}
[|endofturn|]
[|assistant|]"""
    },
    "Llama - 3b Q8_0": {
        "filename": "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "repo_id": "lmstudio-community/Llama-3.2-3B-Instruct-GGUF",
        "allow_patterns": ["Llama-3.2-3B-Instruct-Q8_0.gguf"],
        "prompt_template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
{jeeves_system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    },
    "Qwen - 3b Q8_0": {
        "filename": "Qwen2.5-3B-Instruct-Q8_0.gguf",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "allow_patterns": ["Qwen2.5-3B-Instruct-Q8_0.gguf"],
        "prompt_template": """<|im_start|>system
{jeeves_system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    },
    "Zephyr - 3b Q8_0": {
        "filename": "stablelm-zephyr-3b-q8_0.gguf",
        "repo_id": "ysn-rfd/stablelm-zephyr-3b-Q8_0-GGUF",
        "allow_patterns": ["stablelm-zephyr-3b-q8_0.gguf"],
        "prompt_template": """<|system|>
{jeeves_system_message}<|endoftext|>
<|user|>
{user_message}<|endoftext|>
<|assistant|>
"""
    },
    "EXAONE - 7.8b Q4_K_M": {
        "filename": "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf",
        "repo_id": "bartowski/EXAONE-3.5-7.8B-Instruct-GGUF",
        "allow_patterns": ["EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"],
        "prompt_template": """[|system|]{jeeves_system_message}[|endofturn|]
[|user|]{user_message}
[|endofturn|]
[|assistant|]"""
    },
}

kobold_config = {
  "benchmark": None,
  "blasbatchsize": 512,
  "blasthreads": None,
  "chatcompletionsadapter": None,
  "config": None,
  "contextsize": 2048,
  "debugmode": 0,
  "flashattention": False,
  "forceversion": 0,
  "foreground": False,
  "gpulayers": -1,
  "highpriority": True,
  "hordeconfig": None,
  "hordegenlen": 0,
  "hordekey": "",
  "hordemaxctx": 0,
  "hordemodelname": "",
  "hordeworkername": "",
  "host": "",
  "ignoremissing": False,
  "istemplate": True,
  "launch": False,
  "lora": None,
  "mmproj": None,
  "model": "",
  "model_param": "",
  "multiuser": 1,
  "noblas": False,
  "nocertify": False,
  "noavx2": False,
  "nommap": False,
  "nomodel": False,
  "noshift": True,
  "onready": "",
  "password": None,
  "port": 5001,
  "port_param": 5001,
  "preloadstory": None,
  "prompt": "",
  "promptlimit": 100,
  "quantkv": 0,
  "quiet": True,
  "remotetunnel": False,
  "ropeconfig": [0.0, 10000.0],
  "sdclamped": 0,
  "sdconfig": None,
  "sdlora": "",
  "sdloramult": 1.0,
  "sdmodel": "",
  "sdquant": False,
  "sdthreads": 0,
  "sdvae": "",
  "sdvaeauto": False,
  "showgui": False,
  "skiplauncher": True,
  "smartcontext": True,
  "ssl": None,
  "tensor_split": None,
  "threads": -1,
  "unpack": "",
  "useblascpu": None,
  "useclblast": None,
  "usecpu": False,
  "usecublas": None,
  "usemlock": True,
  "usevulkan": None,
  "whispermodel": ""
}

DOCUMENT_LOADERS = {
    ".pdf": "PyMuPDFLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
    ".enex": "EverNoteLoader",
    ".epub": "UnstructuredEPubLoader",
    ".eml": "UnstructuredEmailLoader",
    ".msg": "UnstructuredEmailLoader",
    ".csv": "CSVLoader",
    ".xls": "UnstructuredExcelLoader",
    ".xlsx": "UnstructuredExcelLoader",
    ".xlsm": "UnstructuredExcelLoader",
    ".rtf": "UnstructuredRTFLoader",
    ".odt": "UnstructuredODTLoader",
    ".md": "UnstructuredMarkdownLoader",
    ".html": "BSHTMLLoader",
}

PROMPT_FORMATS = {
    "ChatML": {
        "prefix": "",
        "suffix": ""
    },
    "Llama2/Mistral": {
        "prefix": "[INST]",
        "suffix": "[/INST]"
    },
    "Neural Chat/SOLAR": {
        "prefix": "### User:",
        "suffix": "### Assistant:"
    },
    "Orca2": {
        "prefix": "user",
        "suffix": "assistant"
    },
    "StableLM-Zephyr": {
        "prefix": "",
        "suffix": " "
    }
}

TOOLTIPS = {
    "AUDIO_FILE_SELECT": "Select an audio file. Supports various audio formats.",
    "CHOOSE_FILES": "Select documents to add to the database. Remember to transcribe audio files in the Tools tab first.",
    "CHUNK_OVERLAP": "Characters shared between chunks. Set to 25-50% of chunk size.",
    "CHUNK_SIZE": (
        "<html><body>"
        "Upper limit (in characters, not tokens) that a chunk can be after being split.  Make sure that it falls within"
        "the Max Sequence of the embedding model being used, which is measured in tokens (not characters), remembering that"
        "approximately 3-4 characters = 1 token."
        "</body></html>"
    ),
    "CHUNKS_ONLY": "Solely query the vector database and get relevant chunks. Very useful to test the chunk size/overlap settings.",
    "CONTEXTS": "Maximum number of chunks (aka contexts) to return.",
    "COPY_RESPONSE": "Copy the chunks (if chunks only is checked) or model's response to the clipboard.",
    "CREATE_DEVICE_DB": "Choose 'cpu' or 'cuda'. Use 'cuda' if available.",
    "CREATE_DEVICE_QUERY": "Choose 'cpu' or 'cuda'. 'cpu' recommended to conserve VRAM.",
    "CREATE_VECTOR_DB": "Creates a new vector database.",
    "DATABASE_NAME_INPUT": "Enter a unique database name. Use only lowercase letters, numbers, underscores, and hyphens.",
    "DATABASE_SELECT": "Vector database that will be queried.",
    "DISABLE_PROMPT_FORMATTING": "Disables built-in prompt formatting, using LM Studio's settings instead.",
    "DOWNLOAD_MODEL": "Download the selected vector model.",
    "EJECT_LOCAL_MODEL": "Unload the current local model from memory.",
    "FILE_TYPE_FILTER": "Only allows chunks that originate from certain file types.",
    "HALF_PRECISION": "Uses bfloat16/float16 for 2x speedup. Requires a GPU.",
    "LOCAL_MODEL_SELECT": "Select a local model for generating responses.",
    "MAX_TOKENS": "Maximum tokens for LLM response. -1 for unlimited.",
    "MODEL_BACKEND_SELECT": "Choose the backend for the large language model response.",
    "PORT": "Must match the port used in LM Studio.",
    "PREFIX_SUFFIX": "Prompt format for LLM. Use preset or custom for different models.",
    "QUESTION_INPUT": "Type your question here or use the voice recorder.",
    "RESTORE_CONFIG": "Restores original config.yaml. May require manual database cleanup.",
    "RESTORE_DATABASE": "Restores backed-up databases. Use with caution.",
    "SEARCH_TERM_FILTER": "Removes chunks without exact term. Case-insensitive.",
    "SELECT_VECTOR_MODEL": "Choose the vector model for text embedding.",
    "SIMILARITY": "Relevance threshold for chunks. 0-1, higher returns more. Don't use 1.",
    "SPEAK_RESPONSE": "Speak the response from the large language model using text-to-speech.",
    "TEMPERATURE": "Controls LLM creativity. 0-1, higher is more creative.",
    "TRANSCRIBE_BUTTON": "Start transcription.",
    "TTS_MODEL": "Choose TTS model. Bark offers customization, Google requires internet.",
    "VECTOR_MODEL_DIMENSIONS": "Higher dimensions captures more nuance but requires more processing time.",
    "VECTOR_MODEL_DOWNLOADED": "Whether the model has been downloaded.",
    "VECTOR_MODEL_LINK": "Huggingface link.",
    "VECTOR_MODEL_MAX_SEQUENCE": "Number of tokens the model can process at once. Different from the Chunk Size setting, which is in characters.",
    "VECTOR_MODEL_NAME": "The name of the vector model.",
    "VECTOR_MODEL_PARAMETERS": "The number of internal weights and biases that the model learns and adjusts during training.",
    "VECTOR_MODEL_PRECISION": (
        "<html>"
        "<body>"
        "<p style='font-size: 14px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-bottom: 10px;'>"
        "<b>The precision ultimately used depends on your setup:</b></p>"
        "<table style='border-collapse: collapse; width: 100%; font-size: 12px; color: #34495e;'>"
        "<thead>"
        "<tr style='background-color: #ecf0f1; text-align: left;'>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Compute Device</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Embedding Model Precision</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>'Half' Checked?</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Precision Ultimately Used</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CPU</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Any</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Either</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float32</code></td>"
        "</tr>"
        "<tr style='background-color: #ecf0f1;'>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float16</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Either</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float16</code></td>"
        "</tr>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>bfloat16</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Either</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>"
        "<code>bfloat16</code> (if CUDA capability &ge; 8.6) or <code>float16</code></td>"
        "</tr>"
        "<tr style='background-color: #ecf0f1;'>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float32</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>No</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float32</code></td>"
        "</tr>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float32</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Yes</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>"
        "<code>bfloat16</code> (if CUDA capability &ge; 8.6) or <code>float16</code>"
        "</td>"
        "</tr>"
        "</tbody>"
        "</table>"
        "</body>"
        "</html>"
    ),
    "VECTOR_MODEL_SELECT": "Choose a vector model to download.",
    "VECTOR_MODEL_SIZE": "Size on disk.",
    "VISION_MODEL": "Select vision model for image processing. Test before bulk processing.",
    "VOICE_RECORDER": "Click to start recording, speak your question, then click again to stop recording.",
    "WHISPER_BATCH_SIZE": "Batch size for transcription. See the User Guid for optimal values.",
    "WHISPER_MODEL_SELECT": "Distil models use ~ 70% VRAM of their non-Distil equivalents with little quality loss."
}

scrape_documentation = {
    "Accelerate 0.34.2": {
        "URL": "https://huggingface.co/docs/accelerate/v0.34.2/en/",
        "folder": "accelerate_0342",
        "scraper_class": "HuggingfaceScraper"
    },
    "Accelerate 1.1.0": {
        "URL": "https://huggingface.co/docs/accelerate/v1.1.0/en",
        "folder": "accelerate_110",
        "scraper_class": "HuggingfaceScraper"
    },
    "aiohttp 3.9.5": {
        "URL": "https://docs.aiohttp.org/en/v3.9.5/",
        "folder": "aiohttp_395"
    },
    "aiohttp": {
        "URL": "https://docs.aiohttp.org/en/stable/",
        "folder": "aiohttp"
    },
    "Argcomplete": {
        "URL": "https://kislyuk.github.io/argcomplete/",
        "folder": "argcomplete"
    },
    "AutoAWQ": {
        "URL": "https://casper-hansen.github.io/AutoAWQ/",
        "folder": "autoawq"
    },
    "Beautiful Soup 4": {
        "URL": "https://beautiful-soup-4.readthedocs.io/en/latest/",
        "folder": "beautiful_soup_4"
    },
    "bitsandbytes 0.45.0": {
        "URL": "https://huggingface.co/docs/bitsandbytes/v0.45.0/en/",
        "folder": "bitsandbytes_0450",
        "scraper_class": "HuggingfaceScraper"
    },
    "Black": {
        "URL": "https://black.readthedocs.io/en/stable/",
        "folder": "Black"
    },
    "chardet": {
        "URL": "https://chardet.readthedocs.io/en/stable/",
        "folder": "chardet"
    },
    "charset-normalizer 3.3.2": {
        "URL": "https://charset-normalizer.readthedocs.io/en/3.3.2/",
        "folder": "charset_normalizer_332"
    },
    "charset-normalizer": {
        "URL": "https://charset-normalizer.readthedocs.io/en/stable/",
        "folder": "charset_normalizer"
    },
    "Click": {
        "URL": "https://click.palletsprojects.com/en/stable/",
        "folder": "click"
    },
    "coloredlogs": {
        "URL": "https://coloredlogs.readthedocs.io/en/latest/",
        "folder": "coloredlogs"
    },
    "CTranslate2": {
        "URL": "https://opennmt.net/CTranslate2/",
        "folder": "ctranslate2"
    },
    "cuDF": {
        "URL": "https://docs.rapids.ai/api/cudf/stable/",
        "folder": "cuDF"
    },
    "CuPy": {
        "URL": "https://docs.cupy.dev/en/stable/",
        "folder": "cupy"
    },
    "CustomTkinter": {
        "URL": "https://customtkinter.tomschimansky.com/documentation/",
        "folder": "customtkinter"
    },
    "Dask": {
        "URL": "https://docs.dask.org/en/stable/",
        "folder": "dask"
    },
    "dill": {
        "URL": "https://dill.readthedocs.io/en/latest/",
        "folder": "dill"
    },
    "dnspython 2.7": {
        "URL": "https://dnspython.readthedocs.io/en/2.7/",
        "folder": "dill"
    },
    "gTTS": {
        "URL": "https://gtts.readthedocs.io/en/latest/",
        "folder": "gtts"
    },
    "Huggingface Hub 0.26.5": {
        "URL": "https://huggingface.co/docs/huggingface_hub/v0.26.5/en/",
        "folder": "huggingface_hub_0265",
        "scraper_class": "HuggingfaceScraper"
    },
    "isort": {
        "URL": "https://pycqa.github.io/isort/",
        "folder": "isort"
    },
    "Jinja": {
        "URL": "https://jinja.palletsprojects.com/en/stable/",
        "folder": "jinja"
    },
    "jiwer": {
        "URL": "https://jitsi.github.io/jiwer/",
        "folder": "jiwer"
    },
    "jsonschema 4.23.0": {
        "URL": "https://python-jsonschema.readthedocs.io/en/v4.23.0/",
        "folder": "jsonschema_423"
    },
    "jsonschema-specifications": {
        "URL": "https://jsonschema-specifications.readthedocs.io/en/stable/",
        "folder": "jsonschema_specifications"
    },
    "Langchain (0.2)": {
        "URL": "https://python.langchain.com/v0.2/api_reference/",
        "folder": "langchain_02",
        "scraper_class": "LangchainScraper"
    },
    "Langchain (0.3)": {
        "URL": "https://python.langchain.com/api_reference/",
        "folder": "langchain_03",
        "scraper_class": "LangchainScraper"
    },
    "Librosa": {
        "URL": "https://librosa.org/doc/latest/",
        "folder": "librosa"
    },
    "llama-cpp-python": {
        "URL": "https://llama-cpp-python.readthedocs.io/en/stable/",
        "folder": "llama_cpp_python"
    },
    "LM Studio": {
        "URL": "https://lmstudio.ai/docs/",
        "folder": "lm_studio"
    },
    "Loguru": {
        "URL": "https://loguru.readthedocs.io/en/stable/",
        "folder": "loguru"
    },
    "lxml 5.3.0": {
        "URL": "https://lxml.de/5.3/",
        "folder": "lxml_530"
    },
    "lxml-html-clean": {
        "URL": "https://lxml-html-clean.readthedocs.io/en/stable/",
        "folder": "lxml_html_clean"
    },
    "marshmallow": {
        "URL": "https://marshmallow.readthedocs.io/en/stable/",
        "folder": "marshmallow"
    },
    # "Matplotlib": {
        # "URL": "https://matplotlib.org/stable/", # won't scrape
        # "folder": "matplotlib"
    # },
    "mpmath": {
        "URL": "https://mpmath.org/doc/current/",
        "folder": "mpmath"
    },
    "msg-parser": {
        "URL": "https://msg-parser.readthedocs.io/en/latest/",
        "folder": "msg_parser"
    },
    "multiprocess": {
        "URL": "https://multiprocess.readthedocs.io/en/stable/",
        "folder": "multiprocess"
    },
    "natsort 8.4.0": {
        "URL": "https://natsort.readthedocs.io/en/8.4.0/",
        "folder": "natsort_840"
    },
    "natsort": {
        "URL": "https://natsort.readthedocs.io/en/stable/",
        "folder": "natsort"
    },
    "NetworkX": {
        "URL": "https://networkx.org/documentation/stable/",
        "folder": "networkx"
    },
    "NLTK": {
        "URL": "https://www.nltk.org/",
        "folder": "nltk"
    },
    "Numba 0.60.0": {
        "URL": "https://numba.readthedocs.io/en/0.60.0/",
        "folder": "numba_0600"
    },
    "Numexpr": {
        "URL": "https://numexpr.readthedocs.io/en/latest/",
        "folder": "numexpr"
    },
    "NumPy 1.26": {
        "URL": "https://numpy.org/doc/1.26/",
        "folder": "numpy_126"
    },
    "NumPy 2.1": {
        "URL": "https://numpy.org/doc/2.1/",
        "folder": "numpy_21"
    },
    "OmegaConf 2.2": {
        "URL": "https://omegaconf.readthedocs.io/en/2.2_branch/",
        "folder": "omegaconf_22"
    },
    "ONNX": {
        "URL": "https://onnx.ai/onnx/",
        "folder": "onnx"
    },
    "ONNX Runtime": {
        "URL": "https://onnxruntime.ai/docs/api/python/",
        "folder": "onnx_runtime"
    },
    "openpyxl": {
        "URL": "https://openpyxl.readthedocs.io/en/stable/",
        "folder": "openpyxl"
    },
    "Optimum 1.23.3": {
        "URL": "https://huggingface.co/docs/optimum/v1.23.3/en/",
        "folder": "optimum_1233",
        "scraper_class": "HuggingfaceScraper"
    },
    "packaging": {
        "URL": "https://packaging.pypa.io/en/stable/",
        "folder": "packaging"
    },
    "pandas": {
        "URL": "https://pandas.pydata.org/docs/",
        "folder": "pandas"
    },
    "Pandoc": {
        "URL": "https://pandoc.org",
        "folder": "pandoc"
    },
    "PathSpec 0.12.1": {
        "URL": "https://python-path-specification.readthedocs.io/en/v0.12.1/",
        "folder": "pathspec_0121"
    },
    "platformdirs": {
        "URL": "https://platformdirs.readthedocs.io/en/stable/",
        "folder": "platformdirs"
    },
    "Playwright": {
        "URL": "https://playwright.dev/python/",
        "folder": "playwright"
    },
    "Pillow": {
        "URL": "https://pillow.readthedocs.io/en/stable/",
        "folder": "pillow"
    },
    "Protocol Buffers": {
        "URL": "https://protobuf.dev/",
        "folder": "protocol_buffers"
    },
    "PyAV": {
        "URL": "https://pyav.org/docs/stable/",
        "folder": "pyav"
    },
    "Pydantic": {
        "URL": "https://docs.pydantic.dev/latest/",
        "folder": "pydantic"
    },
    "Pygments": {
        "URL": "https://pygments.org/docs/",
        "folder": "pygments"
    },
    "PyInstaller 6.10.0": {
        "URL": "https://pyinstaller.org/en/v6.10.0/",
        "folder": "pyinstaller_6100"
    },
    "PyMuPDF": {
        "URL": "https://pymupdf.readthedocs.io/en/latest/",
        "folder": "pymupdf"
    },
    "PyPDF 5.1.0": {
        "URL": "https://pypdf.readthedocs.io/en/5.1.0/",
        "folder": "pypdf_510",
        "scraper_class": "ReadthedocsScraper"
    },
    "PyTorch Lightning": {
        "URL": "https://lightning.ai/docs/pytorch/stable/",
        "folder": "pytorch_lightning"
    },
    # "python-docx": {
        # "URL": "https://python-docx.readthedocs.io/en/stable/", # won't scrape
        # "folder": "python_docx"
    # },
    "PyYAML": {
        "URL": "https://pyyaml.org/wiki/PyYAMLDocumentation",
        "folder": "pyyaml"
    },
    "Pywin32": {
        "URL": "https://mhammond.github.io/pywin32/",
        "folder": "pywin32"
    },
    "Pyside 6": {
        "URL": "https://doc.qt.io/",
        "folder": "pyside6"
    },
    "RapidFuzz": {
        "URL": "https://rapidfuzz.github.io/RapidFuzz/",
        "folder": "rapidfuzz"
    },
    "Referencing": {
        "URL": "https://referencing.readthedocs.io/en/stable/",
        "folder": "referencing"
    },
    "Requests": {
        "URL": "https://requests.readthedocs.io/en/stable/",
        "folder": "requests"
    },
    "Rich": {
        "URL": "https://rich.readthedocs.io/en/stable/",
        "folder": "rich",
        "scraper_class": "ReadthedocsScraper"
    },
    "rpds-py": {
        "URL": "https://rpds.readthedocs.io/en/stable/",
        "folder": "rpds_py"
    },
    "Rust": {
        "URL": "https://doc.rust-lang.org/stable/",
        "folder": "rust"
    },
    "Rust Docs": {
        "URL": "https://docs.rs",
        "folder": "docs_rs"
    },
    "Rust Std Docs": {
        "URL": "https://doc.rust-lang.org/std/",
        "folder": "rust_std"
    },
    "Rust UV": {
        "URL": "https://docs.astral.sh/uv/",
        "folder": "uv"
    },
    "Safetensors 0.3.2": {
        "URL": "https://huggingface.co/docs/safetensors/v0.3.2/en/",
        "folder": "safetensors_032",
        "scraper_class": "HuggingfaceScraper"
    },
    "scikit-learn": {
        "URL": "https://scikit-learn.org/stable/",
        "folder": "scikit_learn"
    },
    "SciPy 1.14.1": {
        "URL": "https://docs.scipy.org/doc/scipy-1.14.1/",
        "folder": "scipy_1141"
    },
    "Sentence-Transformers": {
        "URL": "https://www.sbert.net/docs",
        "folder": "sentence_transformers"
    },
    "Six": {
        "URL": "https://six.readthedocs.io/",
        "folder": "six",
        "scraper_class": "ReadthedocsScraper"
    },
    "SoundFile 0.11.0": {
        "URL": "https://python-soundfile.readthedocs.io/en/0.11.0/",
        "folder": "soundfile_0110",
        "scraper_class": "ReadthedocsScraper"
    },
    "sounddevice 0.4.6": {
        "URL": "https://python-sounddevice.readthedocs.io/en/0.4.6/",
        "folder": "sounddevice_046"
    },
    "Soupsieve": {
        "URL": "https://facelessuser.github.io/soupsieve/",
        "folder": "soupsieve"
    },
    "Soxr": {
        "URL": "https://python-soxr.readthedocs.io/en/stable/",
        "folder": "soxr"
    },
    "SpeechBrain 0.5.15": {
        "URL": "https://speechbrain.readthedocs.io/en/v0.5.15/",
        "folder": "speechbrain_0515",
        "scraper_class": "ReadthedocsScraper"
    },
    "SQLAlchemy 20": {
        "URL": "https://docs.sqlalchemy.org/en/20/",
        "folder": "sqlalchemy_20"
    },
    "SymPy": {
        "URL": "https://docs.sympy.org/latest/",
        "folder": "sympy"
    },
    "TensorRT-LLM": {
        "URL": "https://nvidia.github.io/TensorRT-LLM/",
        "folder": "tensorrt_llm",
        "scraper_class": "ReadthedocsScraper"
    },
    "Timm 0.9.16": {
        "URL": "https://huggingface.co/docs/timm/v0.9.16/en/",
        "folder": "timm_0916",
        "scraper_class": "HuggingfaceScraper"
    },
    "Timm 1.0.11": {
        "URL": "https://huggingface.co/docs/timm/v1.0.11/en/",
        "folder": "timm_1011",
        "scraper_class": "HuggingfaceScraper"
    },
    "torch 2.5": {
        "URL": "https://pytorch.org/docs/2.5/",
        "folder": "torch_25"
    },
    "Torchaudio 2.5": {
        "URL": "https://pytorch.org/audio/2.5.0/",
        "folder": "torchaudio_25"
    },
    "Torchmetrics": {
        "URL": "https://lightning.ai/docs/torchmetrics/stable/",
        "folder": "torchmetrics"
    },
    "Torchvision 0.20": {
        "URL": "https://pytorch.org/vision/0.20/",
        "folder": "torchvision_020"
    },
    "tqdm": {
        "URL": "https://tqdm.github.io",
        "folder": "tqdm"
    },
    "Transformers 4.47.1": {
        "URL": "https://huggingface.co/docs/transformers/v4.47.1/en",
        "folder": "transformers_4471",
        "scraper_class": "HuggingfaceScraper"
    },
    "Transformers.js 3.0.0": {
        "URL": "https://huggingface.co/docs/transformers.js/v3.0.0/en/",
        "folder": "transformers_js_300",
        "scraper_class": "HuggingfaceScraper"
    },
    "urllib3": {
        "URL": "https://urllib3.readthedocs.io/en/stable/",
        "folder": "urllib3"
    },
    "Watchdog": {
        "URL": "https://pythonhosted.org/watchdog/",
        "folder": "watchdog",
    },
    "webdataset": {
        "URL": "https://webdataset.github.io/webdataset/",
        "folder": "webdataset",
        "scraper_class": "ReadthedocsScraper"
    },
    "Wrapt": {
        "URL": "https://wrapt.readthedocs.io/en/master/",
        "folder": "wrapt",
        "scraper_class": "ReadthedocsScraper"
    },
    "xlrd": {
        "URL": "https://xlrd.readthedocs.io/en/stable/",
        "folder": "xlrd",
        "scraper_class": "ReadthedocsScraper"
    },
    "xFormers": {
        "URL": "https://facebookresearch.github.io/xformers/",
        "folder": "xformers"
    },
    "yarl": {
        "URL": "https://yarl.aio-libs.org/en/stable/",
        "folder": "yarl"
    },
}

class CustomButtonStyles:
    # Base colors
    LIGHT_GREY = "#C8C8C8"
    DISABLED_TEXT = "#969696"
    
    # Color definitions with their hover/pressed/disabled variations
    COLORS = {
        "RED": {
            "base": "#320A0A",
            "hover": "#4B0F0F",
            "pressed": "#290909",
            "disabled": "#7D1919"
        },
        "BLUE": {
            "base": "#0A0A32",
            "hover": "#0F0F4B",
            "pressed": "#09092B",
            "disabled": "#19197D"
        },
        "GREEN": {
            "base": "#0A320A",
            "hover": "#0F4B0F",
            "pressed": "#092909",
            "disabled": "#197D19"
        },
        "YELLOW": {
            "base": "#32320A",
            "hover": "#4B4B0F",
            "pressed": "#292909",
            "disabled": "#7D7D19"
        },
        "PURPLE": {
            "base": "#320A32",
            "hover": "#4B0F4B",
            "pressed": "#290929",
            "disabled": "#7D197D"
        },
        "ORANGE": {
            "base": "#321E0A",
            "hover": "#4B2D0F",
            "pressed": "#291909",
            "disabled": "#7D5A19"
        },
        "TEAL": {
            "base": "#0A3232",
            "hover": "#0F4B4B",
            "pressed": "#092929",
            "disabled": "#197D7D"
        },
        "BROWN": {
            "base": "#2B1E0A",
            "hover": "#412D0F",
            "pressed": "#231909",
            "disabled": "#6B5A19"
        }
    }

    @classmethod
    def _generate_button_style(cls, color_values):
        return f"""
            QPushButton {{
                background-color: {color_values['base']};
                color: {cls.LIGHT_GREY};
                padding: 5px;
                border: none;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {color_values['hover']};
            }}
            QPushButton:pressed {{
                background-color: {color_values['pressed']};
            }}
            QPushButton:disabled {{
                background-color: {color_values['disabled']};
                color: {cls.DISABLED_TEXT};
            }}
        """

for color_name, color_values in CustomButtonStyles.COLORS.items():
    setattr(CustomButtonStyles, f"{color_name}_BUTTON_STYLE", 
            CustomButtonStyles._generate_button_style(color_values))

GPUS_NVIDIA = {
    "GeForce GTX 1630": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 512
    },
    "GeForce GTX 1650 (Apr 2019)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 896
    },
    "GeForce GTX 1650 (Apr 2020)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 896
    },
    "GeForce GTX 1650 (Jun 2020)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 896
    },
    "GeForce GTX 1650 (Laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Ti Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1280
    },
    "GeForce GTX 1660": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1408
    },
    "GeForce GTX 1660 (Laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1408
    },
    "GeForce GTX 1660 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1408
    },
    "GeForce GTX 1660 Ti Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1536
    },
    "GeForce GTX 1660 Ti (Laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1536
    },
    "GeForce GTX 1660 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1536
    },
    "GeForce RTX 2060": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 2060 Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 2060 (Jan 2019)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 2060 (Jan 2020)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 3050 Mobile (4GB)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 2048
    },
    "GeForce RTX 2060 (Dec 2021)": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 2176
    },
    "GeForce RTX 2060 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2176
    },
    "GeForce RTX 2070": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2304
    },
    "GeForce RTX 2070 Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2304
    },
    "GeForce RTX 3050 (GA107-325)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 2304
    },
    "GeForce RTX 3050 (GA106-150)": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2304
    },
    "GeForce RTX 3050 (GA107-150-A1)": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2560
    },
    "GeForce RTX 4050 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 2560
    },
    "GeForce RTX 3050 Ti Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 2560
    },
    "GeForce RTX 3050 Mobile (6GB)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 2560
    },
    "GeForce RTX 2070 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2560
    },
    "GeForce RTX 2070 Super Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2560
    },
    "GeForce RTX 4060": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 3072
    },
    "GeForce RTX 2080 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 3072
    },
    "GeForce RTX 2080 Super Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 3072
    },
    "GeForce RTX 3060": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 3584
    },
    "GeForce RTX 3060 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 3840
    },
    "GeForce RTX 4060 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 4352
    },
    "GeForce RTX 2080 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 11,
        "CUDA Cores": 4352
    },
    "GeForce RTX 4070 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 4608
    },
    "Nvidia TITAN RTX": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 4608
    },
    "GeForce RTX 3060 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 4864
    },
    "GeForce RTX 3070 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 5120
    },
    "GeForce RTX 3070": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 5888
    },
    "GeForce RTX 4070": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 5888
    },
    "GeForce RTX 3070 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 6144
    },
    "GeForce RTX 3070 Ti Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": "8-16",
        "CUDA Cores": 6144
    },
    "GeForce RTX 4070 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7168
    },
    "GeForce RTX 4080 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7424
    },
    "GeForce RTX 3080 Ti Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 7424
    },
    "GeForce RTX 4070 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7680
    },
    "GeForce RTX 4080 (AD104-400)": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7680
    },
    "GeForce RTX 4070 Ti Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 8448
    },
    "GeForce RTX 3080": {
        "Brand": "NVIDIA",
        "Size (GB)": 10,
        "CUDA Cores": 8704
    },
    "GeForce RTX 3080 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 8960
    },
    "GeForce RTX 4080 (AD103-300)": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 9728
    },
    "GeForce RTX 4090 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 9728
    },
    "GeForce RTX 4080 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 10240
    },
    "GeForce RTX 3090": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 10496
    },
    "GeForce RTX 3090 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 10752
    },
    "GeForce RTX 4090 D": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 14592
    },
    "GeForce RTX 4090": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 16384
    }
}

GPUS_AMD = {
    "Radeon RX 7600": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 7600 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 2048
    },
    "Radeon RX 7700 XT": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 3456
    },
    "Radeon RX 7800 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 3840
    },
    "Radeon RX 7900 GRE": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 5120
    },
    "Radeon RX 7900 XT": {
        "Brand": "AMD",
        "Size (GB)": 20,
        "Shaders": 5376
    },
    "Radeon RX 7900 XTX": {
        "Brand": "AMD",
        "Size (GB)": 24,
        "Shaders": 6144
    },
    "Radeon RX 6300": {
        "Brand": "AMD",
        "Size (GB)": 2,
        "Shaders": 768
    },
    "Radeon RX 6400": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1024
    },
    "Radeon RX 6500 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1024
    },
    "Radeon RX 6600": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6600 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6650 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6700": {
        "Brand": "AMD",
        "Size (GB)": 10,
        "Shaders": 2304
    },
    "Radeon RX 6750 GRE 10GB": {
        "Brand": "AMD",
        "Size (GB)": 10,
        "Shaders": 2560
    },
    "Radeon RX 6750 XT": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 2560
    },
    "Radeon RX 6800": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 3840
    },
    "Radeon RX 6800 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 4608
    },
    "Radeon RX 6900 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 5120
    },
    "Radeon RX 6950 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 5120
    },
    "Radeon RX 5300": {
        "Brand": "AMD",
        "Size (GB)": 3,
        "Shaders": 1408
    },
    "Radeon RX 5300 XT": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5500": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5500 XT": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5600": {
        "Brand": "AMD",
        "Size (GB)": 6,
        "Shaders": 2048
    },
    "Radeon RX 5600 XT": {
        "Brand": "AMD",
        "Size (GB)": 6,
        "Shaders": 2304
    },
    "Radeon RX 5700": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2304
    },
    "Radeon RX 5700 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2560
    },
    "Radeon RX 5700 XT 50th Anniversary Edition": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2560
    },
    "Radeon RX Vega 56": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 3584
    },
    "Radeon RX Vega 64": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 4096
    },
    "Radeon RX Vega 64 Liquid": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 4096
    },
    "Radeon VII": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 3840
    },
    "Radeon RX 7600S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 7600M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 7600M XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 7700S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 7900M": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 4608
    },
    "Radeon RX 6300M": {
        "Brand": "AMD",
        "Size (GB)": 2,
        "Shaders": 768
    },
    "Radeon RX 6450M": {
        "Brand": "AMD",
        "Size (GB)": 2,
        "Shaders": 768
    },
    "Radeon RX 6550S": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 768
    },
    "Radeon RX 6500M": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1024
    },
    "Radeon RX 6550M": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1024
    },
    "Radeon RX 6600S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6700S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6600M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6650M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6800S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6650M XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6700M": {
        "Brand": "AMD",
        "Size (GB)": 10,
        "Shaders": 2304
    },
    "Radeon RX 6800M": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 2560
    },
    "Radeon RX 6850M XT": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 2560
    },
    "Radeon RX 5300M": {
        "Brand": "AMD",
        "Size (GB)": 3,
        "Shaders": 1408
    },
    "Radeon RX 5500M": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5600M": {
        "Brand": "AMD",
        "Size (GB)": 6,
        "Shaders": 2304
    },
    "Radeon RX 5700M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2304
    }
}

GPUS_INTEL = {
    "Intel Arc A310": {
        "Brand": "Intel",
        "Size (GB)": 4,
        "Shading Cores": 768
    },
    "Intel Arc A380": {
        "Brand": "Intel",
        "Size (GB)": 6,
        "Shading Cores": 1024
    },
    "Intel Arc B570": {
        "Brand": "Intel",
        "Size (GB)": 10,
        "Shading Cores": 2304
    },
    "Intel Arc B580": {
        "Brand": "Intel",
        "Size (GB)": 12,
        "Shading Cores": 2560
    },
    "Intel Arc A580": {
        "Brand": "Intel",
        "Size (GB)": 8,
        "Shading Cores": 3072
    },
    "Intel Arc A750": {
        "Brand": "Intel",
        "Size (GB)": 8,
        "Shading Cores": 3584
    },
    "Intel Arc A770 8GB": {
        "Brand": "Intel",
        "Size (GB)": 8,
        "Shading Cores": 4096
    },
    "Intel Arc A770 16GB": {
        "Brand": "Intel",
        "Size (GB)": 16,
        "Shading Cores": 4096
    }
}

master_questions = [
    "What are the main features and capabilities of this program?",
    "How does LM Studio integrate with this program?",
    "What are local models and how can I access them through Huggingface?",
    "How do I obtain and use a Huggingface access token?",
    "What is the relationship between context limits and chunk sizes?", 
    "How do token limits affect the performance of vector and chat models?",
    "What is the optimal number of contexts to retrieve when querying the vector database?",
    "What purpose does the 'chunks only' checkbox serve?",
    "How do embedding models convert text into vectors for database storage?",
    "What are the main differences between Sentence Retrieval Models and Generalist Models?",
    "What factors should I consider when choosing an embedding model?",
    "How does the dimension size of a vector model affect its performance?",
    "What is the purpose of the half-precision checkbox setting?",
    "What vision models are available in the program and how do they differ?",
    "What are the key differences between Florence2, Moondream2, and Llava vision models?",
    "How does the program utilize Whisper for voice recording and audio transcription?",
    "What are the differences between distil variants and regular Whisper models?",
    "How do floating point formats affect model performance and accuracy?",
    "What are the main differences between float32, float16, and bfloat16 formats?",
    "How does quantization impact model size and performance?",
    "What LM Studio settings are available and how do they affect responses?",
    "How does the temperature setting influence LM Studio's output?",
    "What role do prefix and suffix settings play in prompt formatting?",
    "How do the device, similarity, and contexts settings interact when querying the database?",
    "What is the optimal chunk size for different types of documents?",
    "How does the chunk overlap setting affect context continuity?",
    "What purpose does the similarity threshold serve when querying the database?",
    "How does the search term filter affect context retrieval?",
    "What are the differences between the available text-to-speech backends?",
    "How can I choose between Bark and WhisperSpeech for text-to-speech conversion?",
    "What backup and restoration features are available for databases?",
    "How can I restore a default configuration if the config.yaml file is lost?",
    "What strategies are effective for searching the vector database?",
    "How can I manage VRAM efficiently when using the program?",
    "What is the relationship between maximum context length and maximum sequence length?",
    "How does the documentation scraping feature work?",
    "What vector models are available and how do I download them?",
    "What functionality does the Manage Databases tab provide?",
    "How do I create a new vector database with multiple file types?",
    "What is the purpose of the File Type setting when querying?",
    "How does the program handle audio file transcription?",
    "What happens when processing files with special characters?",
    "How can I optimize search results for technical documentation?",
    "What's the best way to handle code snippets in the database?",
    "How does the program maintain document structure in searches?",
    "What are the limitations of the free Google TTS API?",
    "How does the program handle embedded images in documents?",
    "What determines the speed of database creation?",
    "How can I optimize searches for specific programming languages?",
    "What happens when processing encrypted documents?",
    "What role do LM Studio's prompt templates play in shaping the final query sent to the model?",
    "Why is it advisable to use a GPU during database creation and a CPU during querying?",
    "How can I prevent sending too many tokens when retrieving multiple contexts from the database?",
    "What practical signs indicate that my chosen chunk size is mismatched for my text content?",
    "How does token truncation by the embedding model affect the semantic integrity of the output?",
    "What experimentation can I do in the Tools tab to compare different vision or transcription models?",
    "Why might creating dedicated databases for images and text improve the relevance of search results?",
    "How do I determine when to transition from a sentence retrieval model to a more versatile generalist model?",
    "What adjustments can I make if my GPU struggles to run a particularly large vision model?",
    "What considerations arise when mixing images, audio transcriptions, and documents in one vector database?",
    "How can I leverage audio transcriptions for more effective information retrieval within the database?",
    "In what scenarios should I modify the Whisper batch size to balance speed and GPU memory usage?",
    "What subtle effects can quantization have on the semantic relationships stored within vector embeddings?",
    "When should I try post-training quantization versus quantization-aware training?",
    "When might disabling automatic prompt formatting in LM Studio lead to better query results?",
    "What steps should I take if I suspect my similarity threshold is either too lenient or too strict?",
    "When dealing with small documents, how can I adjust chunking strategies for optimal retrieval?",
    "Is it possible to switch between LM Studio and local models mid-process to refine query responses?",
    "How can I mix different embedding models to tailor the vector database for specialized queries?",
    "What actions can I take if the chat model returns errors due to insufficient available token space?",
    "What features does the Manage Databases tab provide to reorganize or inspect existing vector databases?",
    "How can testing different audio file formats improve transcription accuracy in the vector database?",
    "What are the pros and cons of running vision models entirely on a CPU instead of a GPU?",
    "What practical performance differences might I see when using sentence-t5 versus other generalist models?",
    "What quick experiments can I run to ensure that the vector database retrieval aligns with my search needs?",
    "How does blending image summaries with textual data affect the semantic similarity search results?",
    "Is there a way to continuously update the vector database with newly scraped or added documentation?",
    "How can changes in chunk size, overlap, and similarity influence the final set of returned contexts?",
    "Overview of Program",
    "What is LM Studio?",
    "What are embedding or vector models?",
    "What are local models and how do I use them?",
    "What local models are available to use?",
    "How do I get a huggingface access token?",
    "What are context limits for a chat model?",
    "What happens if I exceed the context limit or maximum sequence length and how does the chunk size and overlap setting relate?",
    "How many context should I retrieve when querying the vector database?",
    "What does the chunks only checkbox do?",
    "Which embedding or vector model should I choose?",
    "What are the characteristics of vector or embedding models?",
    "What are the dimensions of a vector or embedding model?",
    "Tips for using vector or embedding models",
    "What Are Vision Models?",
    "What vision models are available in this program?",
    "Do you have any tips for choosing a vision model?",
    "What is whisper and how does this program use voice recording or transcribing an audio file?",
    "What do the Whisper models do?",
    "How can I record my question for the vector database query?",
    "How can I transcribe an audio file to be put into the vector database?",
    "What is a good batch size to use when transcribing an audio file in this program?",
    "What are the distil variants of the whisper models when transcribing and audio file?",
    "What whisper model should I choose to transcribe a file?",
    "What are floating point formats, precision, and quantization?",
    "What are the common floating point formats?",
    "What does float16 mean in LLMs?",
    "What is the bfloat16 floating point format?",
    "What is the float16 floating point format?",
    "What does exponent mean in floating point formats?",
    "What are precision and range in floating point formats?",
    "What is the difference betwen float32, bfloat16 and float16?",
    "What is quantization?",
    "What's the difference between post-training and quantization-aware training?",
    "What are the LM Studio Server Settings and what do they do?",
    "What are the database creation settings and what do they do?",
    "What is the Device setting when creating or querying a vector database?",
    "What is the chunk size setting when creating a vector database?",
    "What is the chunk overlap setting when creating a vector database?",
    "What is the contexts setting when querying the vector database?",
    "What is the similarity setting when querying the vector database?",
    "What is the search term filter setting when querying the vector database?",
    "What is the File Type setting when querying the vector database?",
    "What are text to speech models (aka TTS models) and how are they used in this program?",
    "Which text to speech backend or models should I use",
    "Can I back up or restore my vector databases and are they backed up automatically",
    "What happens if I lose a configuration file and can I restore it?",
    "What are some good tips for searching a vector database?",
    "How can I conserve memory or vram usage for this program?",
    "What device is best for querying a vector database?",
    "What are maximunm context length of a chat model and and maximum sequence length of an embedding model?",
    "What is the scrape documentaton feature in this program?",
    "Which vector or embedding models are available in this program?",
    "What are the embedding models created by Alibaba?",
    "What are the arctic embedding models?",
    "What are the knunlp or instructor embedding models?",
    "What are the intfloat embedding models?",
    "What are the sentence transformer or sentence-t5 embedding models?",
    "What are the IBM or granite embedding models?",
    "What is the InternVL2.5-1b vision model?",
    "What is the Florence2-Base vision model?",
    "What is the InternVL2.5-4b vision model?",
    "What is the Moondream2 vision model?",
    "What is the Mississippi vision model?",
    "What is the Ovis1.6-Llama3.2 vision model?",
    "What is the GLM4v vision model?",
    "What is the Molmo-D-0924 vision model?",
    "What is the Llava 1.6 vision model?",
    "What are the exaone chat models?",
    "What are the qwen 2.5 coder chat models?",
    "What are the qwen chat models and not the coder models?",
    "What is the mistral or mistral small chat model?",
    "What are the IBM or granite chat models?",
    "What is the manage databaes tab?",
    "How can I create a vector database?",
    "What is the Query Database Tab",
    "What is the Tools Tab?",
    "What is the Create Database Tab?",
    "What is the manage databases tab?",
    "What is the Settings Tab?",
    "What is the Models Tab?",
    "What is the max tokens setting?",
    "What are the prefix and suffix settings?",
    "What does precision mean?"
]