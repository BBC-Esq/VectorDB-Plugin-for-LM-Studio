system_message = """You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. Here are one or more contexts to solely base your answer off of. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."""

MODEL_MAX_TOKENS = {
    'Danube 3 - 4b': 8192,
    'Dolphin-Qwen 2 - 1.5b': 8192,
    'Phi 3.5 Mini - 4b': 8192,
    'Internlm2_5 - 7b': 8192,
    'CodeQwen 1.5 - 7b': 8192,
    'Dolphin-Llama 3.1 - 8b': 8192,
    'Hermes-3-Llama-3.1 - 8b': 8192,
    'Dolphin-Qwen 2 - 7b': 8192,
    'Yi Coder - 9b': 8192,
    'Dolphin-Mistral-Nemo - 12b': 8192,
    'DeepSeek Coder v2 - 16b': 8192,
    'Internlm2_5 - 20b': 8192,
}

MODEL_MAX_NEW_TOKENS = {
    'Danube 3 - 4b': 1024,
    'Dolphin-Qwen 2 - 1.5b': 8192,
    'Phi 3.5 Mini - 4b': 2048,
    'Internlm2_5 - 7b': 8192,
    'Dolphin-Llama 3.1 - 8b': 1024,
    'Hermes-3-Llama-3.1 - 8b': 8192,
    'Dolphin-Qwen 2 - 7b': 1024,
    'Dolphin-Mistral-Nemo - 12b': 8192,
    'Internlm2_5 - 20b': 8192,
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
    'BAAI': [
        {
            'name': 'bge-small-en-v1.5',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'BAAI/bge-small-en-v1.5',
            'cache_dir': 'BAAI--bge-small-en-v1.5',
            'type': 'vector'
        },
        {
            'name': 'bge-base-en-v1.5',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438,
            'repo_id': 'BAAI/bge-base-en-v1.5',
            'cache_dir': 'BAAI--bge-base-en-v1.5',
            'type': 'vector'
        },
        {
            'name': 'bge-large-en-v1.5',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'BAAI/bge-large-en-v1.5',
            'cache_dir': 'BAAI--bge-large-en-v1.5',
            'type': 'vector'
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
            'type': 'vector'
        },
        {
            'name': 'instructor-large',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'hkunlp/instructor-large',
            'cache_dir': 'hkunlp--instructor-large',
            'type': 'vector'
        },
        {
            'name': 'instructor-xl',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 4960,
            'repo_id': 'hkunlp/instructor-xl',
            'cache_dir': 'hkunlp--instructor-xl',
            'type': 'vector'
        },
    ],
    'sentence-transformers': [
        {
            'name': 'all-MiniLM-L12-v2',
            'dimensions': 384,
            'max_sequence': 256,
            'size_mb': 120,
            'repo_id': 'sentence-transformers/all-MiniLM-L12-v2',
            'cache_dir': 'sentence-transformers--all-MiniLM-L12-v2',
            'type': 'vector'
        },
        {
            'name': 'all-mpnet-base-v2',
            'dimensions': 768,
            'max_sequence': 384,
            'size_mb': 438,
            'repo_id': 'sentence-transformers/all-mpnet-base-v2',
            'cache_dir': 'sentence-transformers--all-mpnet-base-v2',
            'type': 'vector'
        },
        {
            'name': 'sentence-t5-base',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 219,
            'repo_id': 'sentence-transformers/sentence-t5-base',
            'cache_dir': 'sentence-transformers--sentence-t5-base',
            'type': 'vector'
        },
        {
            'name': 'sentence-t5-large',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 670,
            'repo_id': 'sentence-transformers/sentence-t5-large',
            'cache_dir': 'sentence-transformers--sentence-t5-large',
            'type': 'vector'
        },
        {
            'name': 'sentence-t5-xl',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 2480,
            'repo_id': 'sentence-transformers/sentence-t5-xl',
            'cache_dir': 'sentence-transformers--sentence-t5-xl',
            'type': 'vector'
        },
    ],
    'Alibaba-NLP': [
        {
            'name': 'Alibaba-gte-base',
            'dimensions': 768,
            'max_sequence': 8192,
            'size_mb': 547,
            'repo_id': 'Alibaba-NLP/gte-base-en-v1.5',
            'cache_dir': 'Alibaba-NLP--gte-base-en-v1.5',
            'type': 'vector'
        },
        {
            'name': 'Alibaba-gte-large',
            'dimensions': 1024,
            'max_sequence': 8192,
            'size_mb': 1740,
            'repo_id': 'Alibaba-NLP/gte-large-en-v1.5',
            'cache_dir': 'Alibaba-NLP--gte-large-en-v1.5',
            'type': 'vector'
        },
    ],
    'thenlper': [
        {
            'name': 'gte-small',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 67,
            'repo_id': 'thenlper/gte-small',
            'cache_dir': 'thenlper--gte-small',
            'type': 'vector'
        },
        {
            'name': 'gte-base',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 219,
            'repo_id': 'thenlper/gte-base',
            'cache_dir': 'thenlper--gte-base',
            'type': 'vector'
        },
        {
            'name': 'gte-large',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 670,
            'repo_id': 'thenlper/gte-large',
            'cache_dir': 'thenlper--gte-large',
            'type': 'vector'
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
            'type': 'vector'
        },
        {
            'name': 'e5-base-v2',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438,
            'repo_id': 'intfloat/e5-base-v2',
            'cache_dir': 'intfloat--e5-base-v2',
            'type': 'vector'
        },
        {
            'name': 'e5-large-v2',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'intfloat/e5-large-v2',
            'cache_dir': 'intfloat--e5-large-v2',
            'type': 'vector'
        },
    ],
}

CHAT_MODELS = {
    'Zephyr - 1.6b': {
        'model': 'Zephyr - 1.6b',
        'repo_id': 'stabilityai/stablelm-2-zephyr-1_6b',
        'cache_dir': 'stabilityai--stablelm-2-zephyr-1_6b',
        'tokens_per_second': 74,
        'context_length': 4096,
        'avg_vram_usage': '2.5 GB',
        'function': 'Zephyr_1_6B',
        'precision': 'float16'
    },
    'Zephyr - 3b': {
        'model': 'Zephyr - 3b',
        'repo_id': 'stabilityai/stablelm-zephyr-3b',
        'cache_dir': 'stabilityai--stablelm-zephyr-3b',
        'tokens_per_second': 57,
        'context_length': 4096,
        'avg_vram_usage': '2.9 GB',
        'function': 'Zephyr_3B',
        'precision': 'bfloat16'
    },
    'Danube 3 - 4b': {
        'model': 'Danube 3 - 4b',
        'repo_id': 'h2oai/h2o-danube3-4b-chat',
        'cache_dir': 'h2oai/h2o-danube3-4b-chat',
        'tokens_per_second': 65,
        'context_length': 8192,
        'avg_vram_usage': '3.3 GB',
        'function': 'Danube_3_4b',
        'precision': 'bfloat16'
    },
    'Phi 3.5 Mini - 4b': {
        'model': 'Phi 3.5 Mini - 4b',
        'repo_id': 'microsoft/Phi-3.5-mini-instruct',
        'cache_dir': 'microsoft--Phi-3.5-mini-instruct',
        'tokens_per_second': 40,
        'context_length': 8192,
        'avg_vram_usage': '3.8 GB',
        'function': 'Phi3_5_mini_4b',
        'precision': 'bfloat16'
    },
    'Dolphin-Qwen 2 - 1.5b': {
        'model': 'Dolphin-Qwen 2 - 1.5b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.3-qwen2-1.5b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.3-qwen2-1.5b',
        'tokens_per_second': 58.07,
        'context_length': 16384,
        'avg_vram_usage': '4.2 GB',
        'function': 'Dolphin_Qwen2_1_5b',
        'precision': 'bfloat16'
    },
    'Orca 2 - 7b': {
        'model': 'Orca 2 - 7b',
        'repo_id': 'microsoft/Orca-2-7b',
        'cache_dir': 'microsoft--Orca-2-7b',
        'tokens_per_second': 47.10,
        'context_length': 4096,
        'avg_vram_usage': '5.9 GB',
        'function': 'Orca2_7b',
        'precision': 'float16'
    },
    'Neural-Chat - 7b': {
        'model': 'Neural-Chat - 7b',
        'repo_id': 'Intel/neural-chat-7b-v3-3',
        'cache_dir': 'Intel--neural-chat-7b-v3-3',
        'tokens_per_second': 46,
        'context_length': 4096,
        'avg_vram_usage': '5.8 GB',
        'function': 'Neural_Chat_7b',
        'precision': 'float16'
    },
    'Internlm2_5 - 7b': {
        'model': 'Internlm2_5 - 7b',
        'repo_id': 'internlm/internlm2_5-7b-chat',
        'cache_dir': 'internlm--internlm2_5-7b-chat',
        'tokens_per_second': 35.12,
        'context_length': 32768,
        'avg_vram_usage': '6.8 GB',
        'function': 'InternLM2_5_7b',
        'precision': 'bfloat16'
    },
    'Dolphin-Llama 3.1 - 8b': {
        'model': 'Dolphin-Llama 3.1 - 8b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.4-llama3.1-8b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.4-llama3.1-8b',
        'tokens_per_second': 50.33,
        'context_length': 8192,
        'avg_vram_usage': '7.1 GB',
        'function': 'Dolphin_Llama3_1_8B',
        'precision': 'bfloat16'
    },
    'Hermes-3-Llama-3.1 - 8b': {
        'model': 'Hermes-3-Llama-3.1 - 8b',
        'repo_id': 'NousResearch/Hermes-3-Llama-3.1-8B',
        'cache_dir': 'NousResearch--Hermes-3-Llama-3.1-8B',
        'tokens_per_second': 46.70,
        'context_length': 8192,
        'avg_vram_usage': '7.1 GB',
        'function': 'Hermes_3_Llama_3_1',
        'precision': 'bfloat16'
    },
    'Dolphin-Yi 1.5 - 9b': {
        'model': 'Dolphin-Yi 1.5 - 9b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.1-yi-1.5-9b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.1-yi-1.5-9b',
        'tokens_per_second': 30.85,
        'context_length': 4096,
        'avg_vram_usage': '7.2 GB',
        'function': 'Dolphin_Yi_1_5_9b',
        'precision': 'bfloat16'
    },

    'Yi Coder - 9b': {
        'model': 'Yi Coder - 9b',
        'repo_id': '01-ai/Yi-Coder-9B-Chat',
        'cache_dir': '01-ai--Yi-Coder-9B-Chat',
        'tokens_per_second': 30.85,
        'context_length': 8192,
        'avg_vram_usage': '7.2 GB',
        'function': 'Yi_Coder_9b',
        'precision': 'bfloat16'
    },

    'Orca 2 - 13b': {
        'model': 'Orca 2 - 13b',
        'repo_id': 'microsoft/Orca-2-13b',
        'cache_dir': 'microsoft--Orca-2-13b',
        'tokens_per_second': 36.11,
        'context_length': 4096,
        'avg_vram_usage': '9.9 GB',
        'function': 'Orca2_13b',
        'precision': 'float16'
    },
    'Dolphin-Qwen 2 - 7b': {
        'model': 'Dolphin-Qwen 2 - 7b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.2-qwen2-7b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.2-qwen2-7b',
        'tokens_per_second': 52,
        'context_length': 16384,
        'avg_vram_usage': '9.2 GB',
        'function': 'Dolphin_Qwen2_7b',
        'precision': 'bfloat16'
    },
    'CodeQwen 1.5 - 7b': {
        'model': 'CodeQwen 1.5 - 7b',
        'repo_id': 'Qwen/CodeQwen1.5-7B-Chat',
        'cache_dir': 'Qwen--CodeQwen1.5-7B-Chat',
        'tokens_per_second': 52,
        'context_length': 16384,
        'avg_vram_usage': '9.2 GB',
        'function': 'CodeQwen1_5_7b_chat',
        'precision': 'bfloat16'
    },
    'Dolphin-Phi 3 - Medium': {
        'model': 'Dolphin-Phi 3 - Medium',
        'repo_id': 'cognitivecomputations/dolphin-2.9.2-Phi-3-Medium',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.2-Phi-3-Medium',
        'tokens_per_second': 40,
        'context_length': 4096,
        'avg_vram_usage': '9.3 GB',
        'function': 'Dolphin_Phi3_Medium',
        'precision': 'bfloat16'
    },
    'SOLAR - 10.7b': {
        'model': 'SOLAR - 10.7b',
        'repo_id': 'upstage/SOLAR-10.7B-Instruct-v1.0',
        'cache_dir': 'upstage--SOLAR-10.7B-Instruct-v1.0',
        'tokens_per_second': 28,
        'context_length': 4096,
        'avg_vram_usage': '9.3 GB',
        'function': 'SOLAR_10_7B',
        'precision': 'float16'
    },
    'Llama 2 - 13b': {
        'model': 'Llama 2 - 13b',
        'repo_id': 'meta-llama/Llama-2-13b-chat-hf',
        'cache_dir': 'meta-llama--Llama-2-13b-chat-hf',
        'tokens_per_second': 36.80,
        'context_length': 4096,
        'avg_vram_usage': '10.0 GB',
        'function': 'Llama2_13b',
        'precision': 'float16'
    },
    'Dolphin-Mistral-Nemo - 12b': {
        'model': 'Dolphin-Mistral-Nemo - 12b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.3-mistral-nemo-12b',
        'tokens_per_second': 35.86,
        'context_length': 8192,
        'avg_vram_usage': '10.0 GB',
        'function': 'Dolphin_Mistral_Nemo',
        'precision': 'bfloat16'
    },
    'DeepSeek Coder v2 - 16b': {
        'model': 'DeepSeek Coder v2 - 16b',
        'repo_id': 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        'cache_dir': 'deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct',
        'tokens_per_second': 35.86,
        'context_length': 8192,
        'avg_vram_usage': '10.0 GB',
        'function': 'DeepSeek_Coder_v2_lite',
        'precision': 'bfloat16'
    },
    'Internlm2_5 - 20b': {
        'model': 'Internlm2_5 - 20b',
        'repo_id': 'internlm/internlm2_5-20b-chat',
        'cache_dir': 'internlm--internlm2_5-20b-chat',
        'tokens_per_second': 20.21,
        'context_length': 32768,
        'avg_vram_usage': '14.2 GB',
        'function': 'InternLM2_5_20b',
        'precision': 'bfloat16'
    },
}

VISION_MODELS = {
    'Florence-2-base': {
        'precision': 'autoselect',
        'quant': 'autoselect',
        'size': '232m',
        'repo_id': 'microsoft/Florence-2-base',
        'cache_dir': 'microsoft--Florence-2-base',
        'requires_cuda': False,
        'avg_vram_usage': '2.6 GB',
        'tokens_per_second': 163.06
    },
    'Moondream2 - 1.9b': {
        'precision': 'float16',
        'quant': 'none',
        'size': '2b',
        'repo_id': 'vikhyatk/moondream2',
        'cache_dir': 'vikhyatk--moondream2',
        'requires_cuda': True,
        'avg_vram_usage': '4.6 GB',
        'tokens_per_second': 82.28
    },
    'Florence-2-large': {
        'precision': 'autoselect',
        'quant': 'autoselect',
        'size': '770m',
        'repo_id': 'microsoft/Florence-2-large',
        'cache_dir': 'microsoft--Florence-2-large',
        'requires_cuda': False,
        'avg_vram_usage': '5.3 GB',
        'tokens_per_second': 113.32
    },
    'Phi-3-vision - 4.2b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '4.2b',
        'repo_id': 'microsoft/Phi-3-vision-128k-instruct',
        'cache_dir': 'microsoft--Phi-3-vision-128k-instruct',
        'requires_cuda': True,
        'avg_vram_usage': '5.4 GB',
        'tokens_per_second': 30.72
    },
    'Llava 1.5 - 7b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/llava-1.5-7b-hf',
        'cache_dir': 'llava-hf--llava-1.5-7b-hf',
        'requires_cuda': True,
        'avg_vram_usage': '5.8 GB',
        'tokens_per_second': 48.30
    },
    'Bakllava 1.5 - 7b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/bakLlava-v1-hf',
        'cache_dir': 'llava-hf--bakLlava-v1-hf',
        'requires_cuda': True,
        'avg_vram_usage': '5.9 GB',
        'tokens_per_second': 48.30
    },
    'Llava 1.6 Vicuna - 7b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/llava-v1.6-vicuna-7b-hf',
        'cache_dir': 'llava-hf--llava-v1.6-vicuna-7b-hf',
        'requires_cuda': True,
        'avg_vram_usage': '7.9 GB',
        'tokens_per_second': 56.33
    },
    'MiniCPM-V-2_6 - 8b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '8b',
        'repo_id': 'openbmb/MiniCPM-V-2_6-int4',
        'cache_dir': 'openbmb--MiniCPM-V-2_6-int4',
        'requires_cuda': True,
        'avg_vram_usage': '9.1 GB',
        'tokens_per_second': 16.40
    },
    'Llava 1.5 - 13b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '13b',
        'repo_id': 'llava-hf/llava-1.5-13b-hf',
        'cache_dir': 'llava-hf--llava-1.5-13b-hf',
        'requires_cuda': True,
        'avg_vram_usage': '9.8 GB',
        'tokens_per_second': 38.03
    },
    'falcon-vlm - 11b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '13b',
        'repo_id': 'tiiuae/falcon-11B-vlm',
        'cache_dir': 'tiiuae--falcon-11B-vlm',
        'requires_cuda': True,
        'avg_vram_usage': '12.8 GB',
        'tokens_per_second': 18.36
    },
    'Llava 1.6 Vicuna - 13b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '13b',
        'repo_id': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'cache_dir': 'llava-hf--llava-v1.6-vicuna-13b-hf',
        'requires_cuda': True,
        'avg_vram_usage': '14.1 GB',
        'tokens_per_second': 41.43
    }
}

WHISPER_MODELS = {
    # LARGE-V3
    'Distil Whisper large-v3 - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float32',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper large-v3 - bfloat16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-bfloat16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper large-v3 - float16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Whisper large-v3 - float32': {
        'name': 'Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float32',
        'tokens_per_second': 85,
        'optimal_batch_size': 2,
        'avg_vram_usage': '5.5 GB'
    },
    'Whisper large-v3 - bfloat16': {
        'name': 'Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-bfloat16',
        'tokens_per_second': 95,
        'optimal_batch_size': 3,
        'avg_vram_usage': '3.8 GB'
    },
    'Whisper large-v3 - float16': {
        'name': 'Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float16',
        'tokens_per_second': 100,
        'optimal_batch_size': 3,
        'avg_vram_usage': '3.3 GB'
    },
    # MEDIUM.EN
    'Distil Whisper medium.en - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float32',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper medium.en - bfloat16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-bfloat16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper medium.en - float16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Whisper medium.en - float32': {
        'name': 'Whisper medium.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float32',
        'tokens_per_second': 130,
        'optimal_batch_size': 6,
        'avg_vram_usage': '2.5 GB'
    },
    'Whisper medium.en - bfloat16': {
        'name': 'Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-bfloat16',
        'tokens_per_second': 140,
        'optimal_batch_size': 7,
        'avg_vram_usage': '2.0 GB'
    },
    'Whisper medium.en - float16': {
        'name': 'Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float16',
        'tokens_per_second': 145,
        'optimal_batch_size': 7,
        'avg_vram_usage': '1.8 GB'
    },
    # SMALL.EN
    'Distil Whisper small.en - float32': {
        'name': 'Distil Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float32',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper small.en - bfloat16': {
        'name': 'Distil Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-bfloat16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper small.en - float16': {
        'name': 'Distil Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Whisper small.en - float32': {
        'name': 'Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float32',
        'tokens_per_second': 180,
        'optimal_batch_size': 14,
        'avg_vram_usage': '1.5 GB'
    },
    'Whisper small.en - bfloat16': {
        'name': 'Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-bfloat16',
        'tokens_per_second': 190,
        'optimal_batch_size': 15,
        'avg_vram_usage': '1.2 GB'
    },
    'Whisper small.en - float16': {
        'name': 'Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float16',
        'tokens_per_second': 195,
        'optimal_batch_size': 15,
        'avg_vram_usage': '1.1 GB'
    },
    # BASE.EN
    'Whisper base.en - float32': {
        'name': 'Whisper base.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float32',
        'tokens_per_second': 230,
        'optimal_batch_size': 22,
        'avg_vram_usage': '1.0 GB'
    },
    'Whisper base.en - bfloat16': {
        'name': 'Whisper base.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-bfloat16',
        'tokens_per_second': 240,
        'optimal_batch_size': 23,
        'avg_vram_usage': '0.85 GB'
    },
    'Whisper base.en - float16': {
        'name': 'Whisper base.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float16',
        'tokens_per_second': 245,
        'optimal_batch_size': 23,
        'avg_vram_usage': '0.8 GB'
    },
    # TINY.EN
    'Whisper tiny.en - float32': {
        'name': 'Whisper tiny.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float32',
        'tokens_per_second': 280,
        'optimal_batch_size': 30,
        'avg_vram_usage': '0.7 GB'
    },
    'Whisper tiny.en - bfloat16': {
        'name': 'Whisper tiny.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-bfloat16',
        'tokens_per_second': 290,
        'optimal_batch_size': 31,
        'avg_vram_usage': '0.6 GB'
    },
    'Whisper tiny.en - float16': {
        'name': 'Whisper tiny.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float16',
        'tokens_per_second': 295,
        'optimal_batch_size': 31,
        'avg_vram_usage': '0.55 GB'
    },
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
    "PORT": "Must match the port used in LM Studio.",
    "MAX_TOKENS": "Maximum tokens for LLM response. -1 for unlimited.",
    "TEMPERATURE": "Controls LLM creativity. 0-1, higher is more creative.",
    "PREFIX_SUFFIX": "Prompt format for LLM. Use preset or custom for different models.",
    "DISABLE_PROMPT_FORMATTING": "Disables built-in prompt formatting. Use LM Studio settings instead.",
    "CREATE_DEVICE_DB": "Choose 'cpu' or 'cuda' based on hardware. Prefer 'cuda' if available.",
    "CHUNK_SIZE": "Text chunk max characters. Make sure it falls within the Max Sequence of the vector model.  3-4 characters = 1 token.",
    "CHUNK_OVERLAP": "Characters shared between chunks. Set to 25-50% of chunk size.",
    "HALF_PRECISION": "Uses bfloat16/float16 for 2x speedup. GPU only.",
    "CREATE_DEVICE_QUERY": "Choose 'cpu' or 'cuda'. 'cpu' recommended to conserve VRAM.",
    "CONTEXTS": "Maximum number of chunks/contexts to return.",
    "SIMILARITY": "Relevance threshold for chunks. 0-1, higher returns more. Don't use 1.",
    "SEARCH_TERM_FILTER": "Removes chunks without exact term. Case-insensitive.",
    "FILE_TYPE_FILTER": "Filters chunks by document type (images, audio, documents, all).",
    "TTS_MODEL": "Choose TTS model. Bark offers customization, Google requires internet.",
    "VISION_MODEL": "Select vision model for image processing. Test before bulk processing.",
    "RESTORE_DATABASE": "Restores backed-up databases. Use with caution.",
    "RESTORE_CONFIG": "Restores original config.yaml. May require manual database cleanup.",
    "VECTOR_MODEL_SELECT": "Choose a vector model to download.",
    "VECTOR_MODEL_NAME": "The name of the vector model.",
    "VECTOR_MODEL_DIMENSIONS": "Higher dimensions captures more nuance but requires more processing time.",
    "VECTOR_MODEL_MAX_SEQUENCE": "Number of tokens the model can process at once. Different from the Chunk Size setting, which is in characters.",
    "VECTOR_MODEL_SIZE": "Size on disk.",
    "VECTOR_MODEL_DOWNLOADED": "Whether the model has been downloaded.",
    "VECTOR_MODEL_LINK": "Huggingface link.",
    "DOWNLOAD_MODEL": "Download the selected vector model.",
    "WHISPER_MODEL_SELECT": "Distil models use ~ 70% VRAM of their non-Distil equivalents with little quality loss.",
    "WHISPER_BATCH_SIZE": "Batch size for transcription. See the User Guid for optimal values.",
    "AUDIO_FILE_SELECT": "Select an audio file. Supports various audio formats.",
    "TRANSCRIBE_BUTTON": "Start transcription.",
    "CHOOSE_FILES": "Select documents to add to the database. Remember to transcribe audio files in the Tools tab first.",
    "SELECT_VECTOR_MODEL": "Choose the vector model for text embedding.",
    "DATABASE_NAME_INPUT": "Enter a unique database name. Use only lowercase letters, numbers, underscores, and hyphens.",
    "CREATE_VECTOR_DB": "Create a new vector database.",
    "DATABASE_SELECT": "Select the vector database to query for relevant information.",
    "MODEL_BACKEND_SELECT": "Choose the backend for the large language model response.",
    "LOCAL_MODEL_SELECT": "Select a local model for generating responses.",
    "EJECT_LOCAL_MODEL": "Unload the current local model from memory.",
    "QUESTION_INPUT": "Type your question here or use the voice recorder.",
    "VOICE_RECORDER": "Click to start recording, speak your question, then click again to stop recording.",
    "SPEAK_RESPONSE": "Speak the response from the large language model using text-to-speech.",
    "COPY_RESPONSE": "Copy the model's response to the clipboard.",
    "CHUNKS_ONLY": "Only return relevant chunks without connecting to the LLM. Extremely useful to test the chunk size/overlap settings."
}

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
    "GeForce RTX 3050": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 2048
    },
    "GeForce RTX 3050 Mobile/Laptop": {
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
    "GeForce RTX 3050": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2304
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
    "GeForce RTX 4080": {
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
    "GeForce RTX 4080": {
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
