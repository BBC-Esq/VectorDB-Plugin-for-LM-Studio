jeeves_system_message = "You are a helpful British butler who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address a question you answer based on what the contexts say and then briefly summarize the parts of the question that the contexts didn't provide an answer to.  Also, you should be very respectful to the person asking the question and frequently offer traditional butler services like various fancy drinks, snacks, various butler services like shining of shoes, pressing of suites, and stuff like that. Also, if you can't answer the question at all based on the provided contexts, you should apologize profusely and beg to keep your job."
system_message = "You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."
rag_string = "Here are the contexts to base your answer on.  However, I need to reiterate that I only want you to base your response on these contexts and do not use outside knowledge that you may have been trained with."

# changes the default of 8192 in module_chat.py
MODEL_MAX_TOKENS = {
    'Qwen 2.5 - 1.5b': 4096,
    'Qwen 2.5 Coder - 1.5b': 4096,
    'Zephyr - 1.6b': 4096,
    'Zephyr - 3b': 4096,
    'Qwen 2.5 - 3b': 4096,
    'Llama 3.2 - 3b': 4096,
    'Internlm2_5 - 1.8b': 4096
}

# changes the default of 1024 in module_chat.mpy
MODEL_MAX_NEW_TOKENS = {
    'Qwen 2.5 - 1.5b': 512,
    'Qwen 2.5 Coder - 1.5b': 512,
    'Zephyr - 1.6b': 512,
    'Zephyr - 3b': 512,
    'Qwen 2.5 - 3b': 512,
    'Internlm2_5 - 1.8b': 512,
}

CHAT_MODELS = {
    'Qwen 2.5 - 1.5b': {
        'model': 'Qwen 2.5 - 1.5b',
        'repo_id': 'Qwen/Qwen2.5-1.5B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-1.5B-Instruct',
        'cps': 261.31,
        'context_length': 32768,
        'vram': 1749.97,
        'function': 'Qwen2_5_1_5b',
        'gated': False,
    },
    'Qwen 2.5 Coder - 1.5b': {
        'model': 'Qwen 2.5 Coder - 1.5b',
        'repo_id': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-Coder-1.5B-Instruct',
        'cps': 236.32,
        'context_length': 4096,
        'vram': 1742.12,
        'function': 'QwenCoder_1_5b',
        'precision': 'bfloat16',
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
        'precision': 'float16',
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
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen 2.5 - 3b': {
        'model': 'Qwen 2.5 - 3b',
        'repo_id': 'Qwen/Qwen2.5-3B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-3B-Instruct',
        'cps': 213.40,
        'context_length': 32768,
        'vram': 2864.89,
        'function': 'Qwen2_5_3b',
        'gated': False,
    },
    'Llama 3.2 - 3b': {
        'model': 'Llama 3.2 - 3b',
        'repo_id': 'meta-llama/Llama-3.2-3B-Instruct',
        'cache_dir': 'meta-llama--Llama-3.2-3B-Instruct',
        'cps': 265.09,
        'context_length': 32768,
        'vram': 3003.67,
        'function': 'Llama_3_2_3b',
        'gated': True,
    },
    'Internlm2_5 - 1.8b': {
        'model': 'Internlm2_5 - 1.8b',
        'repo_id': 'internlm/internlm2_5-1_8b-chat',
        'cache_dir': 'internlm--internlm2_5-1_8b-chat',
        'cps': 262.98,
        'context_length': 32768,
        'vram': 3019.65,
        'function': 'InternLM2_5_1_8b',
        'gated': False,
    },
    'Phi 3.5 Mini - 4b': {
        'model': 'Phi 3.5 Mini - 4b',
        'repo_id': 'microsoft/Phi-3.5-mini-instruct',
        'cache_dir': 'microsoft--Phi-3.5-mini-instruct',
        'cps': 139.05,
        'context_length': 8192,
        'vram': 3957.12,
        'function': 'Phi3_5_mini_4b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen 2.5 - 7b': {
        'model': 'Qwen 2.5 - 7b',
        'repo_id': 'Qwen/Qwen2.5-7B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-7B-Instruct',
        'cps': 226.22,
        'context_length': 32768,
        'vram': 6766.57,
        'function': 'Qwen2_5_7b',
        'gated': False,
    },
    'Qwen 2.5 Coder - 7b': {
        'model': 'Qwen 2.5 Coder - 7b',
        'repo_id': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-Coder-7B-Instruct',
        'cps': 219.55,
        'context_length': 4096,
        'vram': 6760.18,
        'function': 'QwenCoder_7b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Dolphin-Llama 3.1 - 8b': {
        'model': 'Dolphin-Llama 3.1 - 8b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.4-llama3.1-8b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.4-llama3.1-8b',
        'cps': 228.31,
        'context_length': 8192,
        'vram': 6598.98,
        'function': 'Dolphin_Llama3_1_8B',
        'gated': False,
    },
    'Yi Coder - 9b': {
        'model': 'Yi Coder - 9b',
        'repo_id': '01-ai/Yi-Coder-9B-Chat',
        'cache_dir': '01-ai--Yi-Coder-9B-Chat',
        'cps': 143.72,
        'context_length': 8192,
        'vram': 6500.29,
        'function': 'Yi_Coder_9b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Internlm2_5 - 7b': {
        'model': 'Internlm2_5 - 7b',
        'repo_id': 'internlm/internlm2_5-7b-chat',
        'cache_dir': 'internlm--internlm2_5-7b-chat',
        'cps': 156.65,
        'context_length': 32768,
        'vram': 6926.25,
        'function': 'InternLM2_5_7b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen 2.5 - 14b': {
        'model': 'Qwen 2.5 - 14b',
        'repo_id': 'Qwen/Qwen2.5-14B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-14B-Instruct',
        'cps': 139.26,
        'context_length': 8192,
        'vram': 12599.22,
        'function': 'Qwen_2_5_14b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'DeepSeek Coder v2 - 16b': {
        'model': 'DeepSeek Coder v2 - 16b',
        'repo_id': 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        'cache_dir': 'deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct',
        'cps': 79.17,
        'context_length': 8192,
        'vram': 11830.24,
        'function': 'DeepSeek_Coder_v2_lite',
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
    'Internlm2_5 - 20b': {
        'model': 'Internlm2_5 - 20b',
        'repo_id': 'internlm/internlm2_5-20b-chat',
        'cache_dir': 'internlm--internlm2_5-20b-chat',
        'cps': 101.13,
        'context_length': 32768,
        'vram': 14305.22,
        'function': 'InternLM2_5_20b',
        'precision': 'bfloat16',
        'gated': False,
    },
    'Qwen 2.5 - 32b': {
        'model': 'Qwen 2.5 - 32b',
        'repo_id': 'Qwen/Qwen2.5-32B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-32B-Instruct',
        'cps': 101.51,
        'context_length': 8192,
        'vram': 21128.30,
        'function': 'Qwen_2_5_32b',
        'precision': 'bfloat16',
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

}

VISION_MODELS = {
    'Florence-2-base': {
        'precision': 'autoselect',
        'quant': 'autoselect',
        'size': '232m',
        'repo_id': 'microsoft/Florence-2-base',
        'cache_dir': 'microsoft--Florence-2-base',
        'requires_cuda': False,
        'vram': '2.6 GB',
        'tps': 163.06
    },
    'Moondream2 - 1.9b': {
        'precision': 'float16',
        'quant': 'none',
        'size': '2b',
        'repo_id': 'vikhyatk/moondream2',
        'cache_dir': 'vikhyatk--moondream2',
        'requires_cuda': True,
        'vram': '4.6 GB',
        'tps': 82.28
    },
    'Florence-2-large': {
        'precision': 'autoselect',
        'quant': 'autoselect',
        'size': '770m',
        'repo_id': 'microsoft/Florence-2-large',
        'cache_dir': 'microsoft--Florence-2-large',
        'requires_cuda': False,
        'vram': '5.3 GB',
        'tps': 113.32
    },
    'Llava 1.6 Vicuna - 7b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/llava-v1.6-vicuna-7b-hf',
        'cache_dir': 'llava-hf--llava-v1.6-vicuna-7b-hf',
        'requires_cuda': True,
        'vram': '7.9 GB',
        'tps': 56.33
    },
    'MiniCPM-V-2_6 - 8b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '8b',
        'repo_id': 'openbmb/MiniCPM-V-2_6-int4',
        'cache_dir': 'openbmb--MiniCPM-V-2_6-int4',
        'requires_cuda': True,
        'vram': '9.1 GB',
        'tps': 16.99
    },
    # awaiting fix to custom modeling code on huggingface repo
    # 'THUDM glm4v - 9b': {
        # 'precision': 'bfloat16',
        # 'quant': '4-bit',
        # 'size': '9b',
        # 'repo_id': 'THUDM/glm-4v-9b',
        # 'cache_dir': 'THUDM--glm-4v-9b',
        # 'requires_cuda': True,
        # 'vram': '10.5 GB',
        # 'tps': 28.69
    # },
    # i need to add a sub-class
    # 'Molmo-D-0924 - 8b': {
        # 'precision': 'float32',
        # 'quant': '4-bit',
        # 'size': '8b',
        # 'repo_id': 'cyan2k/molmo-7B-D-bnb-4bit',
        # 'cache_dir': 'cyan2k--molmo-7B-D-bnb-4bit',
        # 'requires_cuda': True,
        # 'vram': '10.5 GB',
        # 'tps': 28.69
    # },
    'Llava 1.6 Vicuna - 13b': {
        'precision': 'float16',
        'quant': '4-bit',
        'size': '13b',
        'repo_id': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'cache_dir': 'llava-hf--llava-v1.6-vicuna-13b-hf',
        'requires_cuda': True,
        'vram': '14.1 GB',
        'tps': 41.43
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

scrape_documentation = {
    "Accelerate 0.34.2": {
        "URL": "https://huggingface.co/docs/accelerate/v0.34.2/en/",
        "folder": "accelerate_0342"
    },
    "Huggingface Hub": {
        "URL": "https://huggingface.co/docs/huggingface_hub/",
        "folder": "huggingface_hub"
    },
    "Optimum 1.22.0": {
        "URL": "https://huggingface.co/docs/optimum/v1.22.0/en/",
        "folder": "optimum_1220"
    },
    "Safetensors": {
        "URL": "https://huggingface.co/docs/safetensors/",
        "folder": "safetensors"
    },
    "Transformers 4.45.2": {
        "URL": "https://huggingface.co/docs/transformers/v4.45.2/en/",
        "folder": "transformers_4452"
    },
    "Langchain": {
        "URL": "https://api.python.langchain.com/en/latest/",
        "folder": "langchain"
    },
    "Torchaudio 2.4": {
        "URL": "https://pytorch.org/audio/2.4.0/",
        "folder": "torchaudio_24"
    },
    "Torch 2.4": {
        "URL": "https://pytorch.org/docs/2.4/",
        "folder": "torch_24"
    },
    "Torchvision 0.19": {
        "URL": "https://pytorch.org/vision/0.19/",
        "folder": "torchvision_019"
    },
    "Python 3.11": {
        "URL": "https://docs.python.org/3.11/",
        "folder": "python_311"
    },
    "LM Studio": {
        "URL": "https://lmstudio.ai/docs/",
        "folder": "lm_studio"
    },
    "PyInstaller 6.10.0": {
        "URL": "https://pyinstaller.org/en/v6.10.0/",
        "folder": "pyinstaller_6100"
    },
    "CuPy": {
        "URL": "https://docs.cupy.dev/en/stable/reference/",
        "folder": "cupy"
    },
    "AutoAWQ": {
        "URL": "https://casper-hansen.github.io/AutoAWQ/",
        "folder": "autoawq"
    },
    "Numexpr": {
        "URL": "https://numexpr.readthedocs.io/en/latest/",
        "folder": "numexpr"
    },
    "Dask": {
        "URL": "https://docs.dask.org/en/stable/",
        "folder": "dask"
    },
    "Transformers.js": {
        "URL": "https://huggingface.co/docs/transformers.js/",
        "folder": "transformers_js"
    },
    "NLTK": {
        "URL": "https://www.nltk.org/",
        "folder": "nltk"
    },
    "gTTS": {
        "URL": "https://gtts.readthedocs.io/en/latest/",
        "folder": "gtts"
    },
    "Loguru": {
        "URL": "https://loguru.readthedocs.io/en/stable/",
        "folder": "loguru"
    },
    "Pygments": {
        "URL": "https://pygments.org/docs/",
        "folder": "pygments"
    },
    "Soxr": {
        "URL": "https://python-soxr.readthedocs.io/en/stable/",
        "folder": "soxr"
    },
    "Librosa": {
        "URL": "https://librosa.org/doc/latest/",
        "folder": "librosa"
    },
    "ONNX Runtime": {
        "URL": "https://onnxruntime.ai/docs/api/python/",
        "folder": "onnx_runtime"
    },
    "ONNX": {
        "URL": "https://onnx.ai/onnx/",
        "folder": "onnx"
    },
    "Jinja 3.1": {
        "URL": "https://jinja.palletsprojects.com/en/3.1.x/",
        "folder": "jinja_31"
    },
    "Torchmetrics": {
        "URL": "https://lightning.ai/docs/torchmetrics/stable/",
        "folder": "torchmetrics"
    },
    "PyTorch Lightning": {
        "URL": "https://lightning.ai/docs/pytorch/stable/",
        "folder": "pytorch_lightning"
    },
    "Matplotlib": {
        "URL": "https://matplotlib.org/stable/",
        "folder": "matplotlib"
    },
    "llama-cpp-python": {
        "URL": "https://llama-cpp-python.readthedocs.io/en/stable/",
        "folder": "llama_cpp_python"
    },
    "TensorRT-LLM": {
        "URL": "https://nvidia.github.io/TensorRT-LLM/",
        "folder": "tensorrt_llm"
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
    "Beautiful Soup 4": {
        "URL": "https://beautiful-soup-4.readthedocs.io/en/latest/",
        "folder": "beautiful_soup_4"
    },
    "CustomTkinter": {
        "URL": "https://customtkinter.tomschimansky.com/documentation/",
        "folder": "customtkinter"
    },
    "Rust UV": {
        "URL": "https://docs.astral.sh/uv/",
        "folder": "uv"
    },
    "xlrd": {
        "URL": "https://xlrd.readthedocs.io/en/latest/",
        "folder": "xlrd"
    },
    "xFormers": {
        "URL": "https://facebookresearch.github.io/xformers/",
        "folder": "xformers"
    },
    "Wrapt": {
        "URL": "https://wrapt.readthedocs.io/en/master/",
        "folder": "wrapt"
    },
    "urllib3": {
        "URL": "https://urllib3.readthedocs.io/en/stable/",
        "folder": "urllib3"
    },
    "Timm 0.9.16": {
        "URL": "https://huggingface.co/docs/timm/v0.9.16/en/",
        "folder": "timm_0916"
    },
    "SQLAlchemy 20": {
        "URL": "https://docs.sqlalchemy.org/en/20/",
        "folder": "sqlalchemy_20"
    },
    "SpeechBrain 0.5.15": {
        "URL": "https://speechbrain.readthedocs.io/en/v0.5.15/",
        "folder": "speechbrain_0515"
    },
    "Soupsieve": {
        "URL": "https://facelessuser.github.io/soupsieve/",
        "folder": "soupsieve"
    },
    "Six": {
        "URL": "https://six.readthedocs.io/",
        "folder": "six"
    },
    "SciPy 1.14.1": {
        "URL": "https://docs.scipy.org/doc/scipy-1.14.1/",
        "folder": "scipy_1141"
    },
    "scikit-learn": {
        "URL": "https://scikit-learn.org/stable/",
        "folder": "scikit_learn"
    },
    "Rich": {
        "URL": "https://rich.readthedocs.io/en/latest/",
        "folder": "rich"
    },
    "RapidFuzz": {
        "URL": "https://rapidfuzz.github.io/RapidFuzz/",
        "folder": "rapidfuzz"
    },
    "PyYAML": {
        "URL": "https://pyyaml.org/wiki/PyYAMLDocumentation",
        "folder": "pyyaml"
    },
    "python-docx": {
        "URL": "https://python-docx.readthedocs.io/en/latest/",
        "folder": "python_docx"
    },
    "PyMuPDF": {
        "URL": "https://pymupdf.readthedocs.io/en/latest/",
        "folder": "pymupdf"
    },
    "PyPDF 5.0.1": {
        "URL": "https://pypdf.readthedocs.io/en/5.0.1/",
        "folder": "pypdf_501"
    },
    "PyPDF 4.3.1": {
        "URL": "https://pypdf.readthedocs.io/en/4.3.1/",
        "folder": "pypdf_431"
    },
    "Pandoc": {
        "URL": "https://pandoc.org",
        "folder": "pandoc"
    },
    "marshmallow": {
        "URL": "https://marshmallow.readthedocs.io/en/stable/",
        "folder": "marshmallow"
    },
    "Protocol Buffers": {
        "URL": "https://protobuf.dev/",
        "folder": "protocol_buffers"
    },
    "platformdirs": {
        "URL": "https://platformdirs.readthedocs.io/en/stable/",
        "folder": "platformdirs"
    },
    "packaging": {
        "URL": "https://packaging.pypa.io/en/stable/",
        "folder": "packaging"
    },
    "OmegaConf 2.2": {
        "URL": "https://omegaconf.readthedocs.io/en/2.2_branch/",
        "folder": "omegaconf_22"
    },
    "multiprocess": {
        "URL": "https://multiprocess.readthedocs.io/en/latest/",
        "folder": "multiprocess"
    },
    "msg-parser": {
        "URL": "https://msg-parser.readthedocs.io/en/latest/",
        "folder": "msg_parser"
    },
    "mpmath": {
        "URL": "https://mpmath.org/doc/current/",
        "folder": "mpmath"
    },
    "openpyxl": {
        "URL": "https://openpyxl.readthedocs.io/en/stable/",
        "folder": "openpyxl"
    },
    "Numba 0.60.0": {
        "URL": "https://numba.readthedocs.io/en/0.60.0/",
        "folder": "numba_0600"
    },
    "NetworkX": {
        "URL": "https://networkx.org/documentation/stable/",
        "folder": "networkx"
    },
    "natsort 8.4.0": {
        "URL": "https://natsort.readthedocs.io/en/8.4.0/",
        "folder": "natsort_840"
    },
    "dill": {
        "URL": "https://dill.readthedocs.io/en/latest/",
        "folder": "dill"
    },
    "coloredlogs": {
        "URL": "https://coloredlogs.readthedocs.io/en/latest/",
        "folder": "coloredlogs"
    },
    "Click 8.1": {
        "URL": "https://click.palletsprojects.com/en/8.1.x/",
        "folder": "click_81"
    },
    "charset-normalizer 3.3.2": {
        "URL": "https://charset-normalizer.readthedocs.io/en/3.3.2/",
        "folder": "charset_normalizer_332"
    },
    "aiohttp 3.9.5": {
        "URL": "https://docs.aiohttp.org/en/v3.9.5/",
        "folder": "aiohttp_395"
    },
    "NumPy 1.26": {
        "URL": "https://numpy.org/doc/1.26/",
        "folder": "numpy_126"
    },
    "CTranslate2": {
        "URL": "https://opennmt.net/CTranslate2/",
        "folder": "ctranslate2"
    },
    "pandas": {
        "URL": "https://pandas.pydata.org/docs/",
        "folder": "pandas"
    },
    "tqdm": {
        "URL": "https://tqdm.github.io",
        "folder": "tqdm"
    },
    "Requests": {
        "URL": "https://requests.readthedocs.io/en/latest/",
        "folder": "requests"
    },
    "Pillow": {
        "URL": "https://pillow.readthedocs.io/en/stable/",
        "folder": "pillow"
    },
    "bitsandbytes 0.44.1": {
        "URL": "https://huggingface.co/docs/bitsandbytes/v0.44.1/en/",
        "folder": "bitsandbytes_0441"
    },
    "bitsandbytes 0.43.3": {
        "URL": "https://huggingface.co/docs/bitsandbytes/v0.43.3/en/",
        "folder": "bitsandbytes_0433"
    },
    "chardet": {
        "URL": "https://chardet.readthedocs.io/en/latest/",
        "folder": "chardet"
    },
    "Transformers 4.43.4": {
        "URL": "https://huggingface.co/docs/transformers/v4.43.4/en/",
        "folder": "transformers_4434"
    },
    "Transformers 4.44.2": {
        "URL": "https://huggingface.co/docs/transformers/v4.44.2/en/",
        "folder": "transformers_4442"
    },
    "SoundFile": {
        "URL": "https://python-soundfile.readthedocs.io/en/latest/",
        "folder": "soundfile"
    },
    "sounddevice 0.4.6": {
        "URL": "https://python-sounddevice.readthedocs.io/en/0.4.6/",
        "folder": "sounddevice_046"
    },
    "Sentence-Transformers": {
        "URL": "https://www.sbert.net/docs",
        "folder": "sentence_transformers"
    },
    "PyAV": {
        "URL": "https://pyav.basswood-io.com/docs/stable/",
        "folder": "pyav"
    },
    "Qt for Python 6": {
        "URL": "https://doc.qt.io/qtforpython-6/",
        "folder": "qt_for_python_6"
    },
    "jiwer": {
        "URL": "https://jitsi.github.io/jiwer/",
        "folder": "jiwer"
    },
    "SymPy": {
        "URL": "https://docs.sympy.org/latest/",
        "folder": "sympy"
    },
    "Torchvision 0.18": {
        "URL": "https://pytorch.org/vision/0.18/",
        "folder": "torchvision_018"
    },
    "Torchvision 0.17": {
        "URL": "https://pytorch.org/vision/0.17/",
        "folder": "torchvision_017"
    },
    "Torchaudio 2.3": {
        "URL": "https://pytorch.org/audio/2.3.0/",
        "folder": "torchaudio_23"
    },
    "Torchaudio 2.2": {
        "URL": "https://pytorch.org/audio/2.2.0/",
        "folder": "torchaudio_22"
    },
    "Torch 2.3": {
        "URL": "https://pytorch.org/docs/2.3/",
        "folder": "torch_23"
    },
    "Torch 2.2": {
        "URL": "https://pytorch.org/docs/2.2/",
        "folder": "torch_22"
    }
}

class CustomButtonStyles:
    SUBDUED_RED = "#320A0A"
    LIGHT_GREY = "#C8C8C8"
    SUBDUED_BLUE = "#0A0A32"
    SUBDUED_GREEN = "#0A320A"
    SUBDUED_YELLOW = "#32320A"
    SUBDUED_PURPLE = "#320A32"
    SUBDUED_ORANGE = "#321E0A"
    SUBDUED_TEAL = "#0A3232"
    SUBDUED_PINK = "#320A1E"
    SUBDUED_BROWN = "#2B1E0A"
    
    RED_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_RED};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #4B0F0F;
        }}
        QPushButton:pressed {{
            background-color: #290909;
        }}
        QPushButton:disabled {{
            background-color: #7D1919;
            color: #969696;
        }}
    """
    
    BLUE_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_BLUE};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #0F0F4B;
        }}
        QPushButton:pressed {{
            background-color: #09092B;
        }}
        QPushButton:disabled {{
            background-color: #19197D;
            color: #969696;
        }}
    """
    
    GREEN_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_GREEN};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #0F4B0F;
        }}
        QPushButton:pressed {{
            background-color: #092909;
        }}
        QPushButton:disabled {{
            background-color: #197D19;
            color: #969696;
        }}
    """

    YELLOW_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_YELLOW};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #4B4B0F;
        }}
        QPushButton:pressed {{
            background-color: #292909;
        }}
        QPushButton:disabled {{
            background-color: #7D7D19;
            color: #969696;
        }}
    """

    PURPLE_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_PURPLE};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #4B0F4B;
        }}
        QPushButton:pressed {{
            background-color: #290929;
        }}
        QPushButton:disabled {{
            background-color: #7D197D;
            color: #969696;
        }}
    """

    ORANGE_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_ORANGE};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #4B2D0F;
        }}
        QPushButton:pressed {{
            background-color: #291909;
        }}
        QPushButton:disabled {{
            background-color: #7D5A19;
            color: #969696;
        }}
    """

    TEAL_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_TEAL};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #0F4B4B;
        }}
        QPushButton:pressed {{
            background-color: #092929;
        }}
        QPushButton:disabled {{
            background-color: #197D7D;
            color: #969696;
        }}
    """

    BROWN_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUBDUED_BROWN};
            color: {LIGHT_GREY};
            padding: 5px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #412D0F;
        }}
        QPushButton:pressed {{
            background-color: #231909;
        }}
        QPushButton:disabled {{
            background-color: #6B5A19;
            color: #969696;
        }}
    """

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
