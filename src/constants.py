system_message = """You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. Here are one or more contexts to solely base your answer off of. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."""

CHUNKS_ONLY_TOOLTIP = "Only return relevant chunks without connecting to the LLM. Extremely useful to test the chunk size/overlap settings."

SPEAK_RESPONSE_TOOLTIP = "Only click this after the LLM's entire response is received otherwise your computer might explode."

DOWNLOAD_EMBEDDING_MODEL_TOOLTIP = "Remember, wait until downloading is complete!"

VECTOR_MODELS = {
    'Alibaba-NLP': [
        {
            'name': 'gte-base-en-v1.5',
            'dimensions': 768,
            'max_sequence': 8192,
            'size_mb': 547,
            'repo_id': 'Alibaba-NLP/gte-base-en-v1.5',
            'cache_dir': 'Alibaba-NLP--gte-base-en-v1.5',
            'type': 'vector'
        },
        {
            'name': 'gte-large-en-v1.5',
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
        # {
            # 'name': 'sentence-t5-xxl',
            # 'dimensions': 768,
            # 'max_sequence': 256,
            # 'size_mb': 9230,
            # 'repo_id': 'sentence-transformers/sentence-t5-xxl',
            # 'cache_dir': 'sentence-transformers--sentence-t5-xxl',
            # 'type': 'vector'
        # },
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
    'dunzhang': [
        {
            'name': 'stella_en_1.5B_v5',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 547,
            'repo_id': 'dunzhang/stella_en_1.5B_v5',
            'cache_dir': 'dunzhang--stella_en_1.5B_v5',
            'type': 'vector'
        },
    ],

}


CHAT_MODELS = {
    # 'Dolphin-Qwen 2 - .5b': {
        # 'model': 'Dolphin-Qwen 2 - .5b',
        # 'repo_id': 'cognitivecomputations/dolphin-2.9.3-qwen2-0.5b',
        # 'cache_dir': 'cognitivecomputations--dolphin-2.9.3-qwen2-0.5b',
        # 'tokens_per_second': 67.66,
        # 'context_length': 16384,
        # 'avg_vram_usage': '2.4 GB',
        # 'function': 'Dolphin_Qwen2_0_5b'
    # },
    'Zephyr - 1.6b': {
        'model': 'Zephyr - 1.6b',
        'repo_id': 'stabilityai/stablelm-2-zephyr-1_6b',
        'cache_dir': 'stabilityai--stablelm-2-zephyr-1_6b',
        'tokens_per_second': 74,
        'context_length': 4096,
        'avg_vram_usage': '2.5 GB',
        'function': 'Zephyr_1_6B'
    },
    # 'Internlm2 - 1.8b': {
        # 'model': 'Internlm2 - 1.8b',
        # 'repo_id': 'internlm/internlm2-chat-1_8b',
        # 'cache_dir': 'internlm--internlm2-chat-1_8b',
        # 'tokens_per_second': 55.51,
        # 'context_length': 32768,
        # 'avg_vram_usage': '2.8 GB',
        # 'function': 'InternLM2_1_8b'
    # },
    'Zephyr - 3b': {
        'model': 'Zephyr - 3b',
        'repo_id': 'stabilityai/stablelm-zephyr-3b',
        'cache_dir': 'stabilityai--stablelm-zephyr-3b',
        'tokens_per_second': 57,
        'context_length': 4096,
        'avg_vram_usage': '2.9 GB',
        'function': 'Zephyr_3B'
    },
    'Qwen 1.5 - 1.8B': {
        'model': 'Qwen 1.5 - 1.8B',
        'repo_id': 'Qwen/Qwen1.5-1.8B-Chat',
        'cache_dir': 'Qwen--Qwen1.5-1.8B-Chat',
        'tokens_per_second': 65,
        'context_length': 32768,
        'avg_vram_usage': '3.7 GB',
        'function': 'Qwen1_5_1_8b'
    },
    'Dolphin-Qwen 2 - 1.5b': {
        'model': 'Dolphin-Qwen 2 - 1.5b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.3-qwen2-1.5b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.3-qwen2-1.5b',
        'tokens_per_second': 58.07,
        'context_length': 16384,
        'avg_vram_usage': '4.2 GB',
        'function': 'Dolphin_Qwen2_1_5b'
    },
    'Qwen 1.5 - 4B': {
        'model': 'Qwen 1.5 - 4B',
        'repo_id': 'Qwen/Qwen1.5-4B-Chat',
        'cache_dir': 'Qwen--Qwen1.5-4B-Chat',
        'tokens_per_second': 41.57,
        'context_length': 4096,
        'avg_vram_usage': '5.4 GB',
        'function': 'Qwen1_5_4b'
    },
    'Llama 2 - 7b': {
        'model': 'Llama 2 - 7b',
        'repo_id': 'meta-llama/Llama-2-7b-chat-hf',
        'cache_dir': 'meta-llama--Llama-2-7b-chat-hf',
        'tokens_per_second': 45,
        'context_length': 4096,
        'avg_vram_usage': '5.8 GB',
        'function': 'Llama2_7b'
    },
    'Orca 2 - 7b': {
        'model': 'Orca 2 - 7b',
        'repo_id': 'microsoft/Orca-2-7b',
        'cache_dir': 'microsoft--Orca-2-7b',
        'tokens_per_second': 47.10,
        'context_length': 4096,
        'avg_vram_usage': '5.9 GB',
        'function': 'Orca2_7b'
    },
    'H2O Danube3 4B': {
        'model': 'H2O Danube3 4B',
        'repo_id': 'h2oai/h2o-danube3-4b-chat',
        'cache_dir': 'h2oai--h2o-danube3-4b-chat',
        'tokens_per_second': None,
        'context_length': 8192,
        'avg_vram_usage': None,
        'function': 'H2O_Danube3_4B'
    },
    # GATED
    # 'Mistral 0.3 - 7b': {
        # 'model': 'Mistral 0.3 - 7b',
        # 'repo_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        # 'cache_dir': 'mistralai--Mistral-7B-Instruct-v0.3',
        # 'tokens_per_second': 50.40,
        # 'context_length': 4096,
        # 'avg_vram_usage': '5.7 GB',
        # 'function': 'Mistral7B'
    # },
    'Neural-Chat - 7b': {
        'model': 'Neural-Chat - 7b',
        'repo_id': 'Intel/neural-chat-7b-v3-3',
        'cache_dir': 'Intel--neural-chat-7b-v3-3',
        'tokens_per_second': 46,
        'context_length': 4096,
        'avg_vram_usage': '5.8 GB',
        'function': 'Neural_Chat_7b'
    },
    'Internlm2 - 7b': {
        'model': 'Internlm2 - 7b',
        'repo_id': 'internlm/internlm2-chat-7b',
        'cache_dir': 'internlm--internlm2-chat-7b',
        'tokens_per_second': 36.83,
        'context_length': 32768,
        'avg_vram_usage': '6.7 GB',
        'function': 'InternLM2_7b'
    },
    'Internlm2_5 - 7b': {
        'model': 'Internlm2_5 - 7b',
        'repo_id': 'internlm/internlm2_5-7b-chat',
        'cache_dir': 'internlm--internlm2_5-7b-chat',
        'tokens_per_second': 35.12,
        'context_length': 32768,
        'avg_vram_usage': '6.8 GB',
        'function': 'InternLM2_5_7b'
    },
    'Dolphin-Llama 3 - 8b': {
        'model': 'Dolphin-Llama 3 - 8b',
        'repo_id': 'cognitivecomputations/dolphin-2.9-llama3-8b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9-llama3-8b',
        'tokens_per_second': 49.77,
        'context_length': 8192,
        'avg_vram_usage': '7.1 GB',
        'function': 'Dolphin_Llama3_8B'
    },
    'Dolphin-Llama 3.1 - 8b': {
        'model': 'Dolphin-Llama 3.1 - 8b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.4-llama3.1-8b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.4-llama3.1-8b',
        'tokens_per_second': 50.33,
        'context_length': 8192,
        'avg_vram_usage': '7.1 GB',
        'function': 'Dolphin_Llama3_1_8B'
    },
    'Dolphin-Yi 1.5 - 9b': {
        'model': 'Dolphin-Yi 1.5 - 9b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.1-yi-1.5-9b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.1-yi-1.5-9b',
        'tokens_per_second': 30.85,
        'context_length': 4096,
        'avg_vram_usage': '7.2 GB',
        'function': 'Dolphin_Yi_1_5_9b'
    },
    'Orca 2 - 13b': {
        'model': 'Orca 2 - 13b',
        'repo_id': 'microsoft/Orca-2-13b',
        'cache_dir': 'microsoft--Orca-2-13b',
        'tokens_per_second': 36.11,
        'context_length': 4096,
        'avg_vram_usage': '9.9 GB',
        'function': 'Orca2_13b'
    },
    'Dolphin-Qwen 2 - 7b': {
        'model': 'Dolphin-Qwen 2 - 7b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.2-qwen2-7b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.2-qwen2-7b',
        'tokens_per_second': 52,
        'context_length': 16384,
        'avg_vram_usage': '9.2 GB',
        'function': 'Dolphin_Qwen2_7b'
    },
    'Dolphin-Phi 3 - Medium': {
        'model': 'Dolphin-Phi 3 - Medium',
        'repo_id': 'cognitivecomputations/dolphin-2.9.2-Phi-3-Medium',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.2-Phi-3-Medium',
        'tokens_per_second': 40,
        'context_length': 4096,
        'avg_vram_usage': '9.3 GB',
        'function': 'Dolphin_Phi3_Medium'
    },
    'SOLAR - 10.7b': {
        'model': 'SOLAR - 10.7b',
        'repo_id': 'upstage/SOLAR-10.7B-Instruct-v1.0',
        'cache_dir': 'upstage--SOLAR-10.7B-Instruct-v1.0',
        'tokens_per_second': 28,
        'context_length': 4096,
        'avg_vram_usage': '9.3 GB',
        'function': 'SOLAR_10_7B'
    },
    # GATED
    # 'Gemma 2 9B': {
        # 'model': 'Gemma 2 9B',
        # 'repo_id': 'google/gemma-2b-it',
        # 'cache_dir': 'google--gemma-2b-it',
        # 'tokens_per_second': None,
        # 'context_length': 8192,
        # 'avg_vram_usage': None,
        # 'function': 'Gemma_2_9B'
    # },
    'Llama 2 - 13b': {
        'model': 'Llama 2 - 13b',
        'repo_id': 'meta-llama/Llama-2-13b-chat-hf',
        'cache_dir': 'meta-llama--Llama-2-13b-chat-hf',
        'tokens_per_second': 36.80,
        'context_length': 4096,
        'avg_vram_usage': '10.0 GB',
        'function': 'Llama2_13b'
    },
    'Dolphin-Mistral-Nemo - 12b': {
        'model': 'Dolphin-Mistral-Nemo - 12b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.3-mistral-nemo-12b',
        'tokens_per_second': 35.86,
        'context_length': 8192,
        'avg_vram_usage': '10.0 GB',
        'function': 'Dolphin_Mistral_Nemo'
    },
    'Internlm2 - 20b': {
        'model': 'Internlm2 - 20b',
        'repo_id': 'internlm/internlm2-chat-20b',
        'cache_dir': 'internlm--internlm2-chat-20b',
        'tokens_per_second': 20.21,
        'context_length': 32768,
        'avg_vram_usage': '14.2 GB',
        'function': 'InternLM2_20b'
    },
}


VISION_MODELS = {
    'Florence-2-base': {
        'precision': 'autoselect',
        'size': '232m',
        'repo_id': 'microsoft/Florence-2-base',
        'cache_dir': 'microsoft--Florence-2-base',
        'requires_cuda': False,
        'avg_vram_usage': '2.6 GB',
        'tokens_per_second': 163.06
    },
    'Moondream2 - 1.9b': {
        'precision': 'float16',
        'size': '2b',
        'repo_id': 'vikhyatk/moondream2',
        'cache_dir': 'vikhyatk--moondream2',
        'requires_cuda': True,
        'avg_vram_usage': '4.6 GB',
        'tokens_per_second': 82.28
    },
    'Florence-2-large': {
        'precision': 'autoselect',
        'size': '770m',
        'repo_id': 'microsoft/Florence-2-large',
        'cache_dir': 'microsoft--Florence-2-large',
        'requires_cuda': False,
        'avg_vram_usage': '5.3 GB',
        'tokens_per_second': 113.32
    },
    'Phi-3-vision - 4.2b': {
        'precision': '4-bit',
        'size': '4.2b',
        'repo_id': 'microsoft/Phi-3-vision-128k-instruct',
        'cache_dir': 'microsoft--Phi-3-vision-128k-instruct',
        'requires_cuda': True,
        'avg_vram_usage': '5.4 GB',
        'tokens_per_second': 30.72
    },
    'Llava 1.5 - 7b': {
        'precision': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/llava-1.5-7b-hf',
        'cache_dir': 'llava-hf--llava-1.5-7b-hf',
        'requires_cuda': True,
        'avg_vram_usage': '5.8 GB',
        'tokens_per_second': 48.30
    },
    'Bakllava 1.5 - 7b': {
        'precision': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/bakLlava-v1-hf',
        'cache_dir': 'llava-hf--bakLlava-v1-hf',
        'requires_cuda': True,
        'avg_vram_usage': '5.9 GB',
        'tokens_per_second': 48.30
    },
    'Llava 1.6 Vicuna - 7b': {
        'precision': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/llava-v1.6-vicuna-7b-hf',
        'cache_dir': 'llava-hf--llava-v1.6-vicuna-7b-hf',
        'requires_cuda': True,
        'avg_vram_usage': '7.9 GB',
        'tokens_per_second': 56.33
    },
    'MiniCPM-V-2_6 - 8b': {
        'precision': '4-bit',
        'size': '8b',
        'repo_id': 'openbmb/MiniCPM-V-2_6-int4',
        'cache_dir': 'openbmb--MiniCPM-V-2_6-int4',
        'requires_cuda': True,
        'avg_vram_usage': '9.1 GB',
        'tokens_per_second': 16.40
    },
    'Llava 1.5 - 13b': {
        'precision': '4-bit',
        'size': '13b',
        'repo_id': 'llava-hf/llava-1.5-13b-hf',
        'cache_dir': 'llava-hf--llava-1.5-13b-hf',
        'requires_cuda': True,
        'avg_vram_usage': '9.8 GB',
        'tokens_per_second': 38.03
    },
    'falcon-vlm - 11b': {
        'precision': '4-bit',
        'size': '13b',
        'repo_id': 'tiiuae/falcon-11B-vlm',
        'cache_dir': 'tiiuae--falcon-11B-vlm',
        'requires_cuda': True,
        'avg_vram_usage': '12.8 GB',
        'tokens_per_second': 18.36
    },
    'Llava 1.6 Vicuna - 13b': {
        'precision': '4-bit',
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
    ".html": "UnstructuredHTMLLoader",
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

graphics_cards = {
    "GeForce RTX 3050 Mobile/Laptop": {
        "Size (GB)": 4,
        "CUDA Cores": 2048,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 3050": {
        "Size (GB)": 8,
        "CUDA Cores": 2304,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4050 Mobile/Laptop": {
        "Size (GB)": 6,
        "CUDA Cores": 2560,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3050 Ti Mobile/Laptop": {
        "Size (GB)": 4,
        "CUDA Cores": 2560,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4060": {
        "Size (GB)": 8,
        "CUDA Cores": 3072,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3060": {
        "Size (GB)": 12,
        "CUDA Cores": 3584,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 3060 Mobile/Laptop": {
        "Size (GB)": 6,
        "CUDA Cores": 3840,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4060 Ti": {
        "Size (GB)": 16,
        "CUDA Cores": 4352,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 4070 Mobile/Laptop": {
        "Size (GB)": 8,
        "CUDA Cores": 4608,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3060 Ti": {
        "Size (GB)": 8,
        "CUDA Cores": 4864,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 3070 Mobile/Laptop": {
        "Size (GB)": 8,
        "CUDA Cores": 5120,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 3070": {
        "Size (GB)": 8,
        "CUDA Cores": 5888,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4070": {
        "Size (GB)": 12,
        "CUDA Cores": 5888,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3070 Ti": {
        "Size (GB)": 8,
        "CUDA Cores": 6144,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 3070 Ti Mobile/Laptop": {
        "Size (GB)": "8-16",
        "CUDA Cores": 6144,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4070 Super": {
        "Size (GB)": 12,
        "CUDA Cores": 7168,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 4080 Mobile/Laptop": {
        "Size (GB)": 12,
        "CUDA Cores": 7424,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3080 Ti Mobile/Laptop": {
        "Size (GB)": 16,
        "CUDA Cores": 7424,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4070 Ti": {
        "Size (GB)": 12,
        "CUDA Cores": 7680,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 4080": {
        "Size (GB)": 12,
        "CUDA Cores": 7680,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3080": {
        "Size (GB)": 10,
        "CUDA Cores": 8704,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4070 Ti Super": {
        "Size (GB)": 16,
        "CUDA Cores": 8448,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3080 Ti": {
        "Size (GB)": 12,
        "CUDA Cores": 8960,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4080": {
        "Size (GB)": 16,
        "CUDA Cores": 9728,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 4090 Mobile/Laptop": {
        "Size (GB)": 16,
        "CUDA Cores": 9728,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 4080 Super": {
        "Size (GB)": 16,
        "CUDA Cores": 10240,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 3090": {
        "Size (GB)": 24,
        "CUDA Cores": 10496,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 3090 Ti": {
        "Size (GB)": 24,
        "CUDA Cores": 10752,
        "Architecture": "Ampere",
        "CUDA Compute": 8.6
    },
    "GeForce RTX 4090 D": {
        "Size (GB)": 24,
        "CUDA Cores": 14592,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    },
    "GeForce RTX 4090": {
        "Size (GB)": 24,
        "CUDA Cores": 16384,
        "Architecture": "Ada Lovelace",
        "CUDA Compute": 8.9
    }
}
