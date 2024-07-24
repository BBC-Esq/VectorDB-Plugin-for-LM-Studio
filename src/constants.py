VECTOR_MODELS = {
    # 'Alibaba-NLP': [
        # {
            # 'name': 'gte-base-en-v1.5',
            # 'dimensions': 768,
            # 'max_sequence': 8192,
            # 'size_mb': 547,
            # 'repo_id': 'Alibaba-NLP/gte-base-en-v1.5',
            # 'cache_dir': 'Alibaba-NLP--gte-base-en-v1.5',
            # 'type': 'vector'
        # },
        # {
            # 'name': 'gte-large-en-v1.5',
            # 'dimensions': 1024,
            # 'max_sequence': 8192,
            # 'size_mb': 1740,
            # 'repo_id': 'Alibaba-NLP/gte-large-en-v1.5',
            # 'cache_dir': 'Alibaba-NLP--gte-large-en-v1.5',
            # 'type': 'vector'
        # },
    # ],
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
        {
            'name': 'sentence-t5-xxl',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 9230,
            'repo_id': 'sentence-transformers/sentence-t5-xxl',
            'cache_dir': 'sentence-transformers--sentence-t5-xxl',
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

    #{'name': 'msmarco-distilbert-cos-v5', 'dimensions': 768, 'max_sequence': 512, 'size_mb': 265},
    #'jinaai/jina-embedding-s-en-v1': {'dimensions': 512, 'max_sequence': 512, 'size_mb': 141},
    #'jinaai/jina-embedding-b-en-v1': {'dimensions': 768, 'max_sequence': 512, 'size_mb': 439},
    #'jinaai/jina-embedding-l-en-v1': {'dimensions': 1024, 'max_sequence': 512, 'size_mb': 1340},
    #'nomic-ai/nomic-embed-text-v1.5': {'dimensions': 768, 'max_sequence': 8192, 'size_mb': 547},
    # 'Alibaba-NLP/gte-large-en-v1.5': {'dimensions': 1024, 'max_sequence': 8192, 'size_mb': 1740},
    # 'Alibaba-NLP/gte-base-en-v1.5': {'dimensions': 768, 'max_sequence': 8192, 'size_mb': 547},


CHAT_MODELS = {
    'Qwen 2 - 0.5b': {
        'model': 'Qwen 2 - 0.5b',
        'repo_id': 'Qwen/Qwen2-0.5B-Instruct',
        'cache_dir': 'Qwen--Qwen2-0.5B-Instruct',
        'tokens_per_second': 66.64,
        'context_length': 4096,
        'avg_vram_usage': '1.9 GB',
        'function': 'Qwen2_0_5b'
    },
    'Qwen 1.5 - 0.5b': {
        'model': 'Qwen 1.5 - 0.5b',
        'repo_id': 'Qwen/Qwen1.5-0.5B-Chat',
        'cache_dir': 'Qwen--Qwen1.5-0.5B-Chat',
        'tokens_per_second': 60,
        'context_length': 4096,
        'avg_vram_usage': '1.9 GB',
        'function': 'Qwen1_5_0_5'
    },
    'Dolphin-Qwen 2 - .5b': {
        'model': 'Dolphin-Qwen 2 - .5b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.3-qwen2-0.5b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.3-qwen2-0.5b',
        'tokens_per_second': 67.66,
        'context_length': 4096,
        'avg_vram_usage': '2.4 GB',
        'function': 'Dolphin_Qwen2_0_5b'
    },
    'Zephyr - 1.6b': {
        'model': 'Zephyr - 1.6b',
        'repo_id': 'stabilityai/stablelm-2-zephyr-1_6b',
        'cache_dir': 'stabilityai--stablelm-2-zephyr-1_6b',
        'tokens_per_second': 74,
        'context_length': 4096,
        'avg_vram_usage': '2.5 GB',
        'function': 'Zephyr_1_6B'
    },
    'Internlm2 - 1.8b': {
        'model': 'Internlm2 - 1.8b',
        'repo_id': 'internlm/internlm2-chat-1_8b',
        'cache_dir': 'internlm--internlm2-chat-1_8b',
        'tokens_per_second': 55.51,
        'context_length': 4096,
        'avg_vram_usage': '2.8 GB',
        'function': 'InternLM2_1_8b'
    },
    'Zephyr - 3b': {
        'model': 'Zephyr - 3b',
        'repo_id': 'stabilityai/stablelm-zephyr-3b',
        'cache_dir': 'stabilityai--stablelm-zephyr-3b',
        'tokens_per_second': 57,
        'context_length': 4096,
        'avg_vram_usage': '2.9 GB',
        'function': 'Zephyr_3B'
    },
    'Qwen 2 - 1.5B': {
        'model': 'Qwen 2 - 1.5B',
        'repo_id': 'Qwen/Qwen2-1.5B-Instruct',
        'cache_dir': 'Qwen--Qwen2-1.5B-Instruct',
        'tokens_per_second': 53.62,
        'context_length': 4096,
        'avg_vram_usage': '3.0 GB',
        'function': 'Qwen2_1_5b'
    },
    'Qwen 1.5 - 1.8B': {
        'model': 'Qwen 1.5 - 1.8B',
        'repo_id': 'Qwen/Qwen1.5-1.8B-Chat',
        'cache_dir': 'Qwen--Qwen1.5-1.8B-Chat',
        'tokens_per_second': 65,
        'context_length': 4096,
        'avg_vram_usage': '3.7 GB',
        'function': 'Qwen1_5_1_8b'
    },
    'Phi-3 Mini 4k - 3.8B': {
        'model': 'Phi-3 Mini 4k - 3.8B',
        'repo_id': 'microsoft/Phi-3-mini-4k-instruct',
        'cache_dir': 'microsoft--Phi-3-mini-4k-instruct',
        'tokens_per_second': 50,
        'context_length': 4096,
        'avg_vram_usage': '4.0 GB',
        'function': 'Phi3_mini_4k'
    },
    'Dolphin-Qwen 2 - 1.5b': {
        'model': 'Dolphin-Qwen 2 - 1.5b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.3-qwen2-1.5b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.3-qwen2-1.5b',
        'tokens_per_second': 58.07,
        'context_length': 4096,
        'avg_vram_usage': '4.2 GB',
        'function': 'Dolphin_Qwen2_1_5b'
    },
    # 'Yi 1.5 - 6B': {
        # 'model': 'Yi 1.5 - 6B',
        # 'repo_id': '01-ai/Yi-1.5-6B-Chat',
        # 'cache_dir': '01-ai--Yi-1.5-6B-Chat',
        # 'tokens_per_second': 45.10,
        # 'context_length': 4096,
        # 'avg_vram_usage': '5.2 GB',
        # 'function': 'Yi_6B'
    # },
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
    'Mistral 0.3 - 7b': {
        'model': 'Mistral 0.3 - 7b',
        'repo_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'cache_dir': 'mistralai--Mistral-7B-Instruct-v0.3',
        'tokens_per_second': 50.40,
        'context_length': 4096,
        'avg_vram_usage': '5.7 GB',
        'function': 'Mistral7B'
    },
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
        'context_length': 4096,
        'avg_vram_usage': '6.7 GB',
        'function': 'InternLM2_7b'
    },
    'Internlm2_5 - 7b': {
        'model': 'Internlm2_5 - 7b',
        'repo_id': 'internlm/internlm2_5-7b-chat',
        'cache_dir': 'internlm--internlm2_5-7b-chat',
        'tokens_per_second': 35.12,
        'context_length': 4096,
        'avg_vram_usage': '6.8 GB',
        'function': 'InternLM2_5_7b'
    },
    # 'Yi 1.5 - 9B': {
        # 'model': 'Yi 1.5 - 9B',
        # 'repo_id': '01-ai/Yi-1.5-9B-Chat',
        # 'cache_dir': '01-ai--Yi-1.5-6B-Chat',
        # 'tokens_per_second': 45.10,
        # 'context_length': 4096,
        # 'avg_vram_usage': '7.0 GB',
        # 'function': 'Yi_9B'
    # },
    'Llama 3 - 8b': {
        'model': 'Llama 3 - 8b',
        'repo_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'cache_dir': 'meta-llama--Meta-Llama-3-8B-Instruct',
        'tokens_per_second': 44.48,
        'context_length': 4096,
        'avg_vram_usage': '7.1 GB',
        'function': 'Llama3_8B'
    },
    'Dolphin-Llama 3 - 8b': {
        'model': 'Dolphin-Llama 3 - 8b',
        'repo_id': 'cognitivecomputations/dolphin-2.9-llama3-8b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9-llama3-8b',
        'tokens_per_second': 41,
        'context_length': 4096,
        'avg_vram_usage': '7.1 GB',
        'function': 'Dolphin_Llama3_8B_Instruct'
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
    'Qwen 2 - 7B': {
        'model': 'Qwen 2 - 7B',
        'repo_id': 'Qwen/Qwen2-7B-Instruct',
        'cache_dir': 'Qwen--Qwen2-7B-Instruct',
        'tokens_per_second': 54.10,
        'context_length': 4096,
        'avg_vram_usage': '8.0 GB',
        'function': 'Qwen2_7b'
    },
    # 'Nous-Llama 2 - 13b': {
        # 'model': 'Nous-Llama 2 - 13b',
        # 'repo_id': 'NousResearch/Nous-Hermes-Llama2-13b',
        # 'cache_dir': 'NousResearch--Nous-Hermes-Llama2-13b',
        # 'tokens_per_second': 38.29,
        # 'context_length': 4096,
        # 'avg_vram_usage': '9.9 GB',
        # 'function': 'Nous_Llama2_13b'
    # },
    'Orca 2 - 13b': {
        'model': 'Orca 2 - 13b',
        'repo_id': 'microsoft/Orca-2-13b',
        'cache_dir': 'microsoft--Orca-2-13b',
        'tokens_per_second': 36.11,
        'context_length': 4096,
        'avg_vram_usage': '9.9 GB',
        'function': 'Orca2_13b'
    },
    # 'Phi-3 Medium 4k - 14b': {
        # 'model': 'Phi-3 Medium 4k - 14b',
        # 'repo_id': 'microsoft/Phi-3-medium-4k-instruct',
        # 'cache_dir': 'microsoft--Phi-3-medium-4k-instruct',
        # 'tokens_per_second': 34.60,
        # 'context_length': 4096,
        # 'avg_vram_usage': '9.8 GB',
        # 'function': 'Phi3_medium_4k'
    # },
    'Dolphin-Qwen 2 - 7b': {
        'model': 'Dolphin-Qwen 2 - 7b',
        'repo_id': 'cognitivecomputations/dolphin-2.9.2-qwen2-7b',
        'cache_dir': 'cognitivecomputations--dolphin-2.9.2-qwen2-7b',
        'tokens_per_second': 52,
        'context_length': 4096,
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
    'Stablelm 2 - 12b': {
        'model': 'Stablelm 2 - 12b',
        'repo_id': 'stabilityai/stablelm-2-12b-chat',
        'cache_dir': 'stabilityai--stablelm-2-12b-chat',
        'tokens_per_second': 28,
        'context_length': 4096,
        'avg_vram_usage': '11.3 GB',
        'function': 'Stablelm_2_12b'
    },
    'Internlm2 - 20b': {
        'model': 'Internlm2 - 20b',
        'repo_id': 'internlm/internlm2-chat-20b',
        'cache_dir': 'internlm--internlm2-chat-20b',
        'tokens_per_second': 20.21,
        'context_length': 4096,
        'avg_vram_usage': '14.2 GB',
        'function': 'InternLM2_20b'
    }
}


VISION_MODELS = {
    'Florence-2-base': {
        'precision': 'autoselect',
        'size': '232m',
        'repo_id': 'microsoft/Florence-2-base',
        'cache_dir': 'microsoft--Florence-2-base',
        'requires_cuda': False
    },
    'Florence-2-large': {
        'precision': 'autoselect',
        'size': '770m',
        'repo_id': 'microsoft/Florence-2-large',
        'cache_dir': 'microsoft--Florence-2-large',
        'requires_cuda': False
    },
    'Moondream2': {
        'precision': 'float16',
        'size': '2b',
        'repo_id': 'vikhyatk/moondream2',
        'cache_dir': 'vikhyatk--moondream2',
        'requires_cuda': True
    },
    'Phi-3-vision-128k-instruct': {
        'precision': '4-bit',
        'size': '4.2b',
        'repo_id': 'microsoft/Phi-3-vision-128k-instruct',
        'cache_dir': 'microsoft--Phi-3-vision-128k-instruct',
        'requires_cuda': True
    },
    'llava 1.5 - 7b': {
        'precision': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/llava-1.5-7b-hf',
        'cache_dir': 'llava-hf--llava-1.5-7b-hf',
        'requires_cuda': True
    },
    'bakllava 1.5 - 7b': {
        'precision': '4-bit',
        'size': '7b',
        'repo_id': 'llava-hf/bakLlava-v1-hf',
        'cache_dir': 'llava-hf--bakLlava-v1-hf',
        'requires_cuda': True
    },
    'llava 1.5 - 13b': {
        'precision': '4-bit',
        'size': '13b',
        'repo_id': 'llava-hf/llava-1.5-13b-hf',
        'cache_dir': 'llava-hf--llava-1.5-13b-hf',
        'requires_cuda': True
    },
    'MiniCPM-Llama3-V-2_5-int4': {
        'precision': '4-bit',
        'size': '8.4b',
        'repo_id': 'openbmb/MiniCPM-Llama3-V-2_5-int4',
        'cache_dir': 'openbmb--MiniCPM-Llama3-V-2_5-int4',
        'requires_cuda': True
    },
    'Cogvlm': {
        'precision': '4-bit',
        'size': '17.6b',
        'repo_id': 'THUDM/cogvlm-chat-hf',
        'cache_dir': 'THUDM--cogvlm-chat-hf',
        'requires_cuda': True
    }
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


CHUNKS_ONLY_TOOLTIP = "Only return relevant chunks without connecting to the LLM. Extremely useful to test the chunk size/overlap settings."

SPEAK_RESPONSE_TOOLTIP = "Only click this after the LLM's entire response is received otherwise your computer might explode."

DOWNLOAD_EMBEDDING_MODEL_TOOLTIP = "Remember, wait until downloading is complete!"

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
