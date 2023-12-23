AVAILABLE_MODELS = [
    {
        'details': {
            'description': 'Well rounded & customizable.',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134
        },
        'model': 'BAAI/bge-small-en-v1.5'
    },
    {
        'details': {
            'description': 'Well rounded & customizable.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438
        },
        'model': 'BAAI/bge-base-en-v1.5'
    },
    {
        'details': {
            'description': 'Well rounded & slight RAG improvement.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438
        },
        'model': 'BAAI/llm-embedder'
    },
    {
        'details': {
            'description': 'Well rounded & customizable.',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340
        },
        'model': 'BAAI/bge-large-en-v1.5'
    },
    {
        'details': {
            'description': 'Well rounded & customizable.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 439
        },
        'model': 'hkunlp/instructor-base'
    },
    {
        'details': {
            'description': 'Well rounded & customizable.',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340
        },
        'model': 'hkunlp/instructor-large'
    },
    {
        'details': {
            'description': 'Well rounded & customizable.',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 4960
        },
        'model': 'hkunlp/instructor-xl'
    },
    {
        'details': {
            'description': 'Well rounded',
            'dimensions': 312,
            'max_sequence': 512,
            'size_mb': 58
        },
        'model': 'jinaai/jina-embedding-t-en-v1'
    },
    {
        'details': {
            'description': 'Well rounded',
            'dimensions': 512,
            'max_sequence': 512,
            'size_mb': 141
        },
        'model': 'jinaai/jina-embedding-s-en-v1'
    },
    {
        'details': {
            'description': 'Well rounded',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 439
        },
        'model': 'jinaai/jina-embedding-b-en-v1'
    },
    {
        'details': {
            'description': 'Well rounded',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340
        },
        'model': 'jinaai/jina-embedding-l-en-v1'
    },
    {
        'details': {
            'description': 'Clustering or semantic search',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 329
        },
        'model': 'sentence-transformers/all-distilroberta-v1'
    },
    {
        'details': {
            'description': 'Clustering or semantic search',
            'dimensions': 384,
            'max_sequence': 256,
            'size_mb': 91
        },
        'model': 'sentence-transformers/all-MiniLM-L6-v2'
    },
    {
        'details': {
            'description': 'Clustering or semantic search',
            'dimensions': 768,
            'max_sequence': 384,
            'size_mb': 438
        },
        'model': 'sentence-transformers/all-mpnet-base-v2'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 219
        },
        'model': 'sentence-transformers/gtr-t5-base'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 670
        },
        'model': 'sentence-transformers/gtr-t5-large'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 2480
        },
        'model': 'sentence-transformers/gtr-t5-xl'
    },
    {
        'details': {
            'description': 'Clustering or semantic search',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 265
        },
        'model': 'sentence-transformers/msmarco-distilbert-base-v4'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 768,
            'max_sequence': 384,
            'size_mb': 265
        },
        'model': 'sentence-transformers/msmarco-distilbert-cos-v5'
    },
    {
        'details': {
            'description': 'Clustering or semantic search',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 91
        },
        'model': 'sentence-transformers/msmarco-MiniLM-L-6-v3'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 384,
            'max_sequence': 384,
            'size_mb': 91
        },
        'model': 'sentence-transformers/msmarco-MiniLM-L6-cos-v5'
    },
    {
        'details': {
            'description': 'Clustering or semantic search',
            'dimensions': 768,
            'max_sequence': 510,
            'size_mb': 499
        },
        'model': 'sentence-transformers/msmarco-roberta-base-v3'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 265
        },
        'model': 'sentence-transformers/multi-qa-distilbert-cos-v1'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 91
        },
        'model': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    },
    {
        'details': {
            'description': 'Semantic search.',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438
        },
        'model': 'sentence-transformers/multi-qa-mpnet-base-cos-v1'
    },
    {
        'details': {
            'description': 'Sentence similarity',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 219
        },
        'model': 'sentence-transformers/sentence-t5-base'
    },
    {
        'details': {
            'description': 'Sentence similarity',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 670
        },
        'model': 'sentence-transformers/sentence-t5-large'
    },
    {
        'details': {
            'description': 'Sentence similarity',
            'dimensions': 768,
            'max_sequence': 256,
            'size_mb': 2480
        },
        'model': 'sentence-transformers/sentence-t5-xl'
    },
    {
        'details': {
            'description': 'Well rounded',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 67
        },
        'model': 'thenlper/gte-small'
    },
    {
        'details': {
            'description': 'Well rounded',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 219
        },
        'model': 'thenlper/gte-base'
    },
    {
        'details': {
            'description': 'Well rounded',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 670
        },
        'model': 'thenlper/gte-large'
    }
]

DOCUMENT_LOADERS = {
    ".pdf": "PyMuPDFLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
    ".json": "JSONLoader",
    ".enex": "EverNoteLoader",
    ".eml": "UnstructuredEmailLoader",
    ".msg": "UnstructuredEmailLoader",
    ".csv": "UnstructuredCSVLoader",
    ".xls": "UnstructuredExcelLoader",
    ".xlsx": "UnstructuredExcelLoader",
    ".xlsm": "UnstructuredExcelLoader",
    ".rtf": "UnstructuredRTFLoader",
    ".odt": "UnstructuredODTLoader",
    ".md": "UnstructuredMarkdownLoader",
}

# Define model names
WHISPER_MODEL_NAMES = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2"]

CHUNKS_ONLY_TOOLTIP = "Only return relevant chunks without connecting to the LLM. Extremely useful to test the chunk size/overlap settings."

SPEAK_RESPONSE_TOOLTIP = "Only click this after the LLM's entire response is received otherwise your computer might explode."

DOWNLOAD_EMBEDDING_MODEL_TOOLTIP = "Remember, wait until downloading is complete!"