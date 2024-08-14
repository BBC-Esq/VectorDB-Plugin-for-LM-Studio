import os
import datetime
import hashlib
import re
import json
from langchain_core.documents import Document
from typing import List, Tuple

def compute_file_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def extract_common_metadata(file_path):
    file_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    file_type = os.path.splitext(file_path)[1]
    file_size = os.path.getsize(file_path)
    creation_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    modification_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    file_hash = compute_file_hash(file_path)

    return {
        "file_path": file_path,
        "file_type": file_type,
        "file_name": file_name,
        # "file_size": file_size, # was creating unspecified problems...
        "creation_date": creation_date,
        "modification_date": modification_date,
        "hash": file_hash
    }

def extract_image_metadata(file_path):
    metadata = extract_common_metadata(file_path)
    metadata["document_type"] = "image"
    return metadata

def extract_document_metadata(file_path):
    metadata = extract_common_metadata(file_path)
    metadata["document_type"] = "document"
    return metadata

def extract_audio_metadata(file_path):
    metadata = extract_common_metadata(file_path)
    metadata["document_type"] = "audio"
    return metadata

def add_pymupdf_page_metadata(doc: Document, chunk_size: int = 1200, chunk_overlap: int = 600) -> List[Document]:
    """
    Splits document objects originating from the custom pymupdfparser within langchain and adds page metadata.
    Called by document_processor.py.
    """
    def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int]]:
        page_markers = [(m.start(), int(m.group(1))) for m in re.finditer(r'\[\[page(\d+)\]\]', text)]
        clean_text = re.sub(r'\[\[page\d+\]\]', '', text)
        
        chunks = []
        start = 0
        while start < len(clean_text):
            end = start + chunk_size
            if end > len(clean_text):
                end = len(clean_text)
            chunk = clean_text[start:end].strip()
            
            page_num = None
            for marker_pos, page in reversed(page_markers):
                if marker_pos <= start:
                    page_num = page
                    break
            
            if chunk and page_num is not None:
                chunks.append((chunk, page_num))
            start += chunk_size - chunk_overlap
        
        return chunks

    chunks = split_text(doc.page_content, chunk_size, chunk_overlap)
    
    new_docs = []
    for chunk, page_num in chunks:
        new_metadata = doc.metadata.copy()
        new_metadata['page_number'] = page_num
        
        new_doc = Document(
            page_content=chunk,
            metadata=new_metadata
        )
        new_docs.append(new_doc)
    
    return new_docs