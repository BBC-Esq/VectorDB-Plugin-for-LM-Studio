import os
import datetime
import hashlib

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
        # "file_size": file_size,
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
