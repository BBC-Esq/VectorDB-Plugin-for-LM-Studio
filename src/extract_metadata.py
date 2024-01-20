import os
import datetime

def extract_image_metadata(file_path, file_name):

    file_type = os.path.splitext(file_name)[1]
    file_size = os.path.getsize(file_path)
    creation_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    modification_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    return {
        "file_path": file_path,
        "file_type": file_type,
        "file_name": file_name,
        "file_size": file_size,
        "creation_date": creation_date,
        "modification_date": modification_date,
        "image": "True"
    }

def extract_document_metadata(file_path):
    file_type = os.path.splitext(file_path)[1]
    file_size = os.path.getsize(file_path)
    creation_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    modification_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    return {
        "file_path": str(file_path),
        "file_type": file_type,
        "file_name": file_path.name,
        "file_size": file_size,
        "creation_date": creation_date,
        "modification_date": modification_date,
        "image": "False"
    }


    """
    Extract metadata from an image file.

    Parameters:
    file_path (str): Full path to the image file.
    file_name (str): Name of the image file.

    Returns:
    dict: A dictionary containing extracted metadata.
    """