import os
import time
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import datetime
from tqdm import tqdm
from langchain.docstore.document import Document
import platform
import gc

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    elif hasattr(torch.version, 'hip') and torch.version.hip and platform.system() == 'Linux':
        return 'cuda'
    else:
        return 'cpu'

def salesforce_process_images():
    script_dir = os.path.dirname(__file__)
    image_dir = os.path.join(script_dir, "Images_for_DB")
    documents = []

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print("The 'Images_for_DB' directory was created as it was not detected.")
        return documents

    if not os.listdir(image_dir):
        print("No files detected in the 'Images_for_DB' directory.")
        return documents

    device = get_best_device()
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    total_tokens = 0
    total_start_time = time.time()

    with tqdm(total=len(os.listdir(image_dir)), unit="image") as progress_bar:
        for file_name in os.listdir(image_dir):
            full_path = os.path.join(image_dir, file_name)
            try:
                with Image.open(full_path) as raw_image:
                    inputs = processor(raw_image, return_tensors="pt").to(device)
                    output = model.generate(**inputs, max_new_tokens=50)
                    caption = processor.decode(output[0], skip_special_tokens=True)
                    total_tokens += output[0].shape[0]

                    # Create a Document object for each image
                    extracted_metadata = {
                        "file_path": full_path,
                        "file_name": file_name,
                        "file_type": os.path.splitext(file_name)[1],
                        "file_size": os.path.getsize(full_path),
                        "creation_date": datetime.datetime.fromtimestamp(os.path.getctime(full_path)).isoformat(),
                        "modification_date": datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat(),
                        "caption": caption
                    }
                    document = Document(page_content=caption, metadata=extracted_metadata)
                    documents.append(document)

                    progress_bar.update(1)

            except Exception as e:
                print(f"{file_name}: Error processing image - {e}")

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    print(f"Total image processing time: {total_time_taken:.2f} seconds")
    print(f"Tokens per second: {total_tokens / total_time_taken:.2f}")

    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return documents
