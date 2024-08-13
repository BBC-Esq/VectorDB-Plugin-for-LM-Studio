import re
import json
from langchain_core.documents import Document
from typing import List, Tuple
import pprint

def chunk_and_annotate_document(doc: Document, chunk_size: int = 1200, overlap: int = 600) -> List[Document]:
    def split_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int]]:
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
            start += chunk_size - overlap
        
        return chunks

    chunks = split_text(doc.page_content, chunk_size, overlap)
    
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

def main():
    # Read the documents from the specified file
    input_file_path = r"D:\Scripts\VectorDB-Plugin-for-LM-Studio\working\text_documents_pdf.json"
    with open(input_file_path, 'r', encoding='utf-8') as file:
        docs_data = json.load(file)

    all_chunked_docs = []
    
    # Handle both list and dictionary cases
    if isinstance(docs_data, list):
        for doc_data in docs_data:
            if 'kwargs' in doc_data and 'page_content' in doc_data['kwargs']:
                original_doc = Document(
                    page_content=doc_data['kwargs']['page_content'],
                    metadata=doc_data['kwargs'].get('metadata', {})
                )
                chunked_docs = chunk_and_annotate_document(original_doc)
                all_chunked_docs.extend(chunked_docs)
            else:
                print(f"Skipping document due to unexpected structure: {doc_data}")
    elif isinstance(docs_data, dict):
        if 'kwargs' in docs_data and 'page_content' in docs_data['kwargs']:
            original_doc = Document(
                page_content=docs_data['kwargs']['page_content'],
                metadata=docs_data['kwargs'].get('metadata', {})
            )
            all_chunked_docs = chunk_and_annotate_document(original_doc)
        else:
            print("Unexpected document structure")
    else:
        print("Unexpected data type in input file")

    output_data = [
        {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
        for doc in all_chunked_docs
    ]

    # Save the chunked documents to a new file
    output_file_path = r"D:\Scripts\VectorDB-Plugin-for-LM-Studio\working\pdf_docs_chunked.json"
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    print(f"Processed {len(all_chunked_docs)} chunks. Output saved to {output_file_path}")

if __name__ == "__main__":
    main()