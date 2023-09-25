from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)
    texts = text_splitter.split_documents(documents)
    
    return texts
