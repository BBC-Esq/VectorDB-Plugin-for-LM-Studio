import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/Docs_for_DB"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
INGEST_THREADS = os.cpu_count() or 8
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

openai.api_base = 'http://localhost:1234/v1'
openai.api_key = ''

prefix = "[INST]"
suffix = "[/INST]"

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Variable to store the last response
last_response = None

def connect_to_local_chatgpt(prompt):
    formatted_prompt = f"{prefix}{prompt}{suffix}"
    response = openai.ChatCompletion.create(
        model="local model",
        temperature=0.7,
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    return response.choices[0].message["content"]

def ask_local_chatgpt(query, embed_model_name=EMBEDDING_MODEL_NAME, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS):
    embeddings = HuggingFaceInstructEmbeddings(model_name=embed_model_name)
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=client_settings,
    )
    retriever = db.as_retriever()
    relevant_contexts = retriever.get_relevant_documents(query)
    contexts = [document.page_content for document in relevant_contexts]
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
    response_json = connect_to_local_chatgpt(augmented_query)
    return {"answer": response_json, "sources": relevant_contexts}

def interact_with_chat(user_input):
    global last_response
    response = ask_local_chatgpt(user_input)
    answer = response['answer']
    last_response = answer
    return answer

def get_last_response():
    global last_response
    return last_response
