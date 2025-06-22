import os
import zipfile
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage

STORAGE_DIR = "embeddings"
index = None

def download_and_extract_zip(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def initialize_embeddings(pdf_folder):
    global index
    if os.path.exists(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(pdf_folder, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=OpenAIEmbedding(),
        )
        index.storage_context.persist(persist_dir=STORAGE_DIR)

def query_documents(question):
    global index
    query_engine = index.as_query_engine()
    return query_engine.query(question).response