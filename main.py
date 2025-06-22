import os
import zipfile
import shutil
from fastapi import FastAPI, HTTPException
from utils import download_and_extract_zip, initialize_embeddings, query_documents

app = FastAPI()

DATA_DIR = "/persistent/data"
ZIP_FILE = "Recursos Humanos.zip"  # Debe estar en el repo Git
ZIP_PATH = os.path.join(os.path.dirname(__file__), ZIP_FILE)
EXTRACTED_PATH = os.path.join(DATA_DIR, "Recursos Humanos")

@app.on_event("startup")
async def startup_event():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(EXTRACTED_PATH):
        try:
            download_and_extract_zip(ZIP_PATH, EXTRACTED_PATH)
        except Exception as e:
            raise RuntimeError(f"Error al descomprimir ZIP: {str(e)}")

    initialize_embeddings(EXTRACTED_PATH)

@app.get("/query")
def ask_question(question: str):
    try:
        result = query_documents(question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
