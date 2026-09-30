import os
import sys
import shutil
from pathlib import Path

# Add backend directory to sys.path so sibling imports work reliably
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from foundry_local_sdk import Configuration, FoundryLocalManager
from openai import OpenAI
from database import init_db, clear_chunks
from llm_service import answer_query
from ingestion import ingest_pdf

app = FastAPI(title="Financial Document Assistant API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

config = Configuration(app_name="financial-document-assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
if not embed_model.is_cached:
    embed_model.download(lambda p: print(f"\rDownloading embedding model: {p:.0f}%", end="", flush=True))
    print()
embed_model.load()

chat_model = manager.catalog.get_model("phi-3.5-mini")
if not chat_model.is_cached:
    chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.0f}%", end="", flush=True))
    print()
chat_model.load()

manager.start_web_service()
base_url = f"{manager.urls[0]}/v1"
rest_client = OpenAI(base_url=base_url, api_key="not-needed")

embed_model_id = embed_model.id
chat_model_id = chat_model.id

class RestEmbeddingClient:
    def generate_embedding(self, text):
        return rest_client.embeddings.create(model=embed_model_id, input=text)

class RestChatClient:
    def complete_chat(self, messages):
        return rest_client.chat.completions.create(model=chat_model_id, messages=messages)

embed_client = RestEmbeddingClient()
chat_client = RestChatClient()

conn = init_db()

DOCS_DIR = str(BACKEND_DIR.parent / "docs")
FRONTEND_DIR = str(BACKEND_DIR.parent / "frontend")
os.makedirs(DOCS_DIR, exist_ok=True)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Question):
    if not q.question or not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer, sources = answer_query(q.question.strip(), embed_client, chat_client, conn)
    return {
        "answer": answer,
        "sources": [
            {"source": s, "page": p, "score": round(score, 3)}
            for score, s, p, content in sources
        ]
    }

@app.get("/status")
def status():
    row = conn.execute("SELECT DISTINCT source FROM chunks LIMIT 1").fetchone()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"source": row[0] if row else None, "chunks": count}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    for f in os.listdir(DOCS_DIR):
        if f.lower().endswith(".pdf"):
            try:
                os.remove(os.path.join(DOCS_DIR, f))
            except Exception:
                pass

    dest_path = os.path.join(DOCS_DIR, file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    clear_chunks(conn)
    source_name = os.path.splitext(file.filename)[0]
    ingest_pdf(dest_path, source_name, conn, embed_client)

    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"status": "ok", "filename": file.filename, "chunks": count}

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")