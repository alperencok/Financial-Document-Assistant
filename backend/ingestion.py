import json
from pypdf import PdfReader
from foundry_local_sdk import Configuration, FoundryLocalManager
from database import init_db

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_pdf(pdf_path, source_name, conn, embed_client):
    reader = PdfReader(pdf_path)
    print(f"Loaded PDF: {len(reader.pages)} pages detected.", flush=True)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        for chunk in chunk_text(text):
            if len(chunk.strip()) < 30:
                continue
            emb = embed_client.generate_embedding(chunk).data[0].embedding
            conn.execute(
                "INSERT INTO chunks (source, page, content, embedding) VALUES (?, ?, ?, ?)",
                (source_name, page_num, chunk, json.dumps(emb))
            )
        if page_num % 20 == 0:
            print(f"Processed page {page_num} / {len(reader.pages)}...", flush=True)
    conn.commit()
    print("Document ingestion completed successfully.", flush=True)

if __name__ == "__main__":
    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.download(lambda p: print(f"\rDownloading model: {p:.0f}%", end="", flush=True))
    print()
    model.load()
    embed_client = model.get_embedding_client()

    conn = init_db()
    ingest_pdf("../docs/sample.pdf", "sample", conn, embed_client)

    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Total chunks indexed: {count}")