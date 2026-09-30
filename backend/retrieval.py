import json
import math
from foundry_local_sdk import Configuration, FoundryLocalManager
from database import init_db

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def get_top_chunks(query, embed_client, conn, top_k=3):
    query_emb = embed_client.generate_embedding(query).data[0].embedding

    rows = conn.execute("SELECT source, page, content, embedding FROM chunks").fetchall()

    scored = []
    for source, page, content, emb_json in rows:
        emb = json.loads(emb_json)
        score = cosine_similarity(query_emb, emb)
        scored.append((score, source, page, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

if __name__ == "__main__":
    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.load()
    embed_client = model.get_embedding_client()

    conn = init_db()

    test_query = "What are the key risk factors of the company?"
    results = get_top_chunks(test_query, embed_client, conn)

    for score, source, page, content in results:
        print(f"[{score:.3f}] {source} - page {page}")
        print(content[:200])
        print("---")