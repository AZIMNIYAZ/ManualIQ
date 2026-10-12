"""Step 0 smoke test: verifies Qdrant, Redis, the embedding model and .env.

Run from the project root:  python scripts/check_env.py
"""

import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

ok = True


def report(name: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"[{'OK' if passed else 'FAIL'}] {name}{' - ' + detail if detail else ''}")


# 1. Qdrant: create a throwaway collection, insert a vector, search it, delete it
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    name = "smoke_test"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "chunk_001"))
    client.upsert(name, points=[PointStruct(id=point_id, vector=[0.1, 0.2, 0.3, 0.4], payload={"chunk_id": "chunk_001"})])
    hits = client.query_points(name, query=[0.1, 0.2, 0.3, 0.4], limit=1).points
    client.delete_collection(name)
    report("Qdrant insert + search", hits[0].payload["chunk_id"] == "chunk_001", f"score={hits[0].score:.3f}")
except Exception as e:
    report("Qdrant", False, repr(e))

# 2. Redis: push, read back, TTL, delete
try:
    import redis

    r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)
    key = "session:smoke_test"
    r.delete(key)
    r.rpush(key, '{"role":"user","content":"Hello"}')
    r.expire(key, 60)
    report("Redis push + read + TTL", r.lrange(key, 0, -1)[0].startswith("{") and 0 < r.ttl(key) <= 60)
    r.delete(key)
except Exception as e:
    report("Redis", False, repr(e))

# 3. Embedding model: 384-dim vectors, related text scores higher than unrelated
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"), device="cpu")
    prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "")
    q = model.encode(prefix + "reset router password", normalize_embeddings=True)
    docs = model.encode(["Change the admin password under System Tools.", "Mount the antenna on the back panel."], normalize_embeddings=True)
    related, unrelated = float(q @ docs[0]), float(q @ docs[1])
    report("Embedding dim", len(q) == 384, f"dim={len(q)}, max_seq_length={model.max_seq_length}")
    report("Embedding similarity", related > unrelated, f"related={related:.3f} unrelated={unrelated:.3f}")
except Exception as e:
    report("Embedding model", False, repr(e))

# 4. .env values that you must fill in yourself
missing = [k for k in ("XAI_API_KEY", "XAI_CHAT_MODEL") if not os.getenv(k)]
print(f"[{'TODO' if missing else 'OK'}] .env xAI settings{' - missing: ' + ', '.join(missing) if missing else ''}")

sys.exit(0 if ok else 1)
