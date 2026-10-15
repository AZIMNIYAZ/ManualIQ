# Conversational RAG Backend

FastAPI backend for question answering over router user manuals, using local Qdrant for vector search,
Redis for session memory, local `bge-small-en-v1.5` embeddings and xAI Grok for generation.

## Scope: text-only

The system uses the text layer of the PDFs only. Images, screenshots and diagrams are not
processed (no OCR, no image models). Tables are included as flattened text; their structure
is not reconstructed. Questions whose answers exist only in images are out of scope.

## Setup

```powershell
# 1. Start Qdrant (localhost:6333, dashboard at /dashboard) and Redis (localhost:6379)
docker compose up -d

# 2. Python environment (Python 3.12)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Config: copy the template, then fill in XAI_API_KEY and XAI_CHAT_MODEL
copy .env.example .env

# 4. Verify everything is wired up
python scripts/check_env.py
```

Stop the services with `docker compose down` (data is kept in Docker volumes; add `-v` to wipe it).

## Building the knowledge base

```powershell
uvicorn app.main:app --reload
# then POST /api/v1/ingest (Postman collection in postman/), or without the API:
python scripts/ingest.py                          # collection rag_350_60 (from .env)
python scripts/ingest.py --size 200 --overlap 40  # experiment collection rag_200_40
```

The collection name is derived from the chunk settings (`rag_{size}_{overlap}`), so a
collection can only hold chunks of its own configuration. Re-running ingestion is safe:
point IDs come from chunk IDs (overwrite, never duplicate), and points from older runs are
deleted only after the new run is fully written and its count verified.

## Asking questions

```powershell
python scripts/llm_check.py                 # free: checks XAI_CHAT_MODEL is available to your key
python scripts/chat_check.py --out chat_check.txt         # 2 live questions (costs a few cents)
python scripts/chat_check.py --all --out chat_check.txt   # all 12
```

`POST /api/v1/chat` answers from the manuals with cited sources; `debug: true` returns the
exact context sent to the LLM, which is what grounding must be judged against.

## Open items for evaluation (Steps 11-12)

- **Tables:** the Tenda factory-settings table is not retrieved for "default login IP"
  (flattened tables embed poorly). Test contextual headers and hybrid keyword search.
- **Context confusion:** chunks holding several variants of a procedure (D-Link Quick VPN
  for iOS/Android/Mac/Windows, Netgear OpenVPN Windows/Mac, Tenda blacklist add/remove).
  Compare prompt 1.1 with a "procedure isolation" rule (1.2); run each question 2-3 times,
  since answers vary between runs.
- **Prompt and model:** non-reasoning vs reasoning Grok models (latency and cost).

## Known limitations

Fine for this single-user project:
- If a write fails part-way, the collection is reported as `mixed` and ingestion must be
  re-run. Production fix: build a new collection and switch a Qdrant alias atomically.
- The "one ingestion at a time" lock is per process: it covers the API with one worker,
  not the API and `scripts/ingest.py` running at the same moment.
