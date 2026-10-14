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
