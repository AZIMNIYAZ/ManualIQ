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

## Errors

Every error has the same shape, with the same `request_id` as the `X-Request-ID` header:

```json
{"error": {"code": "knowledge_base_empty", "message": "...", "request_id": "8f72c1a9b3e0", "details": null}}
```

`details` is a list of `{field, problem}` for `validation_error` (422) and `null` otherwise.
Codes: `validation_error`, `route_not_found`, `method_not_allowed`, `invalid_brand`,
`session_not_found`, `knowledge_base_empty`, `ingestion_in_progress`, `incompatible_collection`,
`confirmation_required`, `documents_invalid`, `ingestion_write_failed` (report in `details`),
`llm_rate_limited` (429, `Retry-After`), `llm_authentication_failed`, `llm_unavailable` (502),
`vector_db_unavailable`, `session_store_unavailable` (503, `Retry-After`),
`generation_not_configured` (503), `llm_timeout` (504), `internal_error` (500, no internal
details; the traceback is logged under the request ID). `/health` keeps its own report format.

## Logs

The API writes JSON lines to `LOG_FILE` (default `logs/app.jsonl`, rotated at 5 MB, 3 backups,
ignored by Git); scripts log to the console only. One `chat_completed` / `search_completed` /
`ingest_completed` event per request with timings, tokens, retrieved and cited chunks, models
and versions. Questions and answers are logged as lengths unless `LOG_USER_TEXT=true`; the
API key is redacted from every log record and error message.

```powershell
python scripts/log_report.py              # requests, errors, latency per stage, tokens
python scripts/failure_drills.py          # real failures and how each is reported
python scripts/failure_drills.py --docker # also stops/starts Qdrant and Redis
```

## Evaluation

`evaluation/questions.jsonl` has 40 questions: direct, paraphrased, no brand named, tables, context
confusion, unanswerable and out of scope. `evaluation/conversations.jsonl` has 6 conversations
(13 turns). Labels are chunk-independent, so they score every chunk size: manual + PDF page
range + key phrase. Several evidence locations count as alternatives, so retrieval is
reported as Hit@K and MRR. Pages are PDF page positions, not the numbers printed in the manuals.

```powershell
python scripts/check_labels.py --review evaluation/label_review.md  # free: labels vs manuals + review sheet
python scripts/eval_retrieval.py --label baseline                  # free: Hit@1/3/5/10, MRR, brand filter
python scripts/eval_generation.py --limit 5                        # paid: small test (asks first)
python scripts/eval_generation.py --label baseline                 # paid: ~73 answers + judge (asks first)
python scripts/judge_agreement.py make  evaluation/results/<run>   # grading sheet for 20 answers
python scripts/judge_agreement.py score evaluation/results/<run>   # judge vs human agreement
python scripts/judge_sensitivity.py evaluation/results/<run>       # paid (~12 judge calls): planted errors caught?
python scripts/eval_compare.py evaluation/results/<A> evaluation/results/<B>   # Step 12
```

- **Label checks:**
  - Every key phrase must be on its stated, indexed pages.
  - It should fit in one chunk at the current chunk size.
  - The terms recorded for each unanswerable question must have 0 hits in the raw and cleaned text. That is a safeguard, not proof: an answer could exist only in an image. The reviewer also searches the PDFs.
- **Answer scoring:**
  - Refusals: correct refusals, false answers and false refusals.
  - Key facts present, and no text from a wrong procedure.
  - An LLM judge checks groundedness, citations and brand mixing. It sees the exact context the model saw, uses a different (non-reasoning) model at temperature 0, and has a versioned prompt.
  - The judge's numbers are only used once its agreement with human grades (Cohen's kappa) is reported, separately for groundedness and citations.
- **No brand named:** the system should answer separately per brand. Mixing brands is a fail; asking which router is recorded as "asked", not as a pass.
- **Conversations:** the rewrite (brand, required terms) is scored separately. Retrieval is run both with the actual rewrite and with the expected question, so a failure is blamed on the rewriter, retrieval or generation.
- **Repeats:** tables and context-confusion questions run 3 times, and every run is shown.
- **Outputs:** each run saves `config.json` (all versions, models, settings and the dataset hash), `results.jsonl` and `summary.md` under `evaluation/results/`, which Git ignores. `eval_compare.py` refuses to compare runs made on different datasets.

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
- Two simultaneous `/chat` requests on the same session could interleave their turns in
  the history (each turn itself is written atomically). One user in one chat does not
  send overlapping requests, so no per-session lock is used.

## Conversations

Send the `session_id` returned by `/chat` with the next message. Follow-ups are rewritten
into standalone questions by a fast model (`XAI_REWRITE_MODEL`) using the last
`HISTORY_MESSAGES_IN_PROMPT` messages; previous answers are never sent to the answer
model. Brand precedence: the request's `brand` field (per request, never stored) >
brands from the rewrite > brands detected in the question. Without Redis, `/chat` still
answers (warning `history_unavailable`). `GET`/`DELETE /api/v1/sessions/{id}` show or
remove a conversation; sessions expire 24 h after the last message.

```powershell
python scripts/conversation_check.py --out conversation_check.txt   # 5 scripted conversations (live)
```
