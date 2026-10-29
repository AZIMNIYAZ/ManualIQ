# ManualIQ: conversational RAG over router manuals

FastAPI backend (with a chat page) for question answering over five router user manuals: TP-Link
Archer AX55, Netgear Nighthawk RAX50, ASUS RT-AX58U V2, D-Link DIR-X1560 and Tenda AX3000. It uses local
Qdrant for vector search, Redis for conversation memory, local `bge-small-en-v1.5` embeddings and
xAI Grok for generation. Every answer cites the manual and PDF pages it comes from; questions the
manuals do not cover get a fixed "I couldn't find this in the provided manuals."

```
question ─► (follow-up? rewrite into a standalone question, Redis history)
         ─► brand detection ─► vector search in Qdrant (500-token chunks embedded with "Brand Model | Section")
         ─► top 5 chunks ─► Grok answers from those sources only, with [n] citations
         ─► refusal? ─► retrieval fallback: alternative searches in the manuals' own vocabulary
                        (table-of-contents glossary) ─► one more answer
```

## Results

Measured on 40 questions + 6 conversations (development set) and 15 frozen held-out questions that
were never used for decisions. Details, every experiment and every accepted item: `evaluation/EXPERIMENTS.md`.

| | Step 11 baseline | Final (Step 12) |
|---|---|---|
| Development set (73 answer runs) | 75% | 94% automatic, 100% after human review |
| Held-out set (23 answer runs) | 23/23 | 21/23 automatic, 22/23 after review |
| False answers (answering when the manuals do not contain the answer) | 0 | 0 on every set |

What Step 12 changed: 500/60 chunks, contextual headers in the embeddings, the retrieval fallback,
answer prompt 1.2 and a 1,500-token answer limit. Tried and rejected (no clear gain): hybrid BM25
search, cross-encoder re-ranking, parent-child chunking and rewriting every query.

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

## The manuals (not included)

The PDFs are copyrighted, so they are not in the repository (`documents/*.pdf` is ignored). Download
them from each vendor's support site and save them in `documents/` under these names
(`documents/manifest.json` lists the original file names):

| Save as | Manual (original file name) |
|---|---|
| `tplink_archer_ax55.pdf` | TP-Link Archer AX55 user guide (`tp-link_Archer AX55_UG_REV1.0.0.pdf`) |
| `netgear_rax50.pdf` | Netgear Nighthawk RAX50 user manual (`NETGEAR-RAX50_UM_EN.pdf`) |
| `asus_rt_ax58u.pdf` | ASUS RT-AX58U V2 user manual (`ASUS-E23448_RT-AX58U_V2_UM_V2_WEB.pdf`) |
| `dlink_dir_x1560.pdf` | D-Link DIR-X1560 manual (`D-LINK_DIR-X1560_REVA_MANUAL_v1.00_WW.pdf`) |
| `tenda_wifi6.pdf` | Tenda AX3000 Wi-Fi 6 (RX12 Pro / TX12 Pro) user guide (`tenda-wifi6-20230130160159_7414.pdf`) |

`documents/glossary.json` (the manuals' headings, used by the retrieval fallback) is included; rebuild it
with `python scripts/build_glossary.py` if a manual changes.

## Building the knowledge base

```powershell
python scripts/ingest.py                          # collection rag_500_60_ctx (the defaults)
# or start the API and POST /api/v1/ingest (Postman collection in postman/)
python scripts/ingest.py --size 350 --overlap 60  # an experiment collection, e.g. the Step 11 baseline
```

The collection name is derived from the chunk settings (`rag_{size}_{overlap}`, plus `_ctx` with
contextual headers), so a collection can only hold chunks of its own configuration. Re-running ingestion is safe:
point IDs come from chunk IDs (overwrite, never duplicate), and points from older runs are
deleted only after the new run is fully written and its count verified.

## Asking questions

```powershell
uvicorn app.main:app --reload
```

- **Chat page:** open http://127.0.0.1:8000/. Type a question; answers show their sources (manual and
  pages). Follow-ups ("And on the D-Link?") remember the conversation; **New conversation** starts over;
  the Router menu limits answers to one manual.
- **API docs:** http://127.0.0.1:8000/docs. **Health:** http://127.0.0.1:8000/health.

Command-line checks:

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

Step 12 additions: `--questions-file evaluation/fallback_probes.jsonl` (11 questions about features the
manuals do not have, to check the fallback never invents answers), `--conversations cv-02,cv-04` and
`--holdout` (typed confirmation; baseline and final configuration only).

## Retrieval fallback

When the first answer is a refusal (`RETRIEVAL_FALLBACK=true`, the default), the rewrite model is given
the question and the detected manual's table-of-contents headings and writes 3 alternative searches
("kick it off for good" → "Add a client to the blacklist"). They are searched with the original brand
filter and merged with the first results, and the answer model answers the original question once more.
Answered questions never enter it. Each alternative must keep the question's technical terms
("AiMesh", "6 GHz", "OpenVPN") and add no brand, so an unsupported feature is never swapped for a
supported one. A second refusal is final.

## Known limitations

Answers (details in `evaluation/EXPERIMENTS.md`):
- The fallback only runs after a refusal: an answer built on similar but wrong pages does not trigger it.
- The fallback's term check treats common abbreviations such as "IP" as protected terms, so a correct
  alternative search ("Set up Dynamic DNS") can be rejected and the refusal stands.
- "Can it…?" questions may be answered "yes, with feature X" without the steps; ask "how?" to get them.
- Features that are only loosely related are not offered as alternatives: the answer is a refusal.
- Refusals take a few seconds longer (one more search and answer).
- Text only: steps that refer to icons lose the icon ("click [icon] under Network Topology").

Engineering, fine for this single-user project:
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
