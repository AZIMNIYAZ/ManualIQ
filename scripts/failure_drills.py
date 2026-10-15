"""Step 10 check: triggers real failures and shows how the API reports each one.

Run from the project root (Qdrant + Redis running, rag_350_60 ingested):
    python scripts/failure_drills.py            # drills that need no container changes (free)
    python scripts/failure_drills.py --docker   # also stops/starts rag-qdrant and rag-redis
                                                # (restarts them afterwards; one paid chat call)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LOG_FILE"] = "none"  # only the API process writes the log file
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.generation_service import GrokGenerationService, get_generator  # noqa: E402

ROWS = []


def record(name: str, r, expect_status: int, expect_code: str | None) -> None:
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    err = body.get("error") or {}
    code = err.get("code")
    ok = r.status_code == expect_status and code == expect_code
    ids_match = err.get("request_id") == r.headers.get("X-Request-ID") if err else bool(r.headers.get("X-Request-ID"))
    ROWS.append((name, r.status_code, code or "-", r.headers.get("Retry-After", "-"), "yes" if ids_match else "NO",
                 "OK" if ok else f"EXPECTED {expect_status} {expect_code}"))


def docker(*args: str) -> None:
    subprocess.run(["docker", *args], check=True, capture_output=True)


def wait_for(url_check, seconds: int = 60) -> None:
    for _ in range(seconds):
        if url_check():
            return
        time.sleep(1)
    raise SystemExit("service did not come back")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args()
    base = get_settings()

    with TestClient(app, raise_server_exceptions=False) as api:
        chat = {"message": "How do I set up port forwarding on the Netgear router?"}
        # ---- request problems
        record("validation: blank query", api.post("/api/v1/search", json={"query": " "}), 422, "validation_error")
        record("validation: malformed JSON", api.post("/api/v1/search", content="{bad",
               headers={"Content-Type": "application/json"}), 422, "validation_error")
        record("unknown route", api.get("/api/v1/nope"), 404, "route_not_found")
        record("wrong method", api.get("/api/v1/chat"), 405, "method_not_allowed")
        record("unknown brand", api.post("/api/v1/search", json={"query": "reset", "brand": "linksys"}), 422, "invalid_brand")
        record("session not found", api.get("/api/v1/sessions/neverseen1"), 404, "session_not_found")

        # ---- empty knowledge base (a collection that was never ingested)
        empty = Settings(**{**base.model_dump(), "qdrant_collection_suffix": "drillempty"})
        app.dependency_overrides[get_settings] = lambda: empty
        record("empty knowledge base", api.post("/api/v1/search", json={"query": "reset"}), 409, "knowledge_base_empty")
        app.dependency_overrides.clear()

        # ---- LLM failures (real calls to xAI; a rejected key costs nothing)
        wrong = Settings(**{**base.model_dump(), "xai_api_key": "xai-" + "0" * 40})
        app.dependency_overrides[get_generator] = lambda: GrokGenerationService(wrong)
        record("wrong XAI_API_KEY", api.post("/api/v1/chat", json=chat), 502, "llm_authentication_failed")
        slow = Settings(**{**base.model_dump(), "llm_timeout_seconds": 0.001, "llm_max_retries": 0})
        app.dependency_overrides[get_generator] = lambda: GrokGenerationService(slow)
        record("LLM timeout (0.001 s)", api.post("/api/v1/chat", json=chat), 504, "llm_timeout")
        app.dependency_overrides.clear()

        if args.docker:
            # ---- Qdrant down
            docker("stop", "rag-qdrant")
            record("Qdrant down: /search", api.post("/api/v1/search", json={"query": "reset"}), 503, "vector_db_unavailable")
            record("Qdrant down: /chat", api.post("/api/v1/chat", json=chat), 503, "vector_db_unavailable")
            docker("start", "rag-qdrant")
            wait_for(lambda: api.get("/health").json().get("qdrant") == "ok")
            # ---- Redis down: /chat still answers (one paid call), sessions report 503
            docker("stop", "rag-redis")
            r = api.post("/api/v1/chat", json=chat)
            warned = any(w["type"] == "history_unavailable" for w in r.json().get("warnings", []))
            ROWS.append(("Redis down: /chat", r.status_code, "-", r.headers.get("Retry-After", "-"), "yes",
                         "OK (answered, history_unavailable)" if r.status_code == 200 and warned else "UNEXPECTED"))
            record("Redis down: /sessions", api.get("/api/v1/sessions/abcdefgh"), 503, "session_store_unavailable")
            docker("start", "rag-redis")
            wait_for(lambda: api.get("/health").json().get("redis") == "ok")

    print(f"{'drill':28} {'status':>6}  {'code':28} {'Retry-After':11} {'id match':8} result")
    for name, status, code, retry, ids, result in ROWS:
        print(f"{name:28} {status:>6}  {code:28} {retry:11} {ids:8} {result}")
    if not args.docker:
        print("\n(Qdrant/Redis outage drills skipped: run with --docker to include them)")


if __name__ == "__main__":
    main()
