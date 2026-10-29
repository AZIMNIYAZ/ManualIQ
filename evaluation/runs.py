"""Evaluation run folders: evaluation/results/<time>_<collection>_<kind>/

    config.json    everything that produced the numbers (versions, models, settings, dataset hash)
    results.jsonl  one row per question/turn (and per repeat)
    summary.json   the metrics, machine-readable (eval_compare.py reads these)
    summary.md     the same as tables, for the write-up
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from app.core.config import Settings
from app.services.manual_glossary import GLOSSARY_VERSION
from app.services.prompts import PROMPT_VERSION
from app.services.query_rewriter import REWRITE_PROMPT_VERSION
from app.services.retrieval_fallback import FALLBACK_PROMPT_VERSION
from app.services.text_cleaner import CLEANER_VERSION
from evaluation.dataset import HOLDOUT_FILE, dataset_hash

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "app", "evaluation/questions.jsonl",
                                "evaluation/conversations.jsonl", "evaluation/holdout.jsonl"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_config(settings: Settings, kind: str, holdout: bool = False, **extra) -> dict:
    return {
        "kind": kind,
        "dataset": "holdout" if holdout else "dev",
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_hash": dataset_hash(HOLDOUT_FILE) if holdout else dataset_hash(),
        "collection": settings.collection_name,
        "chunking": settings.chunking,
        "contextual_headers": settings.contextual_headers,
        "cleaner_version": CLEANER_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "top_k": settings.top_k,
        "context_max_tokens": settings.context_max_tokens,
        "prompt_version": PROMPT_VERSION,
        "rewrite_prompt_version": REWRITE_PROMPT_VERSION,
        "answer_model": settings.xai_chat_model,
        "rewrite_model": settings.xai_rewrite_model,
        "temperature": settings.llm_temperature,
        "retrieval_fallback": settings.retrieval_fallback,
        **({"fallback_prompt_version": FALLBACK_PROMPT_VERSION, "glossary_version": GLOSSARY_VERSION,
            "fallback_queries": settings.fallback_queries, "fallback_merge": settings.fallback_merge}
           if settings.retrieval_fallback else {}),
        **extra,
    }


def require_collection(client, settings: Settings) -> None:
    """Fails before a results folder is created (or anything is paid for)."""
    try:
        exists = client.collection_exists(settings.collection_name)
    except Exception as e:
        raise SystemExit(f"Qdrant unavailable at {settings.qdrant_url} ({e}): run `docker compose up -d`")
    if not exists:
        raise SystemExit(f"Collection {settings.collection_name} does not exist: run python scripts/ingest.py")
    from app.database.qdrant import collection_problems
    problems = collection_problems(client, settings.collection_name, settings.embedding_model_version,
                                   settings.chunking, settings.contextual_headers)
    if problems:
        raise SystemExit(f"Collection {settings.collection_name} does not match the settings: " + "; ".join(problems))


def new_run_dir(settings: Settings, kind: str, label: str | None = None) -> Path:
    name = f"{datetime.now():%Y%m%d-%H%M%S}_{settings.collection_name}_{kind}" + (f"_{label}" if label else "")
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class JsonlWriter:
    """Appends one row at a time, flushed, so an interrupted paid run keeps what it paid for."""

    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
