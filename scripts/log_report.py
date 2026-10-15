"""Summarises the API's JSON-lines log (LOG_FILE, default logs/app.jsonl, plus rotated files).

Run from the project root:
    python scripts/log_report.py
    python scripts/log_report.py --file logs/app.jsonl --out log_report.txt

- Latency statistics are computed per stage, only from events that contain that stage
  (a request that failed at search contributes no "generate" time).
- Tolerates missing fields, non-JSON lines and an incomplete last line.
- Tokens are summed as reported: on this API, reasoning tokens were observed to be
  reported separately from output tokens (e.g. output 9, reasoning 326), so they are
  added, not double-counted. Confirm billing definitions on xAI's pricing page.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402


def read_events(path: Path) -> tuple[list[dict], int]:
    """Oldest rotated file first (app.jsonl.3 ... .1, then app.jsonl); returns (events, skipped lines)."""
    files = sorted(path.parent.glob(path.name + ".*"), key=lambda p: -int(p.suffix[1:]) if p.suffix[1:].isdigit() else 0)
    events, skipped = [], 0
    for f in [*files, path]:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    events.append(event)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1  # e.g. a line cut off by a crash mid-write
    return events, skipped


def endpoint(path: str | None) -> str:
    """Groups paths by route: /api/v1/sessions/<id> -> /api/v1/sessions/{id}."""
    path = path or "?"
    return "/api/v1/sessions/{id}" if path.startswith("/api/v1/sessions/") else path


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def stats_row(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name:16} {'-':>6}"
    return (f"  {name:16} {len(values):>6} {percentile(values, 50):>10.0f} {percentile(values, 95):>10.0f} "
            f"{max(values):>10.0f}")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=settings.log_file or "logs/app.jsonl")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")

    events, skipped = read_events(Path(args.file))
    if not events:
        sys.exit(f"No log events in {args.file} (is LOG_FILE set and has the API handled requests?)")
    first, last = events[0].get("time", "?"), events[-1].get("time", "?")
    print(f"Log: {args.file} | {len(events)} events from {first} to {last} | {skipped} unreadable lines skipped")

    # ---- requests and errors
    requests = [e for e in events if e.get("message") == "request"]
    by_path = Counter((endpoint(e.get("path")), e.get("status")) for e in requests)
    print(f"\nREQUESTS ({len(requests)})")
    for (path, status), n in sorted(by_path.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        print(f"  {str(path):34} {status}  x{n}")
    failures = [e for e in events if e.get("message") in ("request_failed", "unhandled_error")]
    errors = Counter(e.get("code", "?") for e in failures)
    rate = len(failures) / len(requests) if requests else 0
    print(f"\nERRORS ({len(failures)}, {rate:.0%} of requests): " + (", ".join(f"{c} x{n}" for c, n in errors.most_common()) or "none"))
    slow = [e for e in events if e.get("message") == "slow_request"]
    print(f"SLOW REQUESTS (> {settings.slow_request_ms:.0f} ms): {len(slow)}")

    # ---- latency per stage, only from events that have the stage
    print(f"\nLATENCY (ms)      {'count':>6} {'median':>10} {'p95':>10} {'max':>10}")
    for event_name, label in (("chat_completed", "/chat"), ("search_completed", "/search")):
        stages = defaultdict(list)
        for e in (e for e in events if e.get("message") == event_name):
            for stage, value in (e.get("timings_ms") or {}).items():
                if isinstance(value, (int, float)) and not (stage == "rewrite" and value == 0):
                    stages[stage].append(float(value))
        if stages:
            print(f" {label}")
            for stage in ("rewrite", "checks", "embed", "vector_search", "search", "generate", "total"):
                if stage in stages:
                    print(stats_row(stage, stages[stage]))
    known = ("/api/v1/chat", "/api/v1/search", "/api/v1/ingest", "/api/v1/sessions/{id}")
    print(" whole request (as seen by the client)")
    for path in known:
        durations = [float(e["duration_ms"]) for e in requests if endpoint(e.get("path")) == path and "duration_ms" in e]
        if durations:
            print(stats_row(path.removeprefix("/api/v1/"), durations))

    # ---- chat quality signals and tokens
    chats = [e for e in events if e.get("message") == "chat_completed"]
    if chats:
        not_found = sum(1 for e in chats if e.get("found_in_documents") is False)
        rewritten = sum(1 for e in chats if (e.get("tokens") or {}).get("rewrite_in"))
        warnings = Counter(w for e in chats for w in (e.get("warnings") or []))
        print(f"\nCHAT ({len(chats)} answers): not found {not_found} ({not_found / len(chats):.0%}), "
              f"follow-ups rewritten {rewritten}")
        print("  warnings: " + (", ".join(f"{w} x{n}" for w, n in warnings.most_common()) or "none"))
        tokens = Counter()
        for e in chats:
            tokens.update({k: v for k, v in (e.get("tokens") or {}).items() if isinstance(v, (int, float))})
        print(f"  tokens: answer in {tokens['answer_in']}, out {tokens['answer_out']}, reasoning {tokens['reasoning']}; "
              f"rewrite in {tokens['rewrite_in']}, out {tokens['rewrite_out']} "
              f"(reasoning reported separately from output on this API)")
        if settings.llm_price_input_per_m is not None and settings.llm_price_output_per_m is not None:
            cost = ((tokens["answer_in"] + tokens["rewrite_in"]) * settings.llm_price_input_per_m
                    + (tokens["answer_out"] + tokens["reasoning"] + tokens["rewrite_out"]) * settings.llm_price_output_per_m) / 1e6
            print(f"  estimated cost: ${cost:.4f} (one price for both models; check xAI's pricing page)")
        else:
            print("  cost: set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M to estimate")


if __name__ == "__main__":
    main()
