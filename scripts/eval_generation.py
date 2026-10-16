"""Step 11: answer evaluation through the real /chat code path (PAID: calls xAI).

For every question (and every turn of every conversation, with real Redis sessions and the
real rewriter) it records the answer, the context, the retrieved chunks and the automatic
warnings, then scores:
  - refusals (correct refusals, false answers, false refusals)
  - key facts present, wrong-procedure text present
  - groundedness, citation correctness and brand mixing (LLM judge, temperature 0)
  - conversations: rewrite (brand, required terms), retrieval with the actual rewrite vs
    the expected question, and which component a failure is blamed on
Tables and context-confusion questions run 3 times by default (answers vary between runs);
every run is reported, not just a percentage.

The script prints the number of calls and an estimate of tokens and cost, and asks for
confirmation before calling xAI.

Run from the project root (Qdrant + Redis up, collection ingested, XAI_* in .env):
    python scripts/eval_generation.py --limit 5            # cheap first test (5 questions + 1 conversation)
    python scripts/eval_generation.py --label baseline     # everything
    python scripts/eval_generation.py --category tables --repeats 3
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.database.qdrant import create_qdrant_client  # noqa: E402
from app.database.redis import create_redis_client  # noqa: E402
from app.models.chat import ChatRequest  # noqa: E402
from app.models.search import SearchRequest  # noqa: E402
from app.services import retrieval_service  # noqa: E402
from app.services.brand_detection import alias_table_from_manifest  # noqa: E402
from app.services.chat_service import answer  # noqa: E402
from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.generation_service import (GenerationError, GenerationNotConfigured,  # noqa: E402
                                             GrokGenerationService, LLMAuthenticationError, create_rewriter)
from app.services.ingestion_service import point_id  # noqa: E402
from app.services.session_service import SessionStore  # noqa: E402
from evaluation.dataset import REPEAT_CATEGORIES, load_conversations, load_questions  # noqa: E402
from evaluation.judge import JUDGE_MAX_OUTPUT_TOKENS, JUDGE_PROMPT_VERSION, JudgeError, judge  # noqa: E402
from evaluation.metrics import Retrieved, first_hit_rank, refusal_counts  # noqa: E402
from evaluation.runs import (JsonlWriter, md_table, new_run_dir, pct, require_collection,  # noqa: E402
                             run_config, write_json)
from evaluation.scoring import DEFAULT_TOKENS, blame, estimate, rewrite_ok, score_answer  # noqa: E402

CATEGORY_ORDER = ["direct", "paraphrased", "no_brand", "tables", "context_confusion", "unanswerable", "out_of_scope"]


# ------------------------------------------------------------------ planning

def select_questions(args) -> list:
    questions = load_questions()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    if args.category and args.category != "conversation":
        questions = [q for q in questions if q.category == args.category]
    elif args.category == "conversation":
        questions = []
    if args.limit:  # round-robin over categories, so a small test still covers several
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q.category].append(q)
        mixed = [q for group in zip_longest(*by_cat.values()) for q in group if q is not None]
        questions = mixed[: args.limit]
    return questions


def select_conversations(args) -> list:
    if args.no_conversations or args.ids or (args.category and args.category != "conversation"):
        return []
    conversations = load_conversations()
    return conversations[:1] if args.limit else conversations


def repeats_for(question, args) -> int:
    if args.repeats is not None:
        return args.repeats if question.category in REPEAT_CATEGORIES or args.category else 1
    return 3 if question.category in REPEAT_CATEGORIES else 1


def measured_tokens(log_file: str | None) -> tuple[dict, str]:
    """Average tokens per call from the API log (chat_completed events), else defaults."""
    path = Path(log_file or "logs/app.jsonl")
    sums, counts = Counter(), Counter()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(e, dict) or e.get("message") != "chat_completed":
                continue
            t = e.get("tokens") or {}
            if t.get("answer_in"):
                for k in ("answer_in", "answer_out", "reasoning"):
                    sums[k] += t.get(k) or 0
                    counts[k] += 1
            if t.get("rewrite_in"):
                for k in ("rewrite_in", "rewrite_out"):
                    sums[k] += t.get(k) or 0
                    counts[k] += 1
    per_call = {k: round(sums[k] / counts[k]) if counts[k] else v for k, v in DEFAULT_TOKENS.items()}
    source = f"averages of {counts['answer_in']} logged answers" if counts["answer_in"] else "defaults (no API log yet)"
    return per_call, source


# ------------------------------------------------------------------ one call

def fetch_chunks(client, settings, chunk_ids: list[str]) -> list[Retrieved]:
    if not chunk_ids:
        return []
    points = client.retrieve(settings.collection_name, ids=[point_id(c) for c in chunk_ids], with_payload=True)
    by_id = {p.payload["chunk_id"]: p.payload for p in points}
    return [Retrieved(chunk_id=c, document_id=by_id[c]["document_id"], brand=by_id[c]["brand"],
                      page_start=by_id[c]["page_start"], page_end=by_id[c]["page_end"],
                      content_type=by_id[c]["content_type"], score=0.0, text=by_id[c]["text"])
            for c in chunk_ids if c in by_id]


def run_one(ctx, item, policy: str, message: str, session_id: str | None, use_sessions: bool) -> dict:
    s = ctx["settings"]
    response = answer(ctx["client"], ctx["embedder"], ctx["generator"], s,
                      ChatRequest(message=message, session_id=session_id, debug=True), ctx["aliases"],
                      sessions=ctx["store"] if use_sessions else None,
                      rewriter=ctx["rewriter"] if use_sessions else None)
    retrieved = fetch_chunks(ctx["client"], s, [r.chunk_id for r in response.retrieved or []])
    refused = not response.found_in_documents
    verdict, judge_usage, judge_error = None, None, None
    if ctx["judge"] is not None and not refused:
        try:
            verdict, judge_usage = judge(ctx["judge"], response.context or "", response.rewritten_query or message,
                                         response.answer)
        except JudgeError as e:
            judge_error = str(e)
    row = {
        "session_id": response.session_id,
        "question": message,
        "rewritten_query": response.rewritten_query,
        "answer": response.answer,
        "refused": refused,
        "brands": response.filters.brands, "brand_source": response.filters.brand_source,
        "sources": [s_.citation for s_ in response.sources],
        "warnings": [w["type"] for w in response.warnings],
        "warning_details": response.warnings,
        "retrieved": [[r.chunk_id, r.document_id, r.page_start, r.page_end] for r in retrieved],
        "rank": first_hit_rank(retrieved, item.evidence) if item.answerable else None,
        "context": response.context,
        "llm": response.llm.model_dump(),
        "rewrite_tokens": response.rewrite.model_dump() if response.rewrite else None,
        "timings_ms": response.timings_ms,
        "judge": verdict.model_dump() if verdict else None,
        "judge_usage": judge_usage,
        "judge_error": judge_error,
    }
    row["score"] = score_answer(item, policy, response.answer, refused, verdict, ctx["aliases"])
    return row


# ------------------------------------------------------------------ summary

def mark(row, ascii_only: bool = False) -> str:
    """✓ pass, ✗ fail, ! error (ASCII P/F/! for the Windows console)."""
    if row.get("error"):
        return "!"
    if ascii_only:
        return "P" if row["score"]["pass"] else "F"
    return "✓" if row["score"]["pass"] else "✗"


def rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    q_rows = [r for r in ok if r["kind"] == "question"]
    by_cat = {}
    for cat in CATEGORY_ORDER:
        rs = [r for r in q_rows if r["category"] == cat]
        if not rs:
            continue
        sc = [r["score"] for r in rs]
        by_cat[cat] = {
            "runs": len(rs),
            "pass": rate([s["pass"] for s in sc]),
            "key_fact_coverage": rate([s["key_fact_coverage"] for s in sc]),
            "must_not_hits": sum(bool(s["must_not_found"]) for s in sc),
            "grounded": rate([s["grounded"] for s in sc]),
            "citations_correct": rate([s["citations_correct"] for s in sc]),
            "brand_mixing": sum(s["brand_mixing"] is True for s in sc),
            "refused": sum(r["refused"] for r in rs),
            "hit@5": rate([r["rank"] is not None for r in rs if r["answerable"]]),
            "outcomes": dict(Counter(s["outcome"] for s in sc)),
        }
    repeats = defaultdict(list)
    for r in rows:
        if r["kind"] == "question":
            repeats[r["id"]].append(r)
    turn_rows = [r for r in ok if r["kind"] == "turn"]
    follow = [r for r in turn_rows if r["turn"] > 0]
    tokens = Counter()
    for r in ok:
        tokens["answer_in"] += r["llm"]["input_tokens"]
        tokens["answer_out"] += r["llm"]["output_tokens"]
        tokens["reasoning"] += r["llm"]["reasoning_tokens"]
        if r.get("rewrite_tokens"):
            tokens["rewrite_in"] += r["rewrite_tokens"]["input_tokens"]
            tokens["rewrite_out"] += r["rewrite_tokens"]["output_tokens"]
        if r.get("judge_usage"):
            tokens["judge_in"] += r["judge_usage"]["input_tokens"]
            tokens["judge_out"] += r["judge_usage"]["output_tokens"] + r["judge_usage"]["reasoning_tokens"]
    return {
        "items": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
        "judge_errors": sum(bool(r.get("judge_error")) for r in ok),
        "questions": {
            "pass": rate([r["score"]["pass"] for r in q_rows]),
            "refusals": refusal_counts([{"answerable": r["answerable"], "refused": r["refused"]} for r in q_rows]),
            "grounded": rate([r["score"]["grounded"] for r in q_rows]),
            "citations_correct": rate([r["score"]["citations_correct"] for r in q_rows]),
            "brand_mixing": sum(r["score"]["brand_mixing"] is True for r in q_rows),
            "warnings": dict(Counter(w for r in q_rows for w in r["warnings"])),
        },
        "by_category": by_cat,
        "repeats": {i: "".join(mark(r) for r in rs) for i, rs in repeats.items() if len(rs) > 1},
        "conversations": {
            "turns": len(turn_rows),
            "pass": rate([r["score"]["pass"] for r in turn_rows]),
            "followups": len(follow),
            "rewrite_brand_ok": rate([r["rewrite_check"]["brand_ok"] for r in follow]),
            "rewrite_terms_ok": rate([r["rewrite_check"]["terms_ok"] for r in follow]),
            "hit@5_actual_rewrite": rate([r["rank"] is not None for r in follow if r["answerable"]]),
            "hit@5_expected_query": rate([r["oracle_rank"] is not None for r in follow if r["answerable"]]),
            "blame": dict(Counter(r["blame"] for r in turn_rows if r["blame"])),
        },
        "tokens": dict(tokens),
        "generate_ms_median": statistics.median([r["timings_ms"]["generate"] for r in ok]) if ok else None,
    }


def write_markdown(run_dir: Path, config: dict, summary: dict, rows: list[dict]) -> str:
    q = summary["questions"]
    lines = [
        f"# Answer evaluation: {config['collection']}",
        "",
        f"Dataset `{config['dataset_hash']}` | prompt {config['prompt_version']} | rewrite prompt "
        f"{config['rewrite_prompt_version']} | answer model {config['answer_model']} | judge {config['judge_model']} "
        f"(prompt {config['judge_prompt_version']}) | commit {config['git_commit']} | {config['started']}",
        "",
        f"Items {summary['items']}, errors {summary['errors']}, judge errors {summary['judge_errors']}. "
        "The judge's numbers are only valid once its agreement with the human grades is reported "
        "(scripts/judge_agreement.py).",
        "",
        "## By category",
        "",
    ]
    table = []
    for cat, s in summary["by_category"].items():
        table.append([cat, s["runs"], pct(s["pass"]), pct(s["hit@5"]), pct(s["key_fact_coverage"]), s["must_not_hits"],
                      pct(s["grounded"]), pct(s["citations_correct"]), s["brand_mixing"], s["refused"],
                      ", ".join(f"{k} {v}" for k, v in s["outcomes"].items())])
    lines.append(md_table(["category", "runs", "pass", "Hit@5", "key facts", "wrong-procedure text", "grounded*",
                           "citations ok*", "brand mixing*", "refused", "outcomes"], table))
    rc = q["refusals"]
    lines += [
        "",
        "\\* LLM judge, answers that were not refusals only.",
        "",
        "## Refusals (all question runs)",
        "",
        md_table(["answered (should)", "false refusals", "correct refusals", "false answers", "refusal precision",
                  "refusal recall"],
                 [[rc["answered"], rc["false_refusals"], rc["correct_refusals"], rc["false_answers"],
                   pct(rc["refusal_precision"]), pct(rc["refusal_recall"])]]),
        "",
        "False answers = answered although the manuals do not contain the answer (the dangerous error).",
        "",
        "Automatic warnings: " + (", ".join(f"{k} x{v}" for k, v in q["warnings"].items()) or "none"),
    ]
    if summary["repeats"]:
        lines += ["", "## Repeated questions (each run shown; ! = error)", ""]
        lines += [f"- {i}: {marks}" for i, marks in summary["repeats"].items()]
    c = summary["conversations"]
    if c["turns"]:
        lines += [
            "", "## Conversations", "",
            md_table(["turns", "pass", "follow-ups", "rewrite brand ok", "rewrite terms ok",
                      "Hit@5 actual rewrite", "Hit@5 expected question", "failures blamed on"],
                     [[c["turns"], pct(c["pass"]), c["followups"], pct(c["rewrite_brand_ok"]),
                       pct(c["rewrite_terms_ok"]), pct(c["hit@5_actual_rewrite"]), pct(c["hit@5_expected_query"]),
                       ", ".join(f"{k} {v}" for k, v in c["blame"].items()) or "none"]]),
        ]
        for r in rows:
            if r["kind"] == "turn" and not r.get("error"):
                lines.append(f"- {r['id']} {mark(r)} {r['question']!r} -> {r['rewritten_query']!r} "
                             f"(brands {r['brands'] or 'none'})" + (f" blame: {r['blame']}" if r["blame"] else ""))
    lines += ["", "## Failures", ""]
    for r in rows:
        if r.get("error"):
            lines.append(f"- **{r['id']}** run {r['run']}: ERROR {r['error']}")
        elif not r["score"]["pass"]:
            s = r["score"]
            why = [s["outcome"]]
            if s["key_facts_missing"] and r["answerable"] and not r["refused"]:
                why.append(f"missing {s['key_facts_missing']}")
            if s["must_not_found"]:
                why.append(f"wrong-procedure text {s['must_not_found']}")
            if s["grounded"] is False:
                why.append("judge: not grounded")
            if s["brand_mixing"]:
                why.append("judge: brand mixing")
            lines.append(f"- **{r['id']}** run {r['run']} (Hit@5 {'yes' if r['rank'] else 'no'}): {'; '.join(why)}")
    t = summary["tokens"]
    lines += ["", "## Usage", "",
              f"Tokens: answer in {t.get('answer_in', 0)}, out {t.get('answer_out', 0)}, reasoning {t.get('reasoning', 0)}; "
              f"rewrite in {t.get('rewrite_in', 0)}, out {t.get('rewrite_out', 0)}; judge in {t.get('judge_in', 0)}, "
              f"out {t.get('judge_out', 0)}. Median generation {summary['generate_ms_median']:.0f} ms."
              if summary["generate_ms_median"] is not None else "No successful calls."]
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only N questions (mixed categories) + the first conversation")
    parser.add_argument("--category", help="one question category, or 'conversation'")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--repeats", type=int, help="runs for tables/context-confusion questions (default 3)")
    parser.add_argument("--no-conversations", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", help="defaults to XAI_REWRITE_MODEL (non-reasoning)")
    parser.add_argument("--label")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # the Windows console cannot show ✓/✗ (summary.md is UTF-8)

    settings = get_settings()
    questions, conversations = select_questions(args), select_conversations(args)
    jobs = [(q, run) for q in questions for run in range(1, repeats_for(q, args) + 1)]
    turns = sum(len(c.turns) for c in conversations)
    followups = sum(len(c.turns) - 1 for c in conversations)
    n_answers = len(jobs) + turns
    if not n_answers:
        sys.exit("Nothing selected.")
    judge_model = args.judge_model or settings.xai_rewrite_model or settings.xai_chat_model
    per_call, source = measured_tokens(settings.log_file)
    est = estimate(n_answers, followups, 0 if args.no_judge else n_answers, per_call,
                   settings.llm_price_input_per_m, settings.llm_price_output_per_m)

    print(f"Plan: {len(questions)} questions -> {len(jobs)} answer runs, {len(conversations)} conversations "
          f"({turns} turns, {followups} rewrites), judge {'off' if args.no_judge else judge_model}")
    print(f"Calls: {est['answer_calls']} answers, {est['rewrite_calls']} rewrites, up to {est['judge_calls']} judge calls")
    print(f"Estimated tokens (upper bound, {source}): input {est['input_tokens']:,}, output+reasoning "
          f"{est['output_tokens']:,}")
    if est["cost_usd"] is not None:
        print(f"Estimated cost: ${est['cost_usd']:.2f} (one price pair for all models; check xAI's pricing page)")
    else:
        print("Cost: set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M in .env for a dollar estimate")
    if not args.yes and input("Proceed with the paid run? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled: nothing was sent to xAI.")

    client = create_qdrant_client(settings)
    require_collection(client, settings)
    try:
        generator = GrokGenerationService(settings)
        judge_llm = None if args.no_judge else GrokGenerationService(
            settings, model=judge_model, temperature=0.0, max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS)
    except GenerationNotConfigured as e:
        sys.exit(str(e))
    store = None
    if conversations:
        redis = create_redis_client(settings)
        try:
            redis.ping()
        except Exception as e:
            sys.exit(f"Redis unavailable ({e}): conversations need sessions. Run `docker compose up -d` "
                     "or pass --no-conversations")
        store = SessionStore(redis, "eval_session", 1800, settings.session_max_stored_messages)
    ctx = {"settings": settings, "client": client, "embedder": get_embedding_service(), "generator": generator,
           "rewriter": create_rewriter(settings), "judge": judge_llm, "store": store,
           "aliases": alias_table_from_manifest(settings.documents_dir)}

    run_dir = new_run_dir(settings, "generation", args.label)
    config = run_config(settings, "generation", judge_model=None if args.no_judge else judge_model,
                        judge_prompt_version=JUDGE_PROMPT_VERSION, judge_temperature=0.0,
                        selection={k: v for k, v in vars(args).items() if k not in ("yes",)}, estimate=est)
    write_json(run_dir / "config.json", config)
    out = JsonlWriter(run_dir / "results.jsonl")
    rows: list[dict] = []

    def record(row: dict) -> None:
        out.write(row)
        rows.append(row)
        status = mark(row, ascii_only=True)
        detail = row.get("error") or row["score"]["outcome"]
        print(f"  {status} {row['id']} run {row['run']}: {detail}", flush=True)

    try:
        for q, run in jobs:
            base = {"id": q.id, "kind": "question", "category": q.category, "run": run, "answerable": q.answerable,
                    "policy": q.policy}
            try:
                record({**base, **run_one(ctx, q, q.policy, q.question, None, use_sessions=False)})
            except (LLMAuthenticationError, GenerationNotConfigured):
                raise
            except GenerationError as e:
                record({**base, "error": f"{e.code}: {e}"})

        for conv in conversations:
            session_id = None
            for n, turn in enumerate(conv.turns):
                base = {"id": turn.id, "kind": "turn", "conversation": conv.id, "turn": n, "category": "conversation",
                        "run": 1, "answerable": turn.answerable, "policy": "single_brand" if turn.answerable else "refuse",
                        "expected_query": turn.expected_query}
                try:
                    row = run_one(ctx, turn, base["policy"], turn.question, session_id, use_sessions=True)
                except (LLMAuthenticationError, GenerationNotConfigured):
                    raise
                except GenerationError as e:
                    record({**base, "error": f"{e.code}: {e}"})
                    break  # the rest of the conversation depends on this turn
                session_id = row["session_id"]
                expected_brands = [turn.brand] if turn.brand else []
                if n > 0:
                    brand_ok, terms_ok = rewrite_ok(expected_brands, row["brands"],
                                                    row["rewritten_query"] or turn.question, turn.required_terms)
                    oracle = retrieval_service.search(client, ctx["embedder"], settings,
                                                      SearchRequest(query=turn.expected_query), ctx["aliases"],
                                                      log_event=False)
                    oracle_rank = first_hit_rank([Retrieved(r.chunk_id, r.document_id, r.brand, r.page_start,
                                                            r.page_end, r.content_type, r.score, r.text)
                                                  for r in oracle.results], turn.evidence) if turn.answerable else None
                else:
                    brand_ok, terms_ok = sorted(row["brands"]) == sorted(expected_brands), True
                    oracle_rank = row["rank"]
                row["rewrite_check"] = {"brand_ok": brand_ok, "terms_ok": terms_ok}
                row["oracle_rank"] = oracle_rank
                row["blame"] = blame(row["score"]["pass"], turn.answerable, brand_ok and terms_ok,
                                     (row["rank"] is not None) if turn.answerable else None,
                                     (oracle_rank is not None) if turn.answerable else None)
                record({**base, **row})
            if session_id and store is not None:
                store.delete(session_id)
    except (LLMAuthenticationError, GenerationNotConfigured) as e:
        print(f"Stopped: {e}")
    finally:
        out.close()

    summary = summarise(rows)
    write_json(run_dir / "summary.json", {"config": config, "summary": summary})
    print("\n" + write_markdown(run_dir, config, summary, rows))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
