"""Step 11: retrieval evaluation (free: no LLM calls).

Runs every standalone question, plus every conversation turn using its *expected*
standalone question (the "oracle" rewrite), through the real /search code path with the
top 10 chunks. Conversations with the system's *actual* rewrites are scored in
eval_generation.py, so a miss can be blamed on the rewriter or on retrieval.

Run from the project root (Qdrant up, collection ingested):
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --label baseline
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.database.qdrant import create_qdrant_client  # noqa: E402
from app.models.search import SearchRequest  # noqa: E402
from app.services import retrieval_service  # noqa: E402
from app.services.brand_detection import alias_table_from_manifest  # noqa: E402
from app.services.embedding_service import get_embedding_service  # noqa: E402
from evaluation.dataset import (HOLDOUT_FILE, QUESTIONS_FILE, confirm_holdout, contains,  # noqa: E402
                                load_conversations, load_questions)
from evaluation.metrics import (K_VALUES, Retrieved, first_hit_rank, hit_at, matched_documents,  # noqa: E402
                                retrieval_summary)
from evaluation.runs import JsonlWriter, md_table, new_run_dir, pct, require_collection, run_config, write_json  # noqa: E402

TOP_K = 10


def to_retrieved(r) -> Retrieved:
    return Retrieved(chunk_id=r.chunk_id, document_id=r.document_id, brand=r.brand, page_start=r.page_start,
                     page_end=r.page_end, content_type=r.content_type, score=r.score, text=r.text)


def items(holdout: bool = False):
    """(category, item, query, source) for every standalone question and conversation turn."""
    for q in load_questions(HOLDOUT_FILE if holdout else QUESTIONS_FILE):
        yield q.category, q, q.question, "question"
    for conv in ([] if holdout else load_conversations()):
        for i, turn in enumerate(conv.turns):
            if i == 0:
                yield "conversation_first", turn, turn.question, "conversation"
            else:
                yield "conversation_followup", turn, turn.expected_query, "expected_rewrite"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="added to the results folder name")
    parser.add_argument("--holdout", action="store_true", help="the held-out questions (baseline and final only)")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.holdout:
        confirm_holdout(args.yes)

    settings = get_settings()
    client, embedder = create_qdrant_client(settings), get_embedding_service()
    aliases = alias_table_from_manifest(settings.documents_dir)
    require_collection(client, settings)
    run_dir = new_run_dir(settings, "retrieval_holdout" if args.holdout else "retrieval", args.label)
    config = run_config(settings, "retrieval", holdout=args.holdout, retrieval_top_k=TOP_K,
                        note="conversation follow-ups use expected_query (oracle rewrite)")
    write_json(run_dir / "config.json", config)
    out = JsonlWriter(run_dir / "results.jsonl")

    rows = []
    for category, item, query, source in items(args.holdout):
        found = retrieval_service.search(client, embedder, settings, SearchRequest(query=query, top_k=TOP_K),
                                         aliases, log_event=False)
        results = [to_retrieved(r) for r in found.results]
        expected_brands = [item.brand] if item.brand else []
        row = {
            "id": item.id, "category": category, "source": source, "query": query,
            "answerable": item.answerable,
            "rank": first_hit_rank(results, item.evidence) if item.answerable else None,
            "expected_brands": expected_brands, "brands": found.filters.brands,
            "brand_source": found.filters.brand_source,
            "brand_ok": sorted(found.filters.brands) == sorted(expected_brands),
            "regulatory_results": sum(r.content_type == "regulatory" for r in results),
            "top": [[r.chunk_id, r.document_id, r.page_start, r.page_end, r.score] for r in results],
        }
        if item.answerable and not hit_at(row["rank"], 5) and item.key_facts:
            # A miss whose top 5 still holds every key fact may be unlabelled valid evidence
            row["check_label"] = [[r.chunk_id, r.document_id, r.page_start, r.page_end] for r in results[:5]
                                  if (not item.brand or r.brand == item.brand)
                                  and all(contains(r.text, f) for f in item.key_facts)]
        if category == "no_brand":
            row["manuals_with_evidence"] = len({ev.document_id for ev in item.evidence})
            row["manuals_found@5"] = sorted(matched_documents(results, item.evidence, 5))
        out.write(row)
        rows.append(row)
    out.close()

    # ---- summary
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    by_category = {}
    for cat, rs in groups.items():
        answerable = [r for r in rs if r["answerable"]]
        by_category[cat] = {**retrieval_summary(answerable), "items": len(rs),
                            "brand_accuracy": round(sum(r["brand_ok"] for r in rs) / len(rs), 3)}
    answerable_q = [r for r in rows if r["answerable"] and r["source"] == "question"]
    summary = {
        "overall_questions": retrieval_summary(answerable_q),
        "overall_with_conversations": retrieval_summary([r for r in rows if r["answerable"]]),
        "by_category": by_category,
        "brand_accuracy": round(sum(r["brand_ok"] for r in rows) / len(rows), 3),
        "regulatory_leakage": sum(r["regulatory_results"] for r in rows),
        "no_brand_coverage@5": {r["id"]: f"{len(r['manuals_found@5'])}/{r['manuals_with_evidence']}"
                                for r in rows if r["category"] == "no_brand"},
        "misses@5": [r["id"] for r in rows if r["answerable"] and not (r["rank"] and r["rank"] <= 5)],
        "brand_errors": [{"id": r["id"], "expected": r["expected_brands"], "got": r["brands"]}
                         for r in rows if not r["brand_ok"]],
    }
    write_json(run_dir / "summary.json", {"config": config, "summary": summary})

    ks = [f"Hit@{k}" for k in K_VALUES]
    table = []
    order = ["direct", "paraphrased", "no_brand", "tables", "context_confusion", "unanswerable", "out_of_scope",
             "conversation_first", "conversation_followup"]
    for cat in [c for c in order if c in by_category]:
        s = by_category[cat]
        table.append([cat, s["items"], s.get("n", 0), *[pct(s.get(f"hit@{k}")) for k in K_VALUES],
                      "-" if not s.get("n") else f"{s['mrr']:.2f}", pct(s["brand_accuracy"])])
    o = summary["overall_questions"]
    table.append(["**all questions**", len([r for r in rows if r["source"] == "question"]), o["n"],
                  *[pct(o[f"hit@{k}"]) for k in K_VALUES], f"{o['mrr']:.2f}",
                  pct(sum(r["brand_ok"] for r in rows if r["source"] == "question")
                      / len([r for r in rows if r["source"] == "question"]))])
    lines = [
        f"# Retrieval evaluation: {config['collection']}",
        "",
        f"Dataset `{config['dataset_hash']}` | chunking {config['chunking']} | cleaner {config['cleaner_version']} | "
        f"{config['embedding_model']} | commit {config['git_commit']} | {config['started']}",
        "",
        "Hit@K: share of answerable questions with a matching chunk (same manual, overlapping pages, "
        "key phrase inside) in the top K. /chat uses the top "
        f"{settings.top_k}, so **Hit@{settings.top_k}** is the number that limits answers.",
        "",
        md_table(["category", "items", "answerable", *ks, "MRR", "brand filter"], table),
        "",
        f"Regulatory chunks in results: {summary['regulatory_leakage']} (should be 0)",
        "",
        "No-brand questions, manuals with a matching chunk in the top 5: "
        + ", ".join(f"{k} {v}" for k, v in summary["no_brand_coverage@5"].items()),
        "",
        "## Misses at 5",
        "",
    ]
    for r in rows:
        if r["answerable"] and not (r["rank"] and r["rank"] <= 5):
            top = "; ".join(f"{t[1]} pp.{t[2]}-{t[3]} ({t[4]:.2f})" for t in r["top"][:3])
            lines.append(f"- **{r['id']}** (rank {r['rank'] or '>10'}): {r['query']}  \n  top 3: {top}")
            for c in r.get("check_label") or []:
                lines.append(f"  - CHECK LABEL: {c[1]} pp. {c[2]}-{c[3]} ({c[0]}) is in the top 5 and contains "
                             "every key fact: unlabelled valid evidence?")
    if summary["brand_errors"]:
        lines += ["", "## Brand filter errors", ""]
        lines += [f"- {e['id']}: expected {e['expected'] or 'none'}, got {e['got'] or 'none'}"
                  for e in summary["brand_errors"]]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
