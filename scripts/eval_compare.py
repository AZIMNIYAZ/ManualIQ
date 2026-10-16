"""Step 12: compares two evaluation runs of the same kind, side by side.

    python scripts/eval_compare.py evaluation/results/<baseline> evaluation/results/<experiment>

Refuses to compare runs made on different datasets (dataset_hash), since the numbers would
not be comparable; --force overrides (the output then says so). Shows every config value
that differs, so a change in a number can be traced to what was changed.
"""

import argparse
import json
import sys
from pathlib import Path

IGNORED_CONFIG = {"started", "git_commit", "selection", "estimate", "note"}


def load(run: Path) -> dict:
    path = run / "summary.json"
    if not path.exists():
        sys.exit(f"No summary.json in {run}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, key: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and "mrr" in key:
        return f"{v:.2f}"
    if isinstance(v, float):
        return f"{v:.0%}" if 0 <= v <= 1 else f"{v:.2f}"
    return str(v)


def delta(a, b, key: str = "") -> str:
    if isinstance(a, float) and isinstance(b, float) and "mrr" in key:
        return f"{b - a:+.2f}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = b - a
        if isinstance(a, float) or isinstance(b, float):
            return f"{d * 100:+.0f} pts" if all(0 <= x <= 1 for x in (a, b)) else f"{d:+.2f}"
        return f"{d:+d}"
    return ""


def rows_for(kind: str, s: dict) -> dict[str, object]:
    out = {}
    if kind == "retrieval":
        for k, v in s["overall_questions"].items():
            out[f"all questions {k}"] = v
        for cat, m in s["by_category"].items():
            for k in ("hit@1", "hit@5", "hit@10", "mrr", "brand_accuracy"):
                if k in m:
                    out[f"{cat} {k}"] = m[k]
        out["regulatory leakage"] = s["regulatory_leakage"]
    else:
        q = s["questions"]
        out["all pass"] = q["pass"]
        out["grounded (judge)"] = q["grounded"]
        out["citations ok (judge)"] = q["citations_correct"]
        out["brand mixing"] = q["brand_mixing"]
        for k in ("false_answers", "false_refusals", "refusal_precision", "refusal_recall"):
            out[k.replace("_", " ")] = q["refusals"][k]
        for cat, m in s["by_category"].items():
            for k in ("pass", "hit@5", "key_fact_coverage", "grounded"):
                out[f"{cat} {k}"] = m[k]
        for k, v in s["conversations"].items():
            if not isinstance(v, dict):
                out[f"conversations {k}"] = v
        out["repeats"] = json.dumps(s["repeats"]).replace("\\u2713", "P").replace("\\u2717", "F")  # P pass, F fail
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # the Windows console cannot show ✓/✗ (use --out for UTF-8)
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")
    a, b = load(args.a), load(args.b)
    ca, cb = a["config"], b["config"]
    if ca["kind"] != cb["kind"]:
        sys.exit(f"Different kinds of run: {ca['kind']} vs {cb['kind']}")
    if ca["dataset_hash"] != cb["dataset_hash"]:
        if not args.force:
            sys.exit(f"Different datasets ({ca['dataset_hash']} vs {cb['dataset_hash']}): not comparable. "
                     "Re-run both on the same dataset, or --force.")
        print(f"WARNING: different datasets ({ca['dataset_hash']} vs {cb['dataset_hash']}): not strictly comparable")

    print(f"A: {args.a.name}\nB: {args.b.name}\n")
    print("CONFIG DIFFERENCES")
    diffs = [(k, ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb))
             if k not in IGNORED_CONFIG and ca.get(k) != cb.get(k)]
    for k, va, vb in diffs or [("(none)", "", "")]:
        print(f"  {k:28} {fmt(va):>30}  ->  {fmt(vb)}")

    ra, rb = rows_for(ca["kind"], a["summary"]), rows_for(cb["kind"], b["summary"])
    print(f"\n{'metric':42} {'A':>10} {'B':>10}  change")
    for k in list(dict.fromkeys([*ra, *rb])):
        va, vb = ra.get(k), rb.get(k)
        if k == "repeats":
            print(f"\nrepeats A: {va}\nrepeats B: {vb}")
            continue
        print(f"{k:42} {fmt(va, k):>10} {fmt(vb, k):>10}  {delta(va, vb, k)}")


if __name__ == "__main__":
    main()
