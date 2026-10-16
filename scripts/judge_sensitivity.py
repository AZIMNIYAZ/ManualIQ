"""Step 11: can the judge catch a bad answer? (PAID, but tiny: ~12 calls to the judge model)

Takes real answers from a generation run that the judge called grounded, plants exactly one
error in each (by code, see evaluation/judge_tests.py) and asks the judge again. Reports how
many planted errors it caught, per error type, and combines these known-bad cases with the
human grades for an agreement/kappa that has negatives in it.

    python scripts/judge_sensitivity.py evaluation/results/<generation run>
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.services.generation_service import GrokGenerationService  # noqa: E402
from evaluation.judge import JUDGE_MAX_OUTPUT_TOKENS, JUDGE_PROMPT_VERSION, JudgeError, judge  # noqa: E402
from evaluation.judge_tests import MUTATIONS, detected, mutate  # noqa: E402
from evaluation.metrics import agreement  # noqa: E402
from evaluation.runs import read_jsonl, write_json  # noqa: E402


def human_pairs(run_dir: Path, rows_by_key: dict) -> list[tuple[bool, bool]]:
    """(judge, human) groundedness pairs from grades.csv, if it has been filled in."""
    import csv
    path = run_dir / "grades.csv"
    if not path.exists():
        return []
    pairs = []
    for line in csv.DictReader(open(path, encoding="utf-8")):
        row = rows_by_key.get(line["item"])
        if row and row.get("judge") and line["grounded"].strip().lower() in ("y", "n"):
            pairs.append((row["judge"]["grounded"], line["grounded"].strip().lower() == "y"))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    rows = read_jsonl(args.run_dir / "results.jsonl")
    by_key = {f"{r['id']}#{r['run']}": r for r in rows}
    good = [r for r in rows if not r.get("error") and not r["refused"] and r.get("judge")
            and r["judge"]["grounded"] and r["judge"]["citations_correct"] is not False]
    rng = random.Random(args.seed)
    rng.shuffle(good)

    cases, used = [], set()
    for i in range(args.n):  # error types in turn; each on a different real answer that allows it
        kind = MUTATIONS[i % len(MUTATIONS)]
        for r in good:
            k = f"{r['id']}#{r['run']}"
            if k in used:
                continue
            brand = r["brands"][0] if len(r["brands"]) == 1 else None
            mutated = mutate(kind, r["answer"], brand, r["context"] or "")
            if mutated:
                used.add(k)
                cases.append({"item": k, "kind": kind, "row": r, "answer": mutated})
                break

    settings = get_settings()
    model = settings.xai_rewrite_model or settings.xai_chat_model
    print(f"{len(cases)} planted-error answers -> {len(cases)} calls to {model} (judge prompt {JUDGE_PROMPT_VERSION})")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")
    judge_llm = GrokGenerationService(settings, model=model, temperature=0.0, max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS)

    results, per_kind = [], defaultdict(lambda: [0, 0])
    for c in cases:
        r = c["row"]
        try:
            verdict, _ = judge(judge_llm, r["context"] or "", r.get("rewritten_query") or r["question"], c["answer"])
            caught = detected(c["kind"], verdict)
            out = {"item": c["item"], "kind": c["kind"], "caught": caught, "judge": verdict.model_dump()}
        except JudgeError as e:
            caught, out = False, {"item": c["item"], "kind": c["kind"], "caught": False, "error": str(e)}
        out["answer"] = c["answer"]
        results.append(out)
        per_kind[c["kind"]][0] += caught
        per_kind[c["kind"]][1] += 1
        print(f"  {'CAUGHT' if caught else 'MISSED'} {c['kind']:17} {c['item']}"
              + (f"  ({out['judge']['explanation'][:110]})" if "judge" in out else f"  ERROR {out['error']}"))

    caught = sum(r["caught"] for r in results)
    # Groundedness agreement with negatives: human grades + planted errors (truth: not grounded)
    planted_g = [(r["judge"]["grounded"], False) for r in results if "judge" in r and r["kind"] != "wrong_citation"]
    combined = human_pairs(args.run_dir, by_key) + planted_g
    summary = {"judge_model": model, "judge_prompt_version": JUDGE_PROMPT_VERSION, "cases": len(results),
               "caught": caught, "per_kind": {k: {"caught": v[0], "cases": v[1]} for k, v in per_kind.items()},
               "grounded_agreement_with_planted": agreement(combined)}
    write_json(args.run_dir / "judge_sensitivity.json", {"summary": summary, "results": results})

    a = summary["grounded_agreement_with_planted"]
    lines = [f"# Judge sensitivity: {args.run_dir.name}", "",
             f"Planted errors caught: **{caught}/{len(results)}**", "",
             "| error type | caught |", "|---|---|"]
    lines += [f"| {k} | {v[0]}/{v[1]} |" for k, v in per_kind.items()]
    if a["n"]:
        lines += ["", f"Groundedness, human grades + planted errors: {a['n']} answers, agreement "
                      f"{a['agreement']:.0%}, Cohen's kappa {a['kappa']}"]
    missed = [f"- {r['item']} ({r['kind']}): " + (r["judge"]["explanation"] if "judge" in r else r.get("error", ""))
              for r in results if not r["caught"]]
    lines += ["", "## Missed", ""] + (missed or ["- none"])
    (args.run_dir / "judge_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
