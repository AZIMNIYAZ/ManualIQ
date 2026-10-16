"""Step 11: checks the LLM judge against human grades.

1. Make a grading sheet from a generation run (20 answers, mixed categories, including
   answers the judge failed, so both good and bad answers are graded). The sheet does NOT
   show the judge's verdicts, so they cannot influence the grades:
       python scripts/judge_agreement.py make evaluation/results/<run>
   -> <run>/grading.md   (question, the exact context the model saw, the answer)
   -> <run>/grades.csv   (fill in: y / n; citations: y / n / na when the answer has no [n])

2. After grading, compare (agreement and Cohen's kappa, separately per dimension):
       python scripts/judge_agreement.py score evaluation/results/<run>
   -> <run>/agreement.md

The judge's groundedness/citation numbers are only used in the write-up if agreement is high.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import agreement  # noqa: E402
from evaluation.runs import read_jsonl, write_json  # noqa: E402

RUBRIC = """Grade each answer against the CONTEXT only (not your own knowledge of routers).

- grounded: y if every factual claim (steps, menu paths, button/field names, values) is in
  the context, for the same brand/OS/procedure the answer applies it to; n otherwise.
- citations: y if every [n] points to a source that supports that sentence/step; n if any
  does not; na if the answer has no [n] markers.
"""


def key(row: dict) -> str:
    return f"{row['id']}#{row['run']}"


def make(run_dir: Path, n: int, seed: int) -> None:
    rows = [r for r in read_jsonl(run_dir / "results.jsonl") if not r.get("error") and not r["refused"]
            and r.get("judge")]
    if not rows:
        sys.exit("No judged answers in this run.")
    rng = random.Random(seed)
    failed = [r for r in rows if not r["judge"]["grounded"] or r["judge"]["citations_correct"] is False]
    passed = [r for r in rows if r not in failed]
    picked = rng.sample(failed, min(len(failed), n // 2))  # up to half: answers the judge marked as bad
    by_cat = defaultdict(list)
    for r in rng.sample(passed, len(passed)):
        by_cat[r["category"]].append(r)
    mixed = [r for group in zip_longest(*by_cat.values()) for r in group if r is not None]
    picked += mixed[: n - len(picked)]
    rng.shuffle(picked)

    md = [f"# Grading sheet: {run_dir.name}", "", RUBRIC, "Fill in grades.csv (same order as below).", ""]
    for i, r in enumerate(picked, 1):
        md += [f"---\n\n## {i}. {key(r)} ({r['category']})", "",
               f"**Question:** {r['question']}" + (f"  \n**Searched as:** {r['rewritten_query']}" if r.get("rewritten_query") else ""),
               "", "**Context given to the model:**", "", "```text", r["context"] or "", "```", "",
               "**Answer:**", "", r["answer"], ""]
    (run_dir / "grading.md").write_text("\n".join(md), encoding="utf-8")
    with open(run_dir / "grades.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "category", "grounded", "citations", "notes"])
        for r in picked:
            w.writerow([key(r), r["category"], "", "", ""])
    print(f"{len(picked)} answers ({len([r for r in picked if r in failed])} the judge marked as failing).")
    print(f"Read {run_dir / 'grading.md'} and fill in {run_dir / 'grades.csv'}")


def _yes_no(value: str, allow_na: bool = False):
    v = value.strip().lower()
    if v in ("y", "yes"):
        return True
    if v in ("n", "no"):
        return False
    if allow_na and v in ("na", "n/a", "-"):
        return None
    raise ValueError(f"expected y/n{'/na' if allow_na else ''}, got {value!r}")


def score(run_dir: Path) -> None:
    by_key = {key(r): r for r in read_jsonl(run_dir / "results.jsonl")}
    grounded, citations, disagreements, problems = [], [], [], []
    with open(run_dir / "grades.csv", encoding="utf-8") as f:
        for line in csv.DictReader(f):
            row = by_key.get(line["item"])
            if row is None or not row.get("judge"):
                problems.append(f"{line['item']}: not a judged answer of this run")
                continue
            try:
                human_g = _yes_no(line["grounded"])
                human_c = _yes_no(line["citations"], allow_na=True)
            except ValueError as e:
                problems.append(f"{line['item']}: {e}")
                continue
            judge_g, judge_c = row["judge"]["grounded"], row["judge"]["citations_correct"]
            grounded.append((judge_g, human_g))
            if human_g != judge_g:
                disagreements.append(f"{line['item']} grounded: judge {judge_g}, human {human_g} "
                                     f"(judge: {row['judge']['explanation']}) {line['notes']}")
            if human_c is not None and judge_c is not None:
                citations.append((judge_c, human_c))
                if human_c != judge_c:
                    disagreements.append(f"{line['item']} citations: judge {judge_c}, human {human_c} "
                                         f"{row['judge']['citation_problems']} {line['notes']}")
    if problems:
        print("Ungraded or invalid rows (fix grades.csv):\n  " + "\n  ".join(problems))
    result = {"grounded": agreement(grounded), "citations": agreement(citations)}
    write_json(run_dir / "agreement.json", result)
    lines = [f"# Judge agreement: {run_dir.name}", "",
             "| dimension | graded | agreement | Cohen's kappa | disagreements |", "|---|---|---|---|---|"]
    for dim, a in result.items():
        if a["n"]:
            kappa = "n/a (judge and human gave one value only)" if a["kappa"] is None else a["kappa"]
            lines.append(f"| {dim} | {a['n']} | {a['agreement']:.0%} | {kappa} | {a['disagreements']} |")
        else:
            lines.append(f"| {dim} | 0 | - | - | - |")
    lines += ["", "Kappa corrects for chance agreement (when almost every answer is 'grounded', raw agreement is "
              "high even for a useless judge). Rough guide: > 0.6 substantial, 0.4-0.6 moderate, < 0.4 do not "
              "use the judge's numbers.", "", "## Disagreements", ""] + [f"- {d}" for d in disagreements or ["none"]]
    (run_dir / "agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["make", "score"])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if not (args.run_dir / "results.jsonl").exists():
        sys.exit(f"No results.jsonl in {args.run_dir}")
    if args.action == "make":
        if (args.run_dir / "grades.csv").exists():
            sys.exit(f"{args.run_dir / 'grades.csv'} already exists: delete it to make a new sheet")
        make(args.run_dir, args.n, args.seed)
    else:
        score(args.run_dir)


if __name__ == "__main__":
    main()
