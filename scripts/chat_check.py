"""Step 8 check with the real LLM: answers, cited sources, warnings and token usage.

Costs money (a few cents): runs 2 questions by default; --all runs all 12.
Run from the project root (Qdrant running, rag_350_60 ingested, XAI_* set in .env):
    python scripts/chat_check.py --out chat_check.txt
    python scripts/chat_check.py --all --out chat_check.txt
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.database.qdrant import create_qdrant_client  # noqa: E402
from app.models.chat import ChatRequest  # noqa: E402
from app.services.brand_detection import alias_table_from_manifest  # noqa: E402
from app.services.chat_service import answer  # noqa: E402
from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.generation_service import GrokGenerationService  # noqa: E402

QUESTIONS = [
    "How do I set up port forwarding on the Netgear router?",          # in scope
    "What is the default login IP address of the Tenda router?",        # grounding test: retrieval misses the table
    "How do I block services on my Netgear router?",
    "How do I remove a client from the blacklist on the Tenda router?",
    "How do I use the Tether app to manage my TP-Link router?",
    "How do I connect a device using the router's WPS PIN on TP-Link?",
    "How do I enable the firewall on the ASUS router?",
    "How do I use AiDisk to share files from a USB drive on ASUS?",
    "How do I connect to Quick VPN from a Mac with the D-Link router?",
    "How do I mount the D-Link router on a wall?",
    "What is the weather forecast for tomorrow?",                       # out of scope
    "Who won the football World Cup in 2022?",                          # out of scope
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="all 12 questions (default: the first 2)")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")

    settings = get_settings()
    client, embedder = create_qdrant_client(settings), get_embedding_service()
    generator = GrokGenerationService(settings)
    aliases = alias_table_from_manifest(settings.documents_dir)

    questions = QUESTIONS if args.all else QUESTIONS[:2]
    totals = {"input": 0, "output": 0, "reasoning": 0}
    for i, q in enumerate(questions, 1):
        r = answer(client, embedder, generator, settings, ChatRequest(message=q, debug=True), aliases)
        totals["input"] += r.llm.input_tokens
        totals["output"] += r.llm.output_tokens
        totals["reasoning"] += r.llm.reasoning_tokens
        print(f"\n{'=' * 100}\n[{i}] Q: {q}")
        print(f"filters: {r.filters.brands or 'all'} ({r.filters.brand_source}) | found_in_documents: "
              f"{r.found_in_documents} | tokens in/out/reasoning: {r.llm.input_tokens}/{r.llm.output_tokens}/"
              f"{r.llm.reasoning_tokens} | {r.timings_ms['total']:.0f} ms")
        print(f"\nANSWER:\n{r.answer}\n")
        print("CITED SOURCES:" + ("" if r.sources else " none"))
        for s in r.sources:
            print(f"  [{s.n}] {s.citation} [{s.section}] score {s.score}")
        print("WARNINGS: " + (", ".join(f"{w['type']}: {w['detail']}" for w in r.warnings) or "none"))
        print("RETRIEVED (sent to the LLM): " + "; ".join(f"[{s.n}] {s.citation}" for s in r.retrieved or []))

    print(f"\n{'=' * 100}\nTOTAL tokens: input {totals['input']}, output {totals['output']}, "
          f"reasoning {totals['reasoning']} (reported separately, usually billed as output), "
          f"{len(questions)} questions, model {settings.xai_chat_model}")
    if settings.llm_price_input_per_m is not None and settings.llm_price_output_per_m is not None:
        billed_output = totals["output"] + totals["reasoning"]
        cost = (totals["input"] * settings.llm_price_input_per_m + billed_output * settings.llm_price_output_per_m) / 1e6
        print(f"Estimated cost: ${cost:.4f}")
    else:
        print("Cost not estimated: set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M from xAI's pricing page")
    print("Check each answer against its cited sources (python scripts/chat_check.py prints them with --out).")


if __name__ == "__main__":
    main()
