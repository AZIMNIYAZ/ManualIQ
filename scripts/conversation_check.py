"""Step 9 check: scripted multi-turn conversations through the live system.

For each turn: the message, the rewritten (standalone) query, brands and their source,
the top citation and the answer. Rewrite latency is reported, not assumed.
Costs a few cents (11 answers + 6 small rewrite calls).

Run from the project root (Qdrant, Redis, rag_350_60 ingested, XAI_* in .env):
    python scripts/conversation_check.py --out conversation_check.txt
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.database.qdrant import create_qdrant_client  # noqa: E402
from app.database.redis import create_redis_client  # noqa: E402
from app.models.chat import ChatRequest  # noqa: E402
from app.services.brand_detection import alias_table_from_manifest  # noqa: E402
from app.services.chat_service import answer  # noqa: E402
from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.generation_service import create_rewriter, GrokGenerationService  # noqa: E402
from app.services.session_service import SessionStore  # noqa: E402

# (turns, what a correct rewrite of the follow-ups looks like)
CONVERSATIONS = [
    (["How do I set up port forwarding on my Netgear router?", "What about on ASUS?", "Can I edit the rule later?"],
     "brand switch to asus, then carry asus over for editing the rule"),
    (["How do I block a device on my Tenda router?", "How do I undo that?"],
     "'undo that' = remove the client from the blacklist, tenda"),
    (["How do I set up a guest network on my TP-Link router?", "Can I limit its bandwidth?"],
     "bandwidth limit for the guest network, tplink"),
    (["How do I update the firmware on my Netgear router?", "What's the weather tomorrow?"],
     "unrelated: no netgear carried over, not found"),
    (["How do I connect to Quick VPN from a Mac with the D-Link router?", "And on Windows?"],
     "Quick VPN on Windows, dlink (context-confusion case: same chunk, different OS)"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--conversation", type=int, help="run only this conversation (1-5)")
    args = parser.parse_args()
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")

    s = get_settings()
    client, embedder = create_qdrant_client(s), get_embedding_service()
    generator, rewriter = GrokGenerationService(s), create_rewriter(s)
    store = SessionStore(create_redis_client(s), "check_session", 600, s.session_max_stored_messages)
    aliases = alias_table_from_manifest(s.documents_dir)
    rewrite_ms = []

    print(f"answer model {generator.model}, rewrite model {rewriter.model}")
    for n, (turns, expected) in enumerate(CONVERSATIONS, 1):
        if args.conversation and n != args.conversation:
            continue
        print(f"\n{'=' * 100}\nCONVERSATION {n} - expected: {expected}")
        session_id = None
        for message in turns:
            r = answer(client, embedder, generator, s, ChatRequest(message=message, session_id=session_id),
                       aliases, sessions=store, rewriter=rewriter)
            session_id = r.session_id
            if r.rewrite:
                rewrite_ms.append(r.timings_ms["rewrite"])
            print(f"\n  USER: {message}")
            print(f"    rewritten: {r.rewritten_query!r} | brands {r.filters.brands or 'all'} ({r.filters.brand_source}) "
                  f"| history {r.history_used} | rewrite {r.timings_ms['rewrite']:.0f} ms, total {r.timings_ms['total']:.0f} ms")
            print(f"    top source: {r.sources[0].citation + ' [' + str(r.sources[0].section) + ']' if r.sources else 'none'}"
                  f" | found: {r.found_in_documents} | warnings: {[w['type'] for w in r.warnings] or 'none'}")
            print("    ANSWER: " + r.answer.replace("\n", "\n            "))
        store.delete(session_id)

    if rewrite_ms:
        print(f"\n{'=' * 100}\nrewrite latency over {len(rewrite_ms)} calls: median {statistics.median(rewrite_ms):.0f} ms, "
              f"max {max(rewrite_ms):.0f} ms (target ~1 s)")


if __name__ == "__main__":
    main()
