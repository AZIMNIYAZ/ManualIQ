"""Builds a Qdrant collection from the manuals, without the API.

Run from the project root:
    python scripts/ingest.py                              # settings from .env -> rag_350_60
    python scripts/ingest.py --size 200 --overlap 40      # experiment collection rag_200_40
    python scripts/ingest.py --recreate --confirm rag_350_60
    python scripts/ingest.py --size 500 --overlap 60 --headers  # contextual headers -> rag_500_60_ctx

The chunker and the collection name are both taken from one settings object, so
--size 200 --overlap 40 really produces 200/40 chunks in rag_200_40.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import Settings, get_settings  # noqa: E402
from app.database.qdrant import create_qdrant_client  # noqa: E402
from app.services.embedding_service import create_embedding_service  # noqa: E402
from app.services.ingestion_service import IngestionError, ingest  # noqa: E402


def main() -> None:
    base = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=base.chunk_size)
    parser.add_argument("--overlap", type=int, default=base.chunk_overlap)
    parser.add_argument("--suffix", default=base.qdrant_collection_suffix)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--confirm", help="collection name, required with --recreate")
    parser.add_argument("--headers", action="store_true", default=base.contextual_headers,
                        help="embed contextual headers (Step 12 R2); default from CONTEXTUAL_HEADERS")
    args = parser.parse_args()

    # One validated settings object for chunker and collection name (model_copy would skip validation)
    settings = Settings(**{**base.model_dump(), "chunk_size": args.size, "chunk_overlap": args.overlap,
                           "qdrant_collection_suffix": args.suffix, "contextual_headers": args.headers})
    print(f"Ingesting into {settings.collection_name} (chunking {settings.chunking}, "
          f"contextual headers {'on' if settings.contextual_headers else 'off'}) ...")

    try:
        report = ingest(create_qdrant_client(settings), create_embedding_service(settings), settings,
                        recreate=args.recreate, confirm=args.confirm)
    except IngestionError as e:
        sys.exit(f"FAILED ({type(e).__name__}): {e}")
    print(json.dumps(report.model_dump(), indent=2))
    sys.exit(0 if report.status == "success" else 1)


if __name__ == "__main__":
    main()
