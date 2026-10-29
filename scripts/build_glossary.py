"""Step 12: builds documents/glossary.json, the manuals' feature names for the retrieval fallback.

Re-run whenever a manual or the cleaner changes:
    python scripts/build_glossary.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.services.manual_glossary import load_glossary, write_glossary  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    settings = get_settings()
    path = write_glossary(settings.documents_dir)
    glossary = load_glossary(settings.documents_dir)
    for brand, entries in glossary["manuals"].items():
        print(f"{brand:10} {len(entries):4} headings")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
