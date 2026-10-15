"""Checks the xAI setup without spending tokens: lists the models your API key can
use and confirms XAI_CHAT_MODEL is one of them. The key itself is never printed.

Run from the project root:  python scripts/llm_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402


def main() -> None:
    s = get_settings()
    if not s.xai_api_key or not s.xai_chat_model:
        sys.exit("XAI_API_KEY and XAI_CHAT_MODEL must both be set in .env")

    client = OpenAI(api_key=s.xai_api_key, base_url=s.xai_base_url, timeout=15, max_retries=0)
    try:
        models = sorted(m.id for m in client.models.list())
    except Exception as e:
        sys.exit(f"Could not list models ({type(e).__name__}): {e}")

    print(f"Models available to this key ({len(models)}):")
    for m in models:
        print(f"  {'->' if m == s.xai_chat_model else '  '} {m}")
    if s.xai_chat_model in models:
        print(f"\nOK: XAI_CHAT_MODEL={s.xai_chat_model!r} is available.")
    else:
        sys.exit(f"\nXAI_CHAT_MODEL={s.xai_chat_model!r} is NOT in the list above: use one of the listed IDs.")


if __name__ == "__main__":
    main()
