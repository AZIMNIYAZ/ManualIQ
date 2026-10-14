"""Finds brand names in a question, so search can filter to that manual.

Deliberately simple and transparent: a word list from the manifest (brand IDs plus
each entry's aliases), no AI. Query rewriting (Step 9) handles follow-ups such as
"what about on Netgear?" and brands carried over from earlier messages.

Rules:
- case-insensitive, whole words ("link" alone never matches D-Link)
- URLs and domain names are removed first ("https://example.com/D-Link" is not a brand mention)
- longer aliases are matched first ("Archer AX55" before "Archer"), and a matched span is
  consumed so it cannot match again
- results are deduplicated and sorted ("Netgear Nighthawk RAX50" -> ["netgear"])
"""

import re
from functools import lru_cache

from app.services.document_loader import load_manifest

URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|cn|io|local|co|info|biz)\b(?:/\S*)?",
    re.IGNORECASE,
)

AliasTable = tuple[tuple[re.Pattern, str], ...]


def build_alias_table(aliases_by_brand: dict[str, list[str]]) -> AliasTable:
    """Compiled (pattern, brand) pairs, longest alias first."""
    pairs = []
    for brand, aliases in aliases_by_brand.items():
        for alias in {brand, *aliases}:
            words = [re.escape(w) for w in alias.split()]
            pattern = re.compile(r"(?<![\w-])" + r"\s+".join(words) + r"(?![\w-])", re.IGNORECASE)
            pairs.append((len(alias), alias.lower(), pattern, brand))
    pairs.sort(key=lambda p: (-p[0], p[1]))  # longest first; alphabetical for a stable order
    return tuple((pattern, brand) for _, _, pattern, brand in pairs)


@lru_cache
def alias_table_from_manifest(documents_dir: str) -> AliasTable:
    aliases: dict[str, list[str]] = {}
    for entry in load_manifest(documents_dir):
        aliases.setdefault(entry.brand, []).extend(entry.aliases)
    return build_alias_table(aliases)


def detect_brands(query: str, table: AliasTable) -> list[str]:
    text = URL_RE.sub(" ", query)
    found = set()
    for pattern, brand in table:
        text, n = pattern.subn(" ", text)  # consume the match so a shorter alias cannot reuse it
        if n:
            found.add(brand)
    return sorted(found)
