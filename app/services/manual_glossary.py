"""Per-manual glossary of feature names (Step 12, retrieval fallback).

Built from each manual's table of contents, so the names are the manual's own wording
("Add a client to the blacklist", "Wireless MAC Filter", "QoS Engine"). The fallback gives
it to the LLM that rewrites a refused question, so "kick it off for good" can become a
search in the manual's vocabulary. Built from every heading of every manual: nothing is
picked for particular questions.

    {"tenda": ["Get to know your device", "Product overview", ...], ...}
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from app.models.document import Document

GLOSSARY_VERSION = "1.0"
GLOSSARY_FILE = "glossary.json"

LEADER_RE = re.compile(r"(\s*\.\s*){2,}.*$|\s+\d+\s*$")  # dot leaders and page numbers
NUMBERING_RE = re.compile(r"^(chapter\s+)?\d+(\s*\.\s*\d+)*\.?\s+", re.I)  # "Chapter 4", "3.6.2", "1. 2. 1."
LETTER_RE = re.compile(r"^[A-Z]\.\s+")  # "A. Wired connection"
SKIP_RE = re.compile(r"(table of )?contents|[ivx]+|.*user manual.*", re.I)
CONTINUATION_RE = re.compile(r"^[a-z][a-z]")  # "allow or block connected devices", not "iOS Devices"


def clean_entry(line: str) -> str | None:
    line = LEADER_RE.sub("", line).strip()
    line = LETTER_RE.sub("", NUMBERING_RE.sub("", line)).strip()
    line = re.sub(r"\bY our\b", "Your", line)  # TP-Link's PDF splits "Y our", "T op"
    line = re.sub(r"\bT (?=[a-z])", "T", line)
    if len(re.findall(r"[A-Za-z]", line)) < 3 or SKIP_RE.fullmatch(line):
        return None
    return line


def glossary_entries(doc: Document) -> list[str]:
    """The cleaned table-of-contents headings of one cleaned document, in order, without repeats."""
    out: list[str] = []
    for page in doc.pages:
        if page.reason != "toc":
            continue
        for line in page.raw_text.splitlines():
            entry = clean_entry(line)
            if entry and out and CONTINUATION_RE.match(entry):  # a heading wrapped onto the next line
                out[-1] = f"{out[-1]} {entry}"
            elif entry and entry not in out:
                out.append(entry)
    return out


def build_glossary(docs: list[Document]) -> dict[str, list[str]]:
    """brand -> headings, from cleaned documents."""
    return {doc.brand: glossary_entries(doc) for doc in docs}


def write_glossary(documents_dir: str | Path) -> Path:
    """Reading and cleaning the manuals takes ~25 s, too slow for a request, so the glossary is
    built once (scripts/build_glossary.py) and saved next to the manuals."""
    from app.services.document_loader import load_all_documents, load_manifest
    from app.services.text_cleaner import CLEANER_VERSION, clean_documents
    cleaned, _ = clean_documents(load_all_documents(documents_dir))
    names = {m.brand: f"{m.aliases[0] if m.aliases else m.brand} {m.model}" for m in load_manifest(documents_dir)}
    path = Path(documents_dir) / GLOSSARY_FILE
    data = {"glossary_version": GLOSSARY_VERSION, "cleaner_version": CLEANER_VERSION, "names": names,
            "manuals": build_glossary(cleaned)}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@lru_cache
def load_glossary(documents_dir: str) -> dict:
    """{"names": {brand: "Tenda AX3000 ..."}, "manuals": {brand: [headings]}, ...versions}"""
    path = Path(documents_dir) / GLOSSARY_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run python scripts/build_glossary.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("glossary_version") != GLOSSARY_VERSION:
        raise ValueError(f"{path} is version {data.get('glossary_version')}, expected {GLOSSARY_VERSION}: "
                         "run python scripts/build_glossary.py")
    return data


def glossary_text(glossary: dict[str, list[str]], names: dict[str, str], brands: list[str] | None = None) -> str:
    """One line per manual: 'Tenda AX3000: Product overview; Add a client to the blacklist; ...'.
    brands limits it to the manuals of the detected brands (all manuals when empty)."""
    keep = [b for b in glossary if not brands or b in brands]
    return "\n".join(f"{names.get(b, b)}: " + "; ".join(glossary[b]) for b in keep)
