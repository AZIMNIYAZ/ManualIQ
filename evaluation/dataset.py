"""The evaluation dataset: schema, loading and the text normalisation used for matching.

Labels are chunk-independent: evidence is (manual, PDF page range, key phrase), never a
chunk ID, so the same labels score every chunk size in Step 12. Pages are PDF page
positions (1 = first page of the file), the convention used by the whole application,
not the numbers printed in the manuals' footers.

A question can list several evidence locations: they are alternatives (any one of them
answers the question), so retrieval is scored with Hit@K, not Recall@K.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DATA_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = DATA_DIR / "questions.jsonl"
CONVERSATIONS_FILE = DATA_DIR / "conversations.jsonl"

Category = Literal["direct", "paraphrased", "no_brand", "tables", "context_confusion",
                   "unanswerable", "out_of_scope"]

# Categories run several times by default (answers vary between runs on these)
REPEAT_CATEGORIES = ("tables", "context_confusion")


class Evidence(BaseModel):
    document_id: str
    pages: tuple[int, int]  # PDF page range, inclusive
    key_phrase: str = Field(min_length=3)

    @model_validator(mode="after")
    def ordered(self):
        if self.pages[0] > self.pages[1]:
            raise ValueError(f"page range {self.pages} is reversed")
        return self


class Absence(BaseModel):
    """How 'not in the manuals' was checked: the terms searched (raw and cleaned text) and
    in which manuals. A safeguard, not proof: a fact can also exist only in an image."""

    terms: list[str] = Field(min_length=1)
    manuals: list[str] | Literal["all"] = "all"
    note: str = ""


# A key fact is a phrase the answer must contain, or a list of alternatives (any one counts)
KeyFact = str | list[str]


class Item(BaseModel):
    """One thing to ask: a standalone question, or one turn of a conversation."""

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    question: str
    brand: str | None = None  # brand the system should filter to (None: no filter expected)
    answerable: bool
    evidence: list[Evidence] = []
    key_facts: list[KeyFact] = []
    must_not_contain: list[str] = []  # text from a wrong procedure / wrong part of a table
    absence: Absence | None = None
    note: str = ""

    @model_validator(mode="after")
    def consistent(self):
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable items need evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable items cannot have evidence")
        return self


class Question(Item):
    category: Category
    policy: Literal["single_brand", "per_brand", "refuse"] = "single_brand"

    @model_validator(mode="after")
    def policy_matches(self):
        expected = ("refuse" if not self.answerable else
                    "per_brand" if self.category == "no_brand" else "single_brand")
        if self.policy != expected:
            raise ValueError(f"{self.id}: policy should be {expected!r}")
        if self.category == "unanswerable" and self.absence is None:
            raise ValueError(f"{self.id}: unanswerable questions must record how absence was checked")
        return self


class Turn(Item):
    # For follow-ups: the standalone question a correct rewrite would produce
    expected_query: str | None = None
    required_terms: list[KeyFact] = []  # must appear in the rewritten query


class Conversation(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    tests: str  # what the conversation checks
    turns: list[Turn] = Field(min_length=2)

    @model_validator(mode="after")
    def follow_ups_labelled(self):
        for t in self.turns[1:]:
            if not t.expected_query:
                raise ValueError(f"{t.id}: follow-up turns need expected_query")
        return self


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("//"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{n}: invalid JSON ({e})") from e
    return rows


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    return [Question(**row) for row in _read_jsonl(path)]


def load_conversations(path: Path = CONVERSATIONS_FILE) -> list[Conversation]:
    return [Conversation(**row) for row in _read_jsonl(path)]


def dataset_hash(*paths: Path) -> str:
    """Identifies the exact question set: Step 12 runs are only comparable on the same hash."""
    paths = paths or (QUESTIONS_FILE, CONVERSATIONS_FILE)
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))  # Git may check files out with CRLF on Windows
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ text matching

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalise(text: str) -> str:
    """Case, quotes, whitespace, markdown and 'Wi-Fi'/'WiFi' differences do not count."""
    text = text.translate(_QUOTES).casefold()
    text = re.sub(r"[*`_]{1,3}", "", text)  # markdown emphasis/code
    text = re.sub(r"\bwi-fi\b", "wifi", text)
    text = re.sub(r"\s*>\s*", " > ", text)  # menu paths: "A>B" == "A > B"
    return re.sub(r"\s+", " ", text).strip()


def contains(text: str, fact: KeyFact) -> bool:
    """True if `text` contains the phrase (or any of the alternatives)."""
    options = [fact] if isinstance(fact, str) else fact
    t = normalise(text)
    return any(normalise(o) in t for o in options)


def fact_label(fact: KeyFact) -> str:
    return fact if isinstance(fact, str) else " | ".join(fact)
