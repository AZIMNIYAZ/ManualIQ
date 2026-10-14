"""Cleans loaded documents before chunking.

Two levels:
  A. Page decisions: each page is indexed or excluded (empty / non_english / toc),
     and tagged with a content_type (normal / regulatory).
  B. Text repairs on kept pages: Unicode, bullets, headers/footers, hyphenation,
     split words, whitespace.

raw_text is never modified; clean_text is always rebuilt from it, so cleaning
is deterministic. Every exclusion, flag and repair is recorded in a report.
"""

import re
import unicodedata
from collections import Counter
from functools import lru_cache

from app.models.document import CleaningReport, Document, Page, PageNote, Repair

# Bump when any rule changes, so experiments know whether collections are comparable.
CLEANER_VERSION = "1.2"  # 1.1: split words joined across line breaks; 1.2: "R outer’s" -> "Router’s"

# ---------------------------------------------------------------- thresholds
MIN_WORDS = 3  # fewer words than this (raw or after cleaning): page is "empty". Short pages are kept:
# they hold real steps and chapter titles, and continuous chunking merges them with neighbours.
MIN_WORDS_FOR_LANGUAGE = 40  # below this, Latin-script pages are never classified by language
NON_LATIN_SHARE = 0.5  # share of letters in non-Latin scripts (CJK, Cyrillic...) to call a page non-English
OTHER_LANGUAGE_MIN = 0.15  # another language's stopword share must be at least this...
OTHER_LANGUAGE_FACTOR = 2.0  # ...and this many times the English share, to exclude a page
DICTIONARY_SHARE_EXCLUDE = 0.55  # below this share of English dictionary words: non-English
DICTIONARY_SHARE_UNCERTAIN = 0.65  # below this: kept but flagged (lowest real English page: 69%)
TOC_MIN_LINES = 5  # dot-leader lines needed for the TOC "layout" signal...
TOC_MIN_SHARE = 0.25  # ...and their share of the page's lines
EDGE_PAGE_SHARE = 0.10  # first/last 10% of a manual count as "near the start/end"
REGULATORY_MIN_FAMILIES = 3  # distinct term families needed to tag a page regulatory...
REGULATORY_MIN_FAMILIES_AT_END = 2  # ...or this many in the last 10% of the manual
LEGAL_MIN_TERMS = 3  # or this many legal terms (copyright/trademark pages)
HEADER_EDGE_LINES = 3  # header/footer lines are searched in the top/bottom 3 lines of each page
WORD_MIN_COUNT = 3  # a fragment seen standalone this often in the manual counts as a real word
LETTER_JOIN_MIN_COUNT = 20  # single letter + word ("R outer") is joined only if the joined word is this common...
LETTER_JOIN_RATIO = 5  # ...and this many times more common than the second word on its own

# Lines that match the header/footer rule but were manually reviewed and are real content.
HEADER_KEEP = {
    "Connect or Disconnect",  # D-Link VPN section heading, repeated once per operating system
}
HEADER_MIN_PAGES = 5  # a header/footer line must repeat on at least this many pages...
HEADER_EDGE_SHARE = 0.9  # ...and sit at a page edge in at least 90% of its occurrences
PREVIEW_CHARS = 100

# ---------------------------------------------------------------- word lists
# Stopwords per language. Words that are also common in English text ("in", "is",
# "to", "a", "e", "die") are deliberately left out of the other languages.
STOPWORDS = {
    "en": {"the", "and", "of", "you", "your", "for", "on", "with", "this", "that", "are", "or",
           "click", "if", "it", "can", "by", "from", "as", "not", "be", "will", "select", "enter"},
    "de": {"der", "und", "ist", "nicht", "mit", "für", "sie", "den", "des", "dem", "ein", "eine",
           "zu", "auf", "werden", "dass", "oder", "wird", "sich", "diese"},
    "fr": {"le", "la", "les", "et", "des", "est", "pour", "vous", "du", "une", "dans", "que",
           "qui", "sur", "avec", "ce", "ne", "pas", "par", "aux", "cet", "être"},
    "es": {"el", "los", "las", "y", "que", "en", "para", "con", "por", "una", "es", "se", "del",
           "su", "este", "esta", "como", "sus"},
    "it": {"il", "di", "che", "per", "con", "non", "sono", "della", "del", "una", "gli", "questo",
           "dispositivo", "alla"},
    "nl": {"de", "het", "en", "van", "een", "dat", "voor", "met", "niet", "op", "zijn", "dit",
           "deze", "wordt"},
    "pt": {"o", "os", "que", "em", "para", "com", "por", "uma", "do", "da", "dos", "das", "este"},
    "cs": {"je", "se", "na", "že", "tento", "jsou", "pro", "které", "podle", "nebo"},
}

# Case-sensitive, whole-word regulatory and legal terms ("CE" must not match "device"),
# grouped into families: "FCC Part 15 Class B" is one piece of evidence, not three.
REGULATORY_FAMILIES = {
    "us_fcc": [r"FCC", r"Part 15", r"Class B"],
    "canada": [r"IC", r"ISED", r"Industry Canada", r"CNR"],
    "eu": [r"CE", r"Declaration of Conformity", r"RoHS", r"WEEE", r"[Rr]ecycl\w*", r"[Ww]aste [Ee]lectrical", r"[Rr]estricted [Ss]ubstances?"],
    "asia": [r"NCC", r"BSMI", r"KC"],
    "radio": [r"EIRP", r"[Rr]adiation [Ee]xposure", r"[Ii]nterference"],
    "safety": [r"Class II", r"[Ss]afety [Ii]nstructions", r"[Ss]afety [Ii]nformation"],
    "generic": [r"[Rr]egulatory", r"[Cc]ompliance"],
    "legal": [r"Copyright", r"[Aa]ll [Rr]ights [Rr]eserved", r"[Tt]rademarks?", r"[Ww]arranty", r"[Ll]iability"],
}
REGULATORY_RE = {
    family: [re.compile(rf"(?<![A-Za-z]){t}(?![A-Za-z])") for t in terms]
    for family, terms in REGULATORY_FAMILIES.items()
}

BULLET_CHARS = "\u2022\u26ab\u25cf\u25aa\u25a0\u25e6\u25cb\uf09f\uf0b7\uf0a7\uf0d8\u2212\u2013"  # bullets, dots, squares, circle, private-use bullets, minus, en dash
BULLET_RE = re.compile(rf"^[ \t]*[{BULLET_CHARS}][ \t]*(?=[^\s{BULLET_CHARS}])", re.M)
# Private-use glyphs that carry meaning are replaced with words, not deleted.
PRIVATE_USE_MAP = {"\uf000": "Apple menu"}  # D-Link macOS steps: "Go to \uf000 > System Preferences"
DOT_LEADER_RE = re.compile(r"(?:\.\s?){4,}\s*[\divxlc]{0,4}\s*$", re.I)
TOC_HEADING_RE = re.compile(r"^\s*(table of\s+)?contents\b|^\s*index\s*$", re.I | re.M)
HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]{2,})-[ \t]*\n[ \t]*([a-z]{2,})")
LETTER_SPACED_RE = re.compile(r"(?<![A-Za-z])((?:[A-Za-z] ){3,}[A-Za-z])(?![A-Za-z])")
KERNING_RE = re.compile(r"(?<![\w.'\u2019-])([TYVWFP]) ([a-z]{2,})(?![\w-])")
SPLIT_PAIR_RE = re.compile(r"(?<![\w.'\u2019-])([A-Za-z]+) ([a-z]+)(?![\w-])")
LINEBREAK_SPLIT_RE = re.compile(r"(?<![\w.'’-])([A-Za-z]+)\n([a-z]+)(?![\w-])")
# (apostrophe before the first fragment blocks a join: "router’s port" must not become "sport";
#  one after the second fragment does not: "R outer’s" -> "Router’s")
LATIN_WORD_RE = re.compile(r"[^\W\d_]+")
INVISIBLE_RE = re.compile("[\u00ad\u200b\u200c\u200d\ufeff]")  # soft hyphen, zero-width chars, BOM


@lru_cache
def english_dictionary() -> frozenset[str]:
    """Whole English words from bge's tokenizer vocabulary (~21k words)."""
    from app.services.tokenizer import get_tokenizer

    vocab = get_tokenizer().get_vocab()
    return frozenset(w for w in vocab if w.isascii() and w.isalpha())


# ---------------------------------------------------------------- helpers
def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE_RE.sub("", text)
    for glyph, word in PRIVATE_USE_MAP.items():
        text = text.replace(glyph, word)
    return text.replace("\u2212", "-")  # minus sign


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _header_key(line: str) -> str:
    """Normalises a line so "User Manual 45" and "User Manual 46" compare equal."""
    line = " ".join(line.split())
    line = re.sub(r"^[ivxlc]{1,6}(?=[A-Z])", "#", line)  # roman page numbers glued to the header
    return re.sub(r"\d+", "#", line)


def _preview(text: str) -> str:
    return " ".join(text.split())[:PREVIEW_CHARS]


def _is_private_use(ch: str) -> bool:
    return 0xE000 <= ord(ch) <= 0xF8FF


# ---------------------------------------------------------------- A. page decisions
def _language(text: str, dictionary: frozenset[str]) -> tuple[str, str | None]:
    """Returns ("english" | "non_english" | "uncertain" | "unknown", detail)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 30:
        non_latin = sum(1 for c in letters if not ("LATIN" in unicodedata.name(c, "")))
        if non_latin / len(letters) > NON_LATIN_SHARE:
            return "non_english", "non-Latin script"

    words = [w.lower() for w in LATIN_WORD_RE.findall(text)]
    if len(words) < MIN_WORDS_FOR_LANGUAGE:
        return "unknown", None

    # Signal 1: a known language's common words clearly outnumber English ones
    shares = {lang: sum(w in sw for w in words) / len(words) for lang, sw in STOPWORDS.items()}
    english = shares.pop("en")
    other_lang, other = max(sorted(shares.items()), key=lambda kv: kv[1])
    if other >= OTHER_LANGUAGE_MIN and other > OTHER_LANGUAGE_FACTOR * english:
        return "non_english", f"{other_lang} {other:.0%} vs en {english:.0%}"

    # Signal 2: share of words found in an English dictionary. Catches languages without
    # a stopword list (Latvian, Polish...) and pages mixing many languages; English tables
    # still score high because their terms are English words.
    long_words = [w for w in words if len(w) >= 2]
    in_dictionary = sum(w in dictionary for w in long_words) / max(len(long_words), 1)
    if in_dictionary < DICTIONARY_SHARE_EXCLUDE:
        return "non_english", f"{in_dictionary:.0%} dictionary words"
    if in_dictionary < DICTIONARY_SHARE_UNCERTAIN or other > english:
        return "uncertain", f"{in_dictionary:.0%} dictionary words, {other_lang} {other:.0%} vs en {english:.0%}"
    return "english", None


def _toc_signals(text: str, page_number: int, total_pages: int) -> tuple[bool, bool]:
    lines = _lines(text)
    leader_lines = sum(1 for line in lines if DOT_LEADER_RE.search(line))
    layout = leader_lines >= TOC_MIN_LINES and leader_lines / max(len(lines), 1) >= TOC_MIN_SHARE
    edge = max(3, round(total_pages * EDGE_PAGE_SHARE))
    near_edge = page_number <= edge or page_number > total_pages - edge
    heading = bool(TOC_HEADING_RE.search("\n".join(lines[:5])))
    return layout, heading or near_edge


def _regulatory_families(text: str) -> set[str]:
    return {f for f, patterns in REGULATORY_RE.items() if any(p.search(text) for p in patterns)}


def _is_regulatory(text: str, near_end: bool) -> bool:
    families = _regulatory_families(text)
    legal_terms = sum(1 for p in REGULATORY_RE["legal"] if p.search(text))
    needed = REGULATORY_MIN_FAMILIES_AT_END if near_end else REGULATORY_MIN_FAMILIES
    return len(families) >= needed or legal_terms >= LEGAL_MIN_TERMS


def _classify(text: str, page_number: int, total_pages: int, dictionary: frozenset[str]) -> dict:
    """Decides status / content_type / reason / flags for one page (from normalised raw text)."""
    flags: list[str] = []
    if len(text.split()) < MIN_WORDS:
        return {"status": "excluded", "reason": "empty", "flags": ()}

    language, _ = _language(text, dictionary)
    if language == "non_english":
        return {"status": "excluded", "reason": "non_english", "flags": ()}
    if language == "uncertain":
        flags.append("language_uncertain")

    layout, position = _toc_signals(text, page_number, total_pages)
    if layout and position:
        return {"status": "excluded", "reason": "toc", "flags": tuple(flags)}
    if layout:
        flags.append("toc_single_signal")

    near_end = page_number > total_pages - max(3, round(total_pages * EDGE_PAGE_SHARE))
    is_regulatory = _is_regulatory(text, near_end)
    if language == "uncertain" and _regulatory_families(text):
        is_regulatory = True  # mixed-language compliance pages: keep, but filter by default
    content_type = "regulatory" if is_regulatory else "normal"
    return {"status": "indexed", "content_type": content_type, "flags": tuple(flags)}


# ---------------------------------------------------------------- B. text repairs
def _find_header_lines(texts: list[str]) -> set[str]:
    edge, anywhere = Counter(), Counter()
    for text in texts:
        lines = _lines(text)
        for i, line in enumerate(lines):
            key = _header_key(line)
            anywhere[key] += 1
            if i < HEADER_EDGE_LINES or i >= len(lines) - HEADER_EDGE_LINES:
                edge[key] += 1
    return {
        key for key in sorted(edge)
        if edge[key] >= HEADER_MIN_PAGES and edge[key] / anywhere[key] >= HEADER_EDGE_SHARE
        and key not in HEADER_KEEP
    }


def _strip_headers(text: str, header_keys: set[str], removed: Counter) -> str:
    """Removes header/footer lines, but only as a continuous block at the very top or
    bottom of the page: a matching line with real content below it is kept."""
    lines = _lines(text)
    start, end = 0, len(lines)
    while start < min(HEADER_EDGE_LINES, end) and _header_key(lines[start]) in header_keys:
        removed[_header_key(lines[start])] += 1
        start += 1
    while end > max(start, len(lines) - HEADER_EDGE_LINES) and _header_key(lines[end - 1]) in header_keys:
        removed[_header_key(lines[end - 1])] += 1
        end -= 1
    return "\n".join(lines[start:end])


class _Repairer:
    """Applies the word-level repairs to one document, logging every change."""

    def __init__(self, raw_texts: list[str], dictionary: frozenset[str]):
        self.dictionary = dictionary
        self.vocab = Counter(w.lower() for t in raw_texts for w in re.findall(r"[A-Za-z]+", t))
        self.repairs: list[Repair] = []
        self.page = 0

    def _known(self, word: str) -> bool:
        """The joined form exists: unsplit somewhere in this manual, or in the dictionary."""
        w = word.lower()
        return self.vocab[w] > 0 or w in self.dictionary

    def _is_word(self, fragment: str) -> bool:
        """A real word: in the dictionary, or used standalone often in this manual ("router")."""
        f = fragment.lower()
        if len(f) == 1:
            return f in ("a", "i")
        return f in self.dictionary or self.vocab[f] >= WORD_MIN_COUNT

    def _log(self, kind: str, original: str, fixed: str) -> None:
        self.repairs.append(Repair(page=self.page, kind=kind, original=original, fixed=fixed))

    def bullets(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            self._log("bullet", m.group(0).strip(), "-")
            return "- "
        return BULLET_RE.sub(sub, text)

    def hyphenation(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            a, b = m.group(1), m.group(2)
            fixed = a + b if self._known(a + b) else f"{a}-{b}"
            self._log("hyphenation", m.group(0).replace("\n", "⏎"), fixed)
            return fixed
        return HYPHEN_BREAK_RE.sub(sub, text)

    def letter_spacing(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            joined = m.group(1).replace(" ", "")
            if not self._known(joined):
                return m.group(0)
            self._log("letter_spacing", m.group(1), joined)
            return joined
        return LETTER_SPACED_RE.sub(sub, text)

    def kerning(self, text: str) -> str:
        """'T raceroute' -> 'Traceroute': a kerning-prone capital split off its word.

        Joined when the result is a known word, or when the rest is 3+ letters
        (a lone "T" before a word is never correct). The second case is logged
        separately as unverified, for review.
        """
        def sub(m: re.Match) -> str:
            joined = m.group(1) + m.group(2)
            if self._known(joined):
                kind = "kerning"
            elif len(m.group(2)) >= 3:
                kind = "kerning_unverified"
            else:
                return m.group(0)  # e.g. "P is"
            self._log(kind, m.group(0), joined)
            return joined
        return KERNING_RE.sub(sub, text)

    def _should_join(self, a: str, b: str) -> bool:
        if self._is_word(a) and self._is_word(b):
            return False  # "set up", "log in", "the router" stay as they are
        if len(a) == 1 and self._is_word(b):
            # "R outer" -> "Router" ("router" x356 vs "outer" x1), but "class B and class C"
            # stays ("band" x3 vs "and" x441): join only if the joined word clearly dominates.
            joined, rest = self.vocab[(a + b).lower()], self.vocab[b.lower()]
            return joined >= LETTER_JOIN_MIN_COUNT and joined >= LETTER_JOIN_RATIO * rest
        if not self._known(a + b):
            return False
        if len(b) == 1 and self.vocab[(a + b).lower()] < 2:
            return False  # "the e" stays: a single trailing letter needs the joined word in this manual
        return True

    def split_words(self, text: str) -> str:
        """'selec t' -> 'select': join only if a fragment is not a word and the result is known."""
        def sub(m: re.Match) -> str:
            a, b = m.group(1), m.group(2)
            if not self._should_join(a, b):
                return m.group(0)
            self._log("split_word", m.group(0), a + b)
            return a + b
        return "\n".join(SPLIT_PAIR_RE.sub(sub, line) for line in text.split("\n"))

    def split_words_across_lines(self, text: str) -> str:
        """'Mak' at a line end + 'e sure' on the next line -> 'Make sure' (same rules as split_words)."""
        def sub(m: re.Match) -> str:
            a, b = m.group(1), m.group(2)
            if not self._should_join(a, b):
                return m.group(0)
            self._log("split_word_linebreak", m.group(0).replace("\n", "⏎"), a + b)
            return a + b
        return LINEBREAK_SPLIT_RE.sub(sub, text)


def _tidy_whitespace(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------- entry point
def clean_document(doc: Document, dictionary: frozenset[str] | None = None) -> tuple[Document, CleaningReport]:
    """Returns a new Document with clean_text/status/content_type/reason/flags set, plus a report."""
    dictionary = english_dictionary() if dictionary is None else dictionary
    total = len(doc.pages)
    normalized = [_normalize(p.raw_text) for p in doc.pages]
    header_keys = _find_header_lines(normalized)
    repairer = _Repairer(normalized, dictionary)
    headers_removed: Counter = Counter()

    pages: list[Page] = []
    excluded, regulatory, flagged = [], [], []
    for page, text in zip(doc.pages, normalized):
        decision = _classify(text, page.page_number, total, dictionary)
        preview = _preview(text)
        if decision["status"] == "excluded":
            excluded.append(PageNote(page=page.page_number, label=decision["reason"], preview=preview))
            pages.append(page.model_copy(update={**decision, "clean_text": None}))
            continue

        repairer.page = page.page_number
        clean = _strip_headers(text, header_keys, headers_removed)
        clean = repairer.bullets(clean)
        clean = repairer.hyphenation(clean)
        clean = repairer.letter_spacing(clean)
        clean = repairer.kerning(clean)
        clean = repairer.split_words(clean)
        clean = repairer.split_words_across_lines(clean)
        clean = _tidy_whitespace(clean)

        if len(clean.split()) < MIN_WORDS:  # e.g. only headers/footers were left on the page
            decision = {"status": "excluded", "reason": "empty", "flags": ()}
            excluded.append(PageNote(page=page.page_number, label="empty", preview=preview))
            pages.append(page.model_copy(update={**decision, "clean_text": None}))
            continue

        if decision["content_type"] == "regulatory":
            regulatory.append(PageNote(page=page.page_number, label="regulatory", preview=preview))
        for flag in decision["flags"]:
            flagged.append(PageNote(page=page.page_number, label=flag, preview=preview))
        pages.append(page.model_copy(update={**decision, "clean_text": clean}))

    cleaned = doc.model_copy(update={"pages": pages})
    indexed = [p for p in pages if p.status == "indexed"]
    private_left = sum(1 for p in indexed for c in p.clean_text if _is_private_use(c))
    repair_counts = dict(sorted(Counter(r.kind for r in repairer.repairs).items()))
    repair_counts["header_lines_removed"] = sum(headers_removed.values())
    repair_counts["private_use_chars_left"] = private_left

    report = CleaningReport(
        document_id=doc.document_id,
        cleaner_version=CLEANER_VERSION,
        pages_total=total,
        pages_indexed=len(indexed),
        pages_excluded=len(excluded),
        chars_raw=doc.text_length,
        chars_clean=sum(len(p.clean_text) for p in indexed),
        excluded=excluded,
        regulatory=regulatory,
        flagged=flagged,
        header_lines=dict(sorted(headers_removed.items(), key=lambda kv: (-kv[1], kv[0]))),
        repair_counts=repair_counts,
        repairs=repairer.repairs,
    )
    return cleaned, report


def clean_documents(docs: list[Document]) -> tuple[list[Document], list[CleaningReport]]:
    results = [clean_document(d) for d in docs]
    return [d for d, _ in results], [r for _, r in results]
