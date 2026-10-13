from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ManifestEntry(BaseModel):
    """One entry in documents/manifest.json."""

    file: str
    document_id: str = Field(pattern=r"^[a-z0-9_]+$")
    brand: str = Field(pattern=r"^[a-z0-9_]+$")
    model: str
    source_file: str | None = None  # original download name, for reference only


class Manifest(BaseModel):
    documents: list[ManifestEntry]


PageStatus = Literal["pending", "indexed", "excluded"]
ContentType = Literal["normal", "regulatory"]
ExclusionReason = Literal["empty", "non_english", "toc"]


class Page(BaseModel):
    """One PDF page. raw_text is never modified; everything else is derived from it."""

    model_config = ConfigDict(frozen=True)

    page_number: int  # original 1-based PDF page number; never renumbered
    raw_text: str  # exactly what pypdf extracted
    clean_text: str | None = None  # set by the cleaner
    status: PageStatus = "pending"  # does this page go into Qdrant?
    content_type: ContentType = "normal"  # what kind of content (regulatory is filtered by default)
    reason: ExclusionReason | None = None  # why it was excluded (only when status == "excluded")
    flags: tuple[str, ...] = ()  # things a person should review, e.g. "language_uncertain"


class Document(BaseModel):
    document_id: str
    brand: str
    model: str
    source: str  # file name in documents/, shown in citations
    pages: list[Page]

    @property
    def text_length(self) -> int:
        return sum(len(p.raw_text) for p in self.pages)


class Repair(BaseModel):
    page: int
    kind: str
    original: str
    fixed: str


class PageNote(BaseModel):
    page: int
    label: str  # exclusion reason, content type or flag
    preview: str


class CleaningReport(BaseModel):
    document_id: str
    cleaner_version: str
    pages_total: int
    pages_indexed: int
    pages_excluded: int
    chars_raw: int
    chars_clean: int  # indexed pages only
    excluded: list[PageNote]
    regulatory: list[PageNote]
    flagged: list[PageNote]
    header_lines: dict[str, int]  # normalised header/footer line -> pages it was removed from
    repair_counts: dict[str, int]
    repairs: list[Repair]
