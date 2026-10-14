from pydantic import BaseModel, ConfigDict

from app.models.document import ContentType


class Chunk(BaseModel):
    """A piece of a manual that is embedded and stored as one Qdrant point."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str  # "{document_id}_{chunk_index:04d}", deterministic
    document_id: str
    brand: str
    model: str
    source: str  # file name, for citations
    chunk_index: int  # position within the document
    text: str
    token_count: int  # bge tokens, never more than the chunk size
    page_start: int  # original PDF page numbers, cited as "p. 51-52"
    page_end: int
    content_type: ContentType  # regulatory chunks are filtered out by default
    section: str | None  # nearest heading or procedure title before the chunk (approximate)
    cleaner_version: str
    chunking: str  # "350/60" = chunk size / overlap, so experiments stay comparable


class ChunkingReport(BaseModel):
    chunk_size: int
    chunk_overlap: int
    documents: int
    segments: int
    chunks: int
    chunks_per_document: dict[str, int]
    dropped_too_small: int
    tail_merges: int
    oversized_units_split: int
    token_min: int
    token_median: int
    token_max: int
    multi_page_chunks: int
    ending_with_title: int  # chunks whose last unit is a heading or "To do X:" (should be ~0)
    avg_overlap_tokens: float
    cut_types: dict[str, int]  # what each chunk was cut before: heading / procedure / step / sentence / ...
