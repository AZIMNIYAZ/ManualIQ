from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20, description="Defaults to TOP_K from .env")
    brand: str | list[str] | None = Field(
        None, description="Filter to these brands; if omitted, brands named in the query are detected"
    )
    include_regulatory: bool = Field(False, description="Also search regulatory/legal chunks")

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()

    @field_validator("brand")
    @classmethod
    def normalise_brand(cls, v):
        if v is None:
            return None
        brands = [v] if isinstance(v, str) else v
        brands = sorted({b.strip().lower() for b in brands if b.strip()})
        if not brands:
            raise ValueError("brand must not be empty")
        return brands


class SearchFilters(BaseModel):
    brands: list[str]  # empty = all brands
    brand_source: Literal["request", "rewrite", "detected", "none"]
    include_regulatory: bool


class SearchResult(BaseModel):
    rank: int
    score: float  # cosine similarity; can range from -1 to 1 (bge scores here are ~0.3-0.9)
    chunk_id: str
    brand: str
    model: str
    source: str
    page_start: int
    page_end: int
    citation: str  # built from the page range, never from the section label
    section: str | None
    content_type: str
    text: str


class SearchResponse(BaseModel):
    query: str
    collection: str
    filters: SearchFilters
    query_tokens: int  # bge tokens incl. the query prefix and special tokens
    query_truncated: bool
    results: list[SearchResult]
    timings_ms: dict[str, float]
