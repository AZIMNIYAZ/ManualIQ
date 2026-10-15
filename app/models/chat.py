from pydantic import BaseModel, Field, field_validator

from app.models.search import SearchFilters, SearchRequest


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20, description="Chunks retrieved; defaults to TOP_K")
    brand: str | list[str] | None = Field(None, description="As in /search: explicit filter, else detected")
    include_regulatory: bool = False
    debug: bool = Field(False, description="Also return the context sent to the LLM and all retrieved chunks")

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()

    def to_search(self) -> SearchRequest:
        return SearchRequest(query=self.message, top_k=self.top_k, brand=self.brand,
                             include_regulatory=self.include_regulatory)


class ChatSource(BaseModel):
    n: int  # the [n] used in the answer
    citation: str
    brand: str
    model: str
    section: str | None
    chunk_id: str
    score: float


class LlmInfo(BaseModel):
    model: str | None  # None when the LLM was not called (nothing retrieved)
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0  # the visible answer
    reasoning_tokens: int = 0  # hidden reasoning, reported separately; usually billed as output
    context_tokens: int = 0
    sources_dropped: int = 0  # retrieved chunks left out of the context by the token budget


class ChatResponse(BaseModel):
    answer: str
    found_in_documents: bool
    sources: list[ChatSource]  # only the sources the answer cites
    filters: SearchFilters
    warnings: list[dict]
    llm: LlmInfo
    timings_ms: dict[str, float]
    context: str | None = None  # debug only
    retrieved: list[ChatSource] | None = None  # debug only: every chunk sent to the LLM
