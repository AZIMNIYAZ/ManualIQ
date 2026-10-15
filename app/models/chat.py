from pydantic import BaseModel, Field, field_validator

from app.models.search import SearchFilters, SearchRequest
from app.models.session import SESSION_ID_PATTERN


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(None, pattern=SESSION_ID_PATTERN,
                                   description="Continue a conversation; omitted = a new one (the id is returned)")
    top_k: int | None = Field(None, ge=1, le=20, description="Chunks retrieved; defaults to TOP_K")
    brand: str | list[str] | None = Field(
        None, description="Per-request filter that always wins (never stored); otherwise brands come from "
                          "query rewriting (follow-ups) or are detected in the question")
    include_regulatory: bool = False
    debug: bool = Field(False, description="Also return the context sent to the LLM and all retrieved chunks")

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()

    def to_search(self, query: str, brand: list[str] | None) -> SearchRequest:
        return SearchRequest(query=query, top_k=self.top_k, brand=brand, include_regulatory=self.include_regulatory)


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


class RewriteInfo(BaseModel):
    model: str | None
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    found_in_documents: bool
    sources: list[ChatSource]  # only the sources the answer cites
    filters: SearchFilters
    warnings: list[dict]
    llm: LlmInfo
    rewritten_query: str | None = None  # the standalone question searched (None: the message itself)
    history_used: int = 0  # previous messages given to the rewriter (0: new or expired session)
    rewrite: RewriteInfo | None = None  # None when no rewrite call was made
    timings_ms: dict[str, float]
    context: str | None = None  # debug only
    retrieved: list[ChatSource] | None = None  # debug only: every chunk sent to the LLM
