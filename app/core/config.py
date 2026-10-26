"""Application settings, loaded from environment variables and .env.

Values are validated when the app starts, so a bad .env stops the app
immediately instead of producing silently wrong results later.
"""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# bge-small-en-v1.5 produces 384-dim vectors and reads at most 512 tokens.
BGE_SMALL_DIM = 384
MAX_CHUNK_TOKENS = 500  # margin below 512 for the special tokens bge adds


class Settings(BaseSettings):
    # env_ignore_empty: "LLM_PRICE_INPUT_PER_M=" (left blank) means "not set", not an invalid number
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    app_name: str = "RAG Backend"
    log_level: str = "INFO"
    # JSON-lines log file for analysis (scripts/log_report.py); "none" or unset = console only.
    # Only the API process writes it (Windows cannot rotate a file another process has open).
    log_file: str | None = None
    # Privacy: questions and answers are logged as lengths unless this is enabled
    log_user_text: bool = False
    slow_request_ms: float = Field(15000, gt=0)  # requests slower than this are logged as warnings

    # xAI / Grok: optional until generation is built (Step 8)
    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"
    xai_chat_model: str | None = None
    llm_temperature: float = Field(0.1, ge=0, le=2)  # low: consistent, factual answers
    llm_max_output_tokens: int = Field(800, gt=0)
    llm_timeout_seconds: float = Field(30, gt=0)
    llm_max_retries: int = Field(2, ge=0)  # automatic retries for rate limits / network errors
    # Query rewriting for follow-up questions: a small task, so a fast non-reasoning model
    # (measured 1.0 s vs 5.1 s for grok-4.3); defaults to XAI_CHAT_MODEL if not set
    xai_rewrite_model: str | None = None
    rewrite_max_output_tokens: int = Field(200, gt=0)
    rewrite_answer_chars: int = Field(300, gt=0)  # previous answers are shortened for the rewriter
    # Optional, from xAI's pricing page (USD per million tokens); without them only tokens are reported
    llm_price_input_per_m: float | None = Field(None, ge=0)
    llm_price_output_per_m: float | None = Field(None, ge=0)

    # Context sent to the LLM (counted with the bge tokenizer as an approximation)
    context_max_tokens: int = Field(2500, gt=0)

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Pinned model snapshot: vectors from different model versions must never be mixed
    embedding_model_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    embedding_dim: int = BGE_SMALL_DIM
    embedding_batch_size: int = Field(16, gt=0)  # measured fastest on CPU (16: 9.0, 32: 7.3, 64: 5.9 chunks/s)
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Chunking
    chunk_size: int = Field(350, gt=0, le=MAX_CHUNK_TOKENS)
    chunk_overlap: int = Field(60, ge=0)
    # Step 12 R2: embed "Brand Model | Section" + chunk text (stored text unchanged). Off = baseline.
    # A collection built with headers gets the suffix _ctx, so the two can never be mixed.
    contextual_headers: bool = False

    # Qdrant
    qdrant_url: str = "http://127.0.0.1:6333"
    # The collection name is derived from the chunk settings (see collection_name), so a
    # collection called rag_200_40 can only ever hold 200/40 chunks
    qdrant_collection_prefix: str = Field("rag", pattern=r"^[a-z][a-z0-9_]*$")
    qdrant_collection_suffix: str = Field("", pattern=r"^[a-z0-9_]*$")  # e.g. "ctx" for an experiment variant
    qdrant_timeout_seconds: int = Field(2, gt=0)  # local upserts of 128 points take milliseconds
    qdrant_upsert_batch_size: int = Field(128, gt=0)

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_timeout_seconds: float = Field(2.0, gt=0)
    session_key_prefix: str = Field("session", pattern=r"^[a-z_]+$")  # tests use another prefix
    session_ttl_seconds: int = Field(86400, gt=0)
    session_max_stored_messages: int = Field(50, gt=0)
    history_messages_in_prompt: int = Field(8, ge=0)

    # Retrieval
    top_k: int = Field(5, gt=0)

    # Paths
    documents_dir: str = "documents"

    @model_validator(mode="after")
    def check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than CHUNK_SIZE ({self.chunk_size})"
            )
        if self.embedding_dim != BGE_SMALL_DIM:
            raise ValueError(f"EMBEDDING_DIM must be {BGE_SMALL_DIM} for {self.embedding_model}")
        return self

    @property
    def collection_name(self) -> str:
        """rag_350_60; rag_350_60_ctx with contextual headers; plus QDRANT_COLLECTION_SUFFIX if set."""
        name = f"{self.qdrant_collection_prefix}_{self.chunk_size}_{self.chunk_overlap}"
        if self.contextual_headers:
            name += "_ctx"
        return f"{name}_{self.qdrant_collection_suffix}" if self.qdrant_collection_suffix else name

    @property
    def embedding_model_version(self) -> str:
        """As stored with every vector ("BAAI/bge-small-en-v1.5@5c38ec7"): model and revision."""
        return f"{self.embedding_model}@{self.embedding_model_revision[:7]}"

    @property
    def chunking(self) -> str:
        return f"{self.chunk_size}/{self.chunk_overlap}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
