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
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Backend"
    log_level: str = "INFO"

    # xAI / Grok: optional until generation is built (Step 8)
    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"
    xai_chat_model: str | None = None

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = BGE_SMALL_DIM
    embedding_batch_size: int = Field(64, gt=0)
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Chunking
    chunk_size: int = Field(350, gt=0, le=MAX_CHUNK_TOKENS)
    chunk_overlap: int = Field(60, ge=0)

    # Qdrant
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "rag_350_60"
    qdrant_timeout_seconds: int = Field(2, gt=0)

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_timeout_seconds: float = Field(2.0, gt=0)
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
