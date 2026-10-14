"""Turns text into vectors with the local bge model.

- embed_documents(): chunks, no prefix (bge model card: "no instruction needs to be
  added to passages").
- embed_query(): user questions, with bge's query instruction
  "Represent this sentence for searching relevant passages: ".

Vectors are L2-normalised, so cosine similarity is a dot product. Cosine scores can
range from -1 to 1; thresholds must come from measured scores, not an assumed range.

The interface lets another model replace bge later. Changing the model (or its
revision) means every chunk must be re-embedded: vectors from different models are
not comparable.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


@dataclass(frozen=True)
class QueryEmbedding:
    vector: np.ndarray
    token_count: int  # including the query prefix and bge's [CLS]/[SEP] tokens
    truncated: bool  # True if the model only read the first max_seq_length tokens


class EmbeddingService(Protocol):
    model_name: str
    model_version: str  # stored with each vector so models are never mixed
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> QueryEmbedding: ...


class BgeEmbeddingService:
    def __init__(
        self,
        model_name: str,
        revision: str,
        dim: int,
        query_prefix: str,
        batch_size: int,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer  # heavy import, only when used

        try:
            self.model = SentenceTransformer(model_name, revision=revision, device=device, local_files_only=True)
        except Exception as e:
            raise EmbeddingError(
                f"Could not load embedding model {model_name}@{revision[:7]} from the local cache: {e}"
            ) from e

        actual_dim = self.model.get_embedding_dimension()
        if actual_dim != dim:
            raise EmbeddingError(f"{model_name} produces {actual_dim}-dim vectors, but EMBEDDING_DIM is {dim}")

        self.model_name = model_name
        self.model_version = f"{model_name}@{revision[:7]}"
        self.dim = dim
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.max_seq_length = self.model.max_seq_length  # 512 for bge-small-en-v1.5

    def _token_count(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if vectors.shape != (len(texts), self.dim):
            raise EmbeddingError(f"Expected {(len(texts), self.dim)} vectors, got {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding produced NaN or infinite values")
        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """One normalised vector per text, in input order. No query prefix."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        too_long = [i for i, t in enumerate(texts) if self._token_count(t) > self.max_seq_length]
        if too_long:  # chunks are capped at 500 tokens, so this means something upstream changed
            logger.warning(
                "Documents longer than the model reads were truncated",
                extra={"fields": {"count": len(too_long), "first_index": too_long[0]}},
            )
        return self._encode(texts)

    def embed_query(self, text: str) -> QueryEmbedding:
        full = self.query_prefix + text
        token_count = self._token_count(full)
        truncated = token_count > self.max_seq_length
        if truncated:
            logger.warning(
                "Query longer than the model reads; only the first part is used",
                extra={"fields": {"token_count": token_count, "max_seq_length": self.max_seq_length}},
            )
        return QueryEmbedding(vector=self._encode([full])[0], token_count=token_count, truncated=truncated)


def create_embedding_service(settings: Settings) -> BgeEmbeddingService:
    return BgeEmbeddingService(
        model_name=settings.embedding_model,
        revision=settings.embedding_model_revision,
        dim=settings.embedding_dim,
        query_prefix=settings.embedding_query_prefix,
        batch_size=settings.embedding_batch_size,
    )


@lru_cache
def get_embedding_service() -> BgeEmbeddingService:
    """The shared service: the model is loaded once per process."""
    return create_embedding_service(get_settings())
