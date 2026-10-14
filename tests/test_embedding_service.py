import os

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.services.embedding_service import (  # noqa: E402
    BgeEmbeddingService,
    EmbeddingError,
    get_embedding_service,
)


@pytest.fixture(scope="module")
def service():
    return get_embedding_service()


TEXTS = [f"Step {i}: open the router settings page and change option number {i}." for i in range(33)]


def test_dimensions_unit_length_and_finite(service):
    vectors = service.embed_documents(TEXTS[:5])
    assert vectors.shape == (5, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.isfinite(vectors).all()


def test_order_kept_across_partial_batch(service):
    """33 texts with batch size 16 = 16 + 16 + 1: nothing dropped, duplicated or reordered."""
    assert service.batch_size == 16
    batched = service.embed_documents(TEXTS)
    assert batched.shape == (33, 384)
    singles = np.vstack([service.embed_documents([t]) for t in TEXTS])
    assert np.allclose(batched, singles, atol=1e-5)
    # each row is closest to its own text's vector
    assert (np.argmax(batched @ singles.T, axis=1) == np.arange(33)).all()


def test_query_prefix_applied_only_to_queries(service):
    q = "how do I reset my router password"
    query = service.embed_query(q)
    as_doc_with_prefix = service.embed_documents([service.query_prefix + q])[0]
    as_doc_without = service.embed_documents([q])[0]
    assert np.allclose(query.vector, as_doc_with_prefix, atol=1e-5)
    assert not np.allclose(query.vector, as_doc_without, atol=1e-3)
    assert query.token_count > len(q.split()) and not query.truncated


def test_meaning(service):
    q = service.embed_query("reset the router admin password").vector
    related, unrelated = service.embed_documents(
        ["To change the admin password, go to System Tools > Administration.", "Mount the antenna on the back panel."]
    )
    assert q @ related > q @ unrelated


def test_long_query_is_flagged_as_truncated(service):
    result = service.embed_query("router " * 600)
    assert result.truncated and result.token_count > service.max_seq_length
    assert result.vector.shape == (384,)


def test_empty_input(service):
    assert service.embed_documents([]).shape == (0, 384)


def test_deterministic(service):
    assert np.array_equal(service.embed_documents(TEXTS[:3]), service.embed_documents(TEXTS[:3]))


def test_wrong_dimension_fails_at_startup():
    s = get_settings()
    with pytest.raises(EmbeddingError, match="384-dim"):
        BgeEmbeddingService(s.embedding_model, s.embedding_model_revision, dim=768,
                            query_prefix=s.embedding_query_prefix, batch_size=16)


def test_model_version_recorded(service):
    assert service.model_version == "BAAI/bge-small-en-v1.5@5c38ec7"
