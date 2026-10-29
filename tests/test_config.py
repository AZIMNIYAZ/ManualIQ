import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make_settings(**overrides) -> Settings:
    # _env_file=None: test the rules themselves, independent of your local .env
    return Settings(_env_file=None, **overrides)


def test_defaults_are_valid():
    s = make_settings()
    assert s.chunk_size == 500 and s.chunk_overlap == 60 and s.embedding_dim == 384
    assert s.contextual_headers and s.retrieval_fallback  # the Step 12 final setup


def test_chunk_size_above_500_rejected():
    with pytest.raises(ValidationError, match="less than or equal to 500"):
        make_settings(chunk_size=600)


def test_overlap_equal_to_chunk_size_rejected():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=350, chunk_overlap=350)


def test_overlap_larger_than_chunk_size_rejected():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=200, chunk_overlap=300)


def test_zero_overlap_allowed():
    assert make_settings(chunk_overlap=0).chunk_overlap == 0


def test_wrong_embedding_dim_rejected():
    with pytest.raises(ValidationError, match="EMBEDDING_DIM"):
        make_settings(embedding_dim=768)


def test_xai_settings_optional_for_now():
    s = make_settings()
    assert s.xai_api_key is None and s.xai_chat_model is None
