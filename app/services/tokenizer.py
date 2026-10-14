"""The bge tokenizer, loaded once and shared.

Chunk sizes are measured with the embedding model's own tokenizer, so a chunk of
"350 tokens" is exactly 350 tokens to bge (which reads at most 512).
"""

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.core.config import get_settings


@lru_cache
def get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(get_settings().embedding_model)


def count_tokens(text: str) -> int:
    """Tokens bge sees for this text, excluding the [CLS]/[SEP] markers it adds."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))
