"""Step 12 R2: contextual headers (embedding input only; stored text unchanged)."""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import Settings  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document, Page  # noqa: E402
from app.services.contextual_headers import (FALLBACK_MAX_PAGES, HEADER_VERSION, build_embedding_texts,  # noqa: E402
                                             classify_label, toc_text)
from app.services.ingestion_service import chunk_payload  # noqa: E402

TOC = "Contents 3.6 Firewall .... 41 Website Filter ..... 79 Quick VPN .... 84 4.1 Use Quick Setup Wizard 18"


def doc() -> Document:
    return Document(document_id="dlink_dir_x1560", brand="dlink", model="DIR-X1560", source="d.pdf", pages=[
        Page(page_number=2, raw_text=TOC, status="excluded", reason="toc"),
        Page(page_number=79, raw_text="x", clean_text="x", status="indexed"),
    ])


def chunk(i: int, section: str | None, page: int, text: str = "1. Click Add Rule. 2. Select DENY.") -> Chunk:
    return Chunk(chunk_id=f"dlink_dir_x1560_{i:04d}", document_id="dlink_dir_x1560", brand="dlink",
                 model="DIR-X1560", source="d.pdf", chunk_index=i, text=text, token_count=10, page_start=page,
                 page_end=page, content_type="normal", section=section, cleaner_version="1.3", chunking="500/60")


def words(text: str) -> int:  # a stand-in tokenizer: 1 token per word + 2 special tokens
    return len(text.split()) + 2


@pytest.mark.parametrize("label", ["Website Filter", "3.6 Firewall", "4 1 Use Quick Setup Wizard",
                                   "9.11.5 Auto system maintenance", "To set up Port Forwarding"])
def test_trusted_labels(label):
    assert classify_label(label, toc_text(doc())) == "trusted"


@pytest.mark.parametrize("label", ["Modem", "Lifetime", "Type", "Figure 7. Cable your router",
                                   "Manually set up the Internet connection on page 24.)",
                                   "OR Smart DeviceComputer", "Copyright Statement", None, "2.4"])
def test_rejected_labels(label):
    assert classify_label(label, toc_text(doc())) == "rejected"


def test_header_format_and_text_unchanged():
    chunks = [chunk(0, "Website Filter", 79)]
    texts, headers, report = build_embedding_texts(chunks, [doc()], {"dlink": "D-Link"}, words)
    assert headers == ["D-Link DIR-X1560 | Website Filter"]
    assert texts == ["D-Link DIR-X1560 | Website Filter\n" + chunks[0].text]
    assert report.trusted == 1 and report.as_dict()["sections"] == {"trusted": 1, "fallback": 0, "omitted": 0}


def test_fallback_only_to_a_nearby_trusted_heading():
    chunks = [chunk(0, "Website Filter", 79), chunk(1, "Modem", 79 + FALLBACK_MAX_PAGES),
              chunk(2, "Lifetime", 79 + FALLBACK_MAX_PAGES + 1)]
    _, headers, report = build_embedding_texts(chunks, [doc()], {"dlink": "D-Link"}, words)
    assert headers == ["D-Link DIR-X1560 | Website Filter",  # trusted
                       "D-Link DIR-X1560 | Website Filter",  # rejected, trusted heading 2 pages back
                       "D-Link DIR-X1560"]  # rejected, too far back: no section rather than a wrong one
    rejected = {r["label"]: r["replaced_by"] for r in report.as_dict()["rejected_labels"]}
    assert rejected == {"Modem": "Website Filter", "Lifetime": "(none)"}
    assert (report.trusted, report.fallback, report.omitted) == (1, 1, 1)


def test_header_is_shortened_so_no_chunk_text_is_ever_truncated():
    long_text = " ".join(["word"] * 506)  # 506 + 2 special tokens: the 5-word full header does not fit, 2 words do
    chunks = [chunk(0, "Website Filter", 79, long_text)]
    texts, headers, report = build_embedding_texts(chunks, [doc()], {"dlink": "D-Link"}, words, max_tokens=512)
    assert headers == ["D-Link DIR-X1560"] and texts[0].endswith(long_text)  # section dropped, text complete
    assert words(texts[0]) <= 512 and report.shortened == {"no_section": 1}
    texts, headers, report = build_embedding_texts(chunks, [doc()], {"dlink": "D-Link"}, words, max_tokens=508)
    assert headers == [""] and texts == [long_text] and report.shortened == {"none": 1}


def test_section_is_capped():
    section = "To " + " ".join(["configure"] * 40)
    _, headers, _ = build_embedding_texts([chunk(0, section, 79)], [doc()], {"dlink": "D-Link"}, words)
    assert words(headers[0].split(" | ")[1]) <= 24


def test_payload_text_is_the_chunk_text_and_records_the_header():
    c = chunk(0, "Website Filter", 79)
    off = chunk_payload(c, "bge@5c38ec7", "run1")
    on = chunk_payload(c, "bge@5c38ec7", "run1", "D-Link DIR-X1560 | Website Filter")
    assert off["contextual_headers"] is False and "embedded_header" not in off
    assert on["contextual_headers"] is True and on["embedded_header"] == "D-Link DIR-X1560 | Website Filter"
    assert on["text"] == off["text"] == c.text and on["header_version"] == HEADER_VERSION


def test_collection_name_and_default_off():
    assert Settings(_env_file=None).contextual_headers is False
    assert Settings(_env_file=None, chunk_size=500).collection_name == "rag_500_60"
    assert Settings(_env_file=None, chunk_size=500, contextual_headers=True).collection_name == "rag_500_60_ctx"
    assert Settings(_env_file=None, embedding_model_revision="5c38ec7c40").embedding_model_version == \
        "BAAI/bge-small-en-v1.5@5c38ec7"


# ---------------------------------------------------------------- real Qdrant: compatibility

def _qdrant():
    from app.database.qdrant import create_qdrant_client
    client = create_qdrant_client(Settings(_env_file=None))
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant not running (docker compose up -d)")
    return client


def test_header_setting_mismatch_is_refused():
    from qdrant_client.models import PointStruct

    from app.database.qdrant import CollectionMismatch, check_compatible, collection_problems, create_collection
    client, name = _qdrant(), "rag_test_headers"
    if client.collection_exists(name):
        client.delete_collection(name)
    create_collection(client, name, 384)
    try:
        c = chunk(0, "Website Filter", 79)
        client.upsert(name, points=[PointStruct(id=1, vector=[0.1] * 384,
                                                payload=chunk_payload(c, "bge@5c38ec7", "r", "D-Link DIR-X1560"))], wait=True)
        assert collection_problems(client, name, "bge@5c38ec7", "500/60", True) == []
        problems = collection_problems(client, name, "bge@5c38ec7", "500/60", False)
        assert len(problems) == 1 and "with contextual headers" in problems[0]
        with pytest.raises(CollectionMismatch):
            check_compatible(client, name, 384, "bge@5c38ec7", "500/60", "1.3", contextual_headers=False)
        # points written before R2 have no field: they count as "no headers"
        legacy = {k: v for k, v in chunk_payload(c, "bge@5c38ec7", "r").items() if k != "contextual_headers"}
        client.upsert(name, points=[PointStruct(id=2, vector=[0.1] * 384, payload=legacy)], wait=True)
        assert any("without contextual headers" in p for p in collection_problems(client, name, "bge@5c38ec7", "500/60", True))
    finally:
        client.delete_collection(name)
