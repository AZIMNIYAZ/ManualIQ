import hashlib
import json
import os
import socket

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Filter

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import Settings  # noqa: E402
from app.database.qdrant import INDEXED_FIELDS, count, create_qdrant_client, field_equals  # noqa: E402
from app.services import ingestion_service as svc  # noqa: E402
from app.services.ingestion_service import (  # noqa: E402
    ConfirmationRequired,
    DocumentsInvalid,
    IncompatibleCollection,
    IngestionInProgress,
    ingest,
    point_id,
)


# ---------------------------------------------------------------- helpers

class FakeEmbedder:
    """Deterministic unit vectors from a hash of the text: fast, no model needed."""

    dim = 384

    def __init__(self, model_version: str = "fake-model@1", fail: bool = False):
        self.model_version = model_version
        self.model_name = model_version.split("@")[0]
        self.fail = fail

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("model crashed")
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)

    def embed_query(self, text):
        from app.services.embedding_service import QueryEmbedding
        return QueryEmbedding(vector=self.embed_documents([text])[0], token_count=len(text.split()) + 11,
                              truncated=False)


def procedure(name: str, steps: int) -> str:
    lines = [f"Configure {name}", f"This section explains how to configure {name} on the router.",
             f"To set up {name}:"]
    lines += [f"{i}. Open the {name} page and set option {i} to the value that you need." for i in range(1, steps + 1)]
    return "\n".join(lines)


def write_docs(folder, acme_topics=("parental controls", "guest network", "port forwarding"), steps=6):
    (folder / "acme.txt").write_text("\n".join(procedure(t, steps) for t in acme_topics), encoding="utf-8")
    (folder / "zeta.txt").write_text(procedure("firewall", steps) + "\n" + procedure("VPN server", steps),
                                     encoding="utf-8")
    manifest = {"documents": [
        {"file": "acme.txt", "document_id": "acme_x1", "brand": "acme", "model": "X1"},
        {"file": "zeta.txt", "document_id": "zeta_z9", "brand": "zeta", "model": "Z9"},
    ]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def make_settings(folder, size=80, overlap=10) -> Settings:
    return Settings(_env_file=None, documents_dir=str(folder), qdrant_collection_prefix="rag_test",
                    chunk_size=size, chunk_overlap=overlap)


def qdrant_running() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", 6333)) == 0


needs_qdrant = pytest.mark.skipif(not qdrant_running(), reason="Qdrant not running (docker compose up -d)")


@pytest.fixture
def qdrant():
    client = create_qdrant_client(Settings(_env_file=None))
    yield client
    for c in client.get_collections().collections:  # never touches rag_350_60 etc.
        if c.name.startswith("rag_test"):
            client.delete_collection(c.name)


# ---------------------------------------------------------------- without Docker

def test_point_id_is_deterministic_and_from_chunk_id_only():
    assert point_id("acme_x1_0003") == point_id("acme_x1_0003")
    assert point_id("acme_x1_0003") != point_id("acme_x1_0004")


def test_collection_name_follows_chunk_settings():
    assert Settings(_env_file=None).collection_name == "rag_350_60"
    assert Settings(_env_file=None, chunk_size=200, chunk_overlap=40).collection_name == "rag_200_40"
    assert Settings(_env_file=None, qdrant_collection_suffix="ctx").collection_name == "rag_350_60_ctx"


def test_recreate_needs_the_collection_name(tmp_path):
    with pytest.raises(ConfirmationRequired, match="confirm='rag_test_80_10'"):
        ingest(None, FakeEmbedder(), make_settings(tmp_path), recreate=True, confirm="yes")


def test_second_ingestion_is_refused_and_lock_is_released_after_errors(tmp_path):
    assert svc._lock.acquire(blocking=False)
    try:
        with pytest.raises(IngestionInProgress):
            ingest(None, FakeEmbedder(), make_settings(tmp_path))
    finally:
        svc._lock.release()
    with pytest.raises(DocumentsInvalid):  # no manifest: fails inside ingestion
        ingest(None, FakeEmbedder(), make_settings(tmp_path))
    assert svc._lock.acquire(blocking=False)  # the error did not leave the lock held
    svc._lock.release()


# ---------------------------------------------------------------- with Qdrant (rag_test_* only)

@needs_qdrant
def test_ingest_twice_gives_same_points_and_no_duplicates(qdrant, tmp_path):
    write_docs(tmp_path)
    s = make_settings(tmp_path)
    first = ingest(qdrant, FakeEmbedder(), s)
    assert first.status == "success" and first.collection_state == "updated"
    assert first.chunks_indexed == first.chunks_created > 3
    second = ingest(qdrant, FakeEmbedder(), s)
    # same chunks -> same IDs -> overwritten in place with the new run tag: nothing stale
    assert second.status == "success" and second.stale_points_deleted == 0
    assert count(qdrant, s.collection_name) == first.chunks_created  # overwritten, not duplicated

    point = qdrant.scroll(s.collection_name, limit=1, with_payload=True)[0][0]
    for field in ("chunk_id", "document_id", "brand", "model", "source", "page_start", "page_end", "section",
                  "content_type", "text", "token_count", "chunking", "cleaner_version", "embedding_model",
                  "ingest_run"):
        assert field in point.payload, field
    assert str(point.id) == point_id(point.payload["chunk_id"])
    schema = qdrant.get_collection(s.collection_name).payload_schema
    assert set(INDEXED_FIELDS) <= set(schema)


@needs_qdrant
def test_stale_chunks_are_removed_after_a_successful_run(qdrant, tmp_path):
    write_docs(tmp_path)
    s = make_settings(tmp_path)
    before = ingest(qdrant, FakeEmbedder(), s).chunks_created
    write_docs(tmp_path, acme_topics=("parental controls",))  # the manual got shorter
    after = ingest(qdrant, FakeEmbedder(), s)
    assert after.status == "success" and after.chunks_created < before
    assert count(qdrant, s.collection_name) == after.chunks_created
    assert after.stale_points_deleted == before - after.chunks_created  # exactly the chunks that disappeared


@needs_qdrant
def test_failure_before_writing_leaves_collection_untouched(qdrant, tmp_path):
    write_docs(tmp_path)
    s = make_settings(tmp_path)
    first = ingest(qdrant, FakeEmbedder(), s)
    with pytest.raises(RuntimeError, match="model crashed"):
        ingest(qdrant, FakeEmbedder(fail=True), s)
    run_filter = Filter(must=[field_equals("ingest_run", first.ingest_run)])
    assert count(qdrant, s.collection_name, run_filter) == first.chunks_indexed  # all old points intact


@needs_qdrant
def test_failed_upsert_reports_mixed_and_deletes_nothing(qdrant, tmp_path):
    write_docs(tmp_path)
    s = make_settings(tmp_path)
    s.qdrant_upsert_batch_size = 3
    first = ingest(qdrant, FakeEmbedder(), s)

    class FlakyClient:  # first upsert batch works, the second fails
        def __init__(self, real):
            self.real, self.calls = real, 0

        def __getattr__(self, name):
            return getattr(self.real, name)

        def upsert(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ConnectionError("Qdrant went away")
            return self.real.upsert(*args, **kwargs)

    second = ingest(FlakyClient(qdrant), FakeEmbedder(model_version="fake-model@1"), s)
    assert second.status == "failed" and second.collection_state == "mixed"
    assert "NOT deleted" in second.error
    old_left = count(qdrant, s.collection_name, Filter(must=[field_equals("ingest_run", first.ingest_run)]))
    assert old_left > 0  # previous points were not deleted


@needs_qdrant
def test_incompatible_collection_is_refused_until_recreate(qdrant, tmp_path):
    write_docs(tmp_path)
    s = make_settings(tmp_path)
    ingest(qdrant, FakeEmbedder("fake-model@1"), s)
    with pytest.raises(IncompatibleCollection, match="embedding_model"):
        ingest(qdrant, FakeEmbedder("other-model@2"), s)
    rebuilt = ingest(qdrant, FakeEmbedder("other-model@2"), s, recreate=True, confirm=s.collection_name)
    assert rebuilt.status == "success" and rebuilt.recreated

    one = qdrant.scroll(s.collection_name, limit=1)[0][0]  # one point with foreign chunking
    qdrant.set_payload(s.collection_name, payload={"chunking": "999/0"}, points=[one.id], wait=True)
    with pytest.raises(IncompatibleCollection, match="chunking"):
        ingest(qdrant, FakeEmbedder("other-model@2"), s)


@needs_qdrant
def test_brand_filter_returns_only_that_brand(qdrant, tmp_path):
    write_docs(tmp_path)
    s = make_settings(tmp_path)
    ingest(qdrant, FakeEmbedder(), s)
    query = FakeEmbedder().embed_documents(["anything"])[0].tolist()
    hits = qdrant.query_points(s.collection_name, query=query, query_filter=Filter(must=[field_equals("brand", "zeta")]),
                               limit=50, with_payload=True).points
    assert hits and {h.payload["brand"] for h in hits} == {"zeta"}


@needs_qdrant
def test_experiment_settings_really_change_the_chunks(qdrant, tmp_path):
    write_docs(tmp_path)
    s = make_settings(tmp_path, size=60, overlap=0)
    report = ingest(qdrant, FakeEmbedder(), s)
    assert report.collection == "rag_test_60_0"
    points = qdrant.scroll("rag_test_60_0", limit=500, with_payload=True)[0]
    assert all(p.payload["chunking"] == "60/0" and p.payload["token_count"] <= 60 for p in points)


# ---------------------------------------------------------------- API

@pytest.mark.parametrize("error, status", [
    (IngestionInProgress("busy"), 409), (ConfirmationRequired("confirm"), 400),
    (DocumentsInvalid("bad manifest"), 422), (IncompatibleCollection("model"), 409),
    (svc.QdrantUnavailable("down"), 503),
])
def test_api_maps_errors_to_status_codes(monkeypatch, error, status):
    from app.api import ingest as ingest_api
    from app.database.qdrant import get_qdrant
    from app.main import app
    from app.services.embedding_service import get_embedder

    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(ingest_api, "ingest", raise_error)
    app.dependency_overrides[get_qdrant] = lambda: None
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    try:
        r = TestClient(app).post("/api/v1/ingest", json={"recreate": False})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == status and r.json()["status"] == "failed"
