"""Builds the knowledge base: manuals -> clean -> chunk -> embed -> Qdrant.

Order (the safety rule):
  1. Load, clean, chunk and embed everything in memory. Nothing is written yet, so a
     failure here (bad manifest, model problem) leaves the collection untouched.
  2. Upsert all points, tagged with this run's ID.
  3. Verify that exactly the expected number of points belongs to this run.
  4. Only then delete points from older runs (stale chunks).

Point IDs come from chunk_id, so re-running overwrites instead of duplicating. That
also means a failure during step 2 leaves old and new points mixed: the report says
so ("mixed") instead of claiming success. Building a new collection and switching a
Qdrant alias would make the swap atomic; not needed for this single-user project.

Only one ingestion runs at a time (in-process lock: fine for one uvicorn worker; with
several workers a shared lock, e.g. in Redis, would be needed).
"""

import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import FilterSelector, Filter, PointStruct

from app.core.errors import AppError
from app.core.config import Settings
from app.database.qdrant import CollectionMismatch, check_compatible, count, create_collection, field_equals
from app.models.chunk import Chunk
from app.models.ingestion import IngestionReport
from app.services.chunker import chunk_documents
from app.services.contextual_headers import HEADER_VERSION, build_embedding_texts
from app.services.document_loader import DocumentLoadError, load_all_documents, load_manifest
from app.services.embedding_service import EmbeddingService
from app.services.text_cleaner import CLEANER_VERSION, clean_documents
from app.services.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

# Fixed namespace: the same chunk_id always maps to the same point ID
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "rag-proj.local/chunks")

_lock = threading.Lock()


class IngestionError(AppError):
    status_code = 500
    code = "ingestion_failed"


class IngestionInProgress(IngestionError):
    status_code = 409
    code = "ingestion_in_progress"
    retry_after = 30


class ConfirmationRequired(IngestionError):
    status_code = 400
    code = "confirmation_required"


class DocumentsInvalid(IngestionError):
    status_code = 422
    code = "documents_invalid"


class IncompatibleCollection(IngestionError):
    status_code = 409
    code = "incompatible_collection"


class QdrantUnavailable(IngestionError):
    status_code = 503
    code = "vector_db_unavailable"
    retry_after = 5


class IngestionWriteFailed(IngestionError):
    """Writing failed part-way; details holds the report (collection_state says what is left)."""
    status_code = 500
    code = "ingestion_write_failed"


def point_id(chunk_id: str) -> str:
    """Deterministic UUID from the chunk ID only (never from the run), so re-runs overwrite."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def chunk_payload(chunk: Chunk, embedding_model: str, ingest_run: str, header: str | None = None) -> dict:
    """`header`: the contextual header embedded with the chunk (None: headers off). The stored
    text is always the chunk text alone."""
    payload = {**chunk.model_dump(), "embedding_model": embedding_model, "ingest_run": ingest_run,
               "contextual_headers": header is not None}
    if header is not None:
        payload["header_version"] = HEADER_VERSION
        payload["embedded_header"] = header
    return payload


def ingest(
    client: QdrantClient,
    embedder: EmbeddingService,
    settings: Settings,
    recreate: bool = False,
    confirm: str | None = None,
) -> IngestionReport:
    if not _lock.acquire(blocking=False):
        raise IngestionInProgress("An ingestion is already running; try again when it finishes")
    try:
        return _ingest(client, embedder, settings, recreate, confirm)
    finally:
        _lock.release()  # released even if ingestion raised


def _ingest(client, embedder, settings: Settings, recreate: bool, confirm: str | None) -> IngestionReport:
    name = settings.collection_name
    if recreate and confirm != name:
        raise ConfirmationRequired(f"recreate deletes {name!r}: pass confirm={name!r} to proceed")

    versions = {"cleaner": CLEANER_VERSION, "chunking": settings.chunking, "embedding_model": embedder.model_version}
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def lap(stage: str, since: float) -> float:
        now = time.perf_counter()
        timings[stage] = round(now - since, 2)
        return now

    # ---- 1. Build everything in memory (no writes)
    t = time.perf_counter()
    try:
        docs = load_all_documents(settings.documents_dir)
    except DocumentLoadError as e:
        raise DocumentsInvalid(str(e)) from e
    t = lap("load", t)
    cleaned, clean_reports = clean_documents(docs)
    t = lap("clean", t)
    chunks, _ = chunk_documents(cleaned, settings.chunk_size, settings.chunk_overlap)
    t = lap("chunk", t)
    if not chunks:
        raise DocumentsInvalid("The documents produced no chunks")
    wrong = [c.chunk_id for c in chunks if c.chunking != settings.chunking]
    if wrong:  # the collection name and the chunker must agree (rag_200_40 holds only 200/40 chunks)
        raise IngestionError(f"{len(wrong)} chunks have chunking other than {settings.chunking}")
    headers: list[str | None] = [None] * len(chunks)
    header_report = None
    texts = [c.text for c in chunks]
    if settings.contextual_headers:
        brand_names = {m.brand: (m.aliases[0] if m.aliases else m.brand) for m in load_manifest(settings.documents_dir)}
        texts, headers, header_report = build_embedding_texts(
            chunks, cleaned, brand_names, lambda text: len(get_tokenizer().encode(text, add_special_tokens=True)))
        versions["contextual_headers"] = HEADER_VERSION
    vectors = embedder.embed_documents(texts)
    t = lap("embed", t)

    report = IngestionReport(
        status="failed", collection=name, collection_state="unchanged", versions=versions,
        documents_processed=len(docs),
        pages_indexed=sum(r.pages_indexed for r in clean_reports),
        pages_excluded=sum(r.pages_excluded for r in clean_reports),
        chunks_created=len(chunks),
        contextual_headers=header_report.as_dict() if header_report else None,
    )

    # ---- 2. Prepare the collection
    try:
        exists = client.collection_exists(name)
        if exists and recreate:
            client.delete_collection(name)
            report.recreated, exists = True, False
            logger.warning("Collection deleted for recreate", extra={"fields": {"collection": name}})
        if exists:
            report.warnings = check_compatible(
                client, name, embedder.dim, embedder.model_version, settings.chunking, CLEANER_VERSION,
                settings.contextual_headers,
            )
        else:
            create_collection(client, name, embedder.dim)
    except CollectionMismatch as e:
        raise IncompatibleCollection(str(e)) from e
    except Exception as e:
        raise QdrantUnavailable(f"Qdrant error while preparing {name}: {e}") from e

    # ---- 3. Upsert, tagged with this run
    run = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    report.ingest_run = run
    batch = settings.qdrant_upsert_batch_size
    # From the first write attempt until verification, the collection may hold old and new
    # points (a failed request can still have been applied server-side, e.g. after a timeout)
    report.collection_state = "mixed"
    try:
        for i in range(0, len(chunks), batch):
            client.upsert(
                name,
                points=[
                    PointStruct(id=point_id(c.chunk_id), vector=v.tolist(),
                                payload=chunk_payload(c, embedder.model_version, run, h))
                    for c, v, h in zip(chunks[i:i + batch], vectors[i:i + batch], headers[i:i + batch])
                ],
                wait=True,
            )
    except Exception as e:
        report.error = f"Upsert failed: {e}. Old points were NOT deleted; re-run ingestion."
        logger.error("Ingestion upsert failed", extra={"fields": {"collection": name, "run": run}}, exc_info=True)
        report.timings_s = {**timings, "total": round(time.perf_counter() - started, 2)}
        return report
    t = lap("upsert", t)

    # ---- 4. Verify, then 5. delete stale points from older runs
    try:
        written = count(client, name, Filter(must=[field_equals("ingest_run", run)]))
        report.chunks_indexed = written
        if written != len(chunks):
            report.error = f"Verification failed: {written} points from this run, expected {len(chunks)}. Old points were NOT deleted."
            report.timings_s = {**timings, "total": round(time.perf_counter() - started, 2)}
            return report
        stale_filter = Filter(must_not=[field_equals("ingest_run", run)])
        stale = count(client, name, stale_filter)
        if stale:
            client.delete(name, points_selector=FilterSelector(filter=stale_filter), wait=True)
        report.stale_points_deleted = stale
    except Exception as e:
        report.error = f"Qdrant error after upsert: {e}. Re-run ingestion."
        report.timings_s = {**timings, "total": round(time.perf_counter() - started, 2)}
        return report
    lap("verify_and_cleanup", t)

    report.status, report.collection_state = "success", "updated"
    report.timings_s = {**timings, "total": round(time.perf_counter() - started, 2)}
    logger.info("ingest_completed", extra={"fields": {"endpoint": "ingest", **report.model_dump(include={
        "collection", "chunks_indexed", "stale_points_deleted", "versions", "timings_s"})}})
    return report
