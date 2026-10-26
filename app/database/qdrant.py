from fastapi import Request
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from app.core.config import Settings

# Payload fields used in filters; indexed when the collection is created
INDEXED_FIELDS = {
    "brand": PayloadSchemaType.KEYWORD,
    "document_id": PayloadSchemaType.KEYWORD,
    "content_type": PayloadSchemaType.KEYWORD,
    "ingest_run": PayloadSchemaType.KEYWORD,
    "page_start": PayloadSchemaType.INTEGER,
    "contextual_headers": PayloadSchemaType.BOOL,
}


class CollectionMismatch(Exception):
    """The existing collection was built with incompatible settings."""


def create_qdrant_client(settings: Settings) -> QdrantClient:
    # Creating the client does not connect yet, so this works even if Qdrant is down.
    return QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)


def get_qdrant(request: Request) -> QdrantClient:
    """FastAPI dependency: the shared client created at startup."""
    return request.app.state.qdrant


def field_equals(field: str, value) -> FieldCondition:
    return FieldCondition(key=field, match=MatchValue(value=value))


def count(client: QdrantClient, collection: str, query_filter: Filter | None = None) -> int:
    return client.count(collection, count_filter=query_filter, exact=True).count


def create_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Creates the collection and its payload indexes, before any point is written."""
    client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    for field, schema in INDEXED_FIELDS.items():
        client.create_payload_index(name, field_name=field, field_schema=schema, wait=True)


def header_mismatches(client: QdrantClient, name: str, contextual_headers: bool) -> int:
    """Points whose contextual-header setting differs. Points written before R2 have no field
    and count as 'no headers'."""
    if contextual_headers:
        return count(client, name, Filter(must_not=[field_equals("contextual_headers", True)]))
    return count(client, name, Filter(must=[field_equals("contextual_headers", True)]))


def collection_problems(client: QdrantClient, name: str, embedding_model: str, chunking: str,
                        contextual_headers: bool) -> list[str]:
    """For the search side (/health, evaluation scripts): does this collection hold points
    built with the current settings? An empty list means compatible."""
    problems = []
    for field, expected in (("embedding_model", embedding_model), ("chunking", chunking)):
        different = count(client, name, Filter(must_not=[field_equals(field, expected)]))
        if different:
            problems.append(f"{different} points have a different {field} than {expected!r}")
    different = header_mismatches(client, name, contextual_headers)
    if different:
        problems.append(f"{different} points were built {'without' if contextual_headers else 'with'} "
                        f"contextual headers (CONTEXTUAL_HEADERS={str(contextual_headers).lower()})")
    return problems


def check_compatible(
    client: QdrantClient, name: str, dim: int, embedding_model: str, chunking: str, cleaner_version: str,
    contextual_headers: bool = False,
) -> list[str]:
    """Checks the whole collection, not one sample point.

    Raises CollectionMismatch if vectors would not be comparable (model, dimension,
    distance) or the experiment would be mixed (chunking). Returns warnings for
    differences that a full re-ingestion resolves (cleaner version).
    """
    vectors = client.get_collection(name).config.params.vectors
    if vectors.size != dim or vectors.distance != Distance.COSINE:
        raise CollectionMismatch(
            f"{name} stores {vectors.size}-dim {vectors.distance} vectors; this setup needs {dim}-dim Cosine"
        )

    for problem in collection_problems(client, name, embedding_model, chunking, contextual_headers):
        raise CollectionMismatch(f"{name}: {problem}; re-run with recreate to rebuild the collection")

    warnings = []
    older = count(client, name, Filter(must_not=[field_equals("cleaner_version", cleaner_version)]))
    if older:
        warnings.append(f"{older} existing points were cleaned with another cleaner version; all are replaced by this run")
    return warnings
