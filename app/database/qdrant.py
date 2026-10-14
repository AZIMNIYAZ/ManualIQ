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


def check_compatible(
    client: QdrantClient, name: str, dim: int, embedding_model: str, chunking: str, cleaner_version: str
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

    for field, expected in (("embedding_model", embedding_model), ("chunking", chunking)):
        different = count(client, name, Filter(must_not=[field_equals(field, expected)]))
        if different:
            raise CollectionMismatch(
                f"{different} points in {name} have a different {field} than {expected!r}; "
                f"re-run with recreate to rebuild the collection"
            )

    warnings = []
    older = count(client, name, Filter(must_not=[field_equals("cleaner_version", cleaner_version)]))
    if older:
        warnings.append(f"{older} existing points were cleaned with another cleaner version; all are replaced by this run")
    return warnings
