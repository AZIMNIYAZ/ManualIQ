"""Vector search over the knowledge base, with brand and regulatory filters.

Pure retrieval (no LLM): answers "is the system retrieving the right chunks?".
"""

import logging
import time

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.core.config import Settings
from app.database.qdrant import count, field_equals
from app.models.search import SearchFilters, SearchRequest, SearchResponse, SearchResult
from app.services.brand_detection import AliasTable, detect_brands
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SearchError(Exception):
    status_code = 500


class KnowledgeBaseEmpty(SearchError):
    status_code = 409


class InvalidBrand(SearchError):
    status_code = 422


class VectorDatabaseUnavailable(SearchError):
    status_code = 503


def citation(source: str, page_start: int, page_end: int) -> str:
    """'netgear_rax50.pdf, p. 139' or 'netgear_rax50.pdf, pp. 139-140' (PDF viewer pages)."""
    if page_start == page_end:
        return f"{source}, p. {page_start}"
    return f"{source}, pp. {page_start}–{page_end}"


def collection_brands(client: QdrantClient, name: str) -> dict[str, int]:
    """Brands actually present in the collection, with chunk counts."""
    return {hit.value: hit.count for hit in client.facet(name, key="brand", limit=100, exact=True).hits}


def search(
    client: QdrantClient,
    embedder: EmbeddingService,
    settings: Settings,
    request: SearchRequest,
    aliases: AliasTable,
    detect_brand: bool = True,
) -> SearchResponse:
    started = time.perf_counter()
    name = settings.collection_name
    top_k = request.top_k or settings.top_k

    # Knowledge base present? (also tells us which brands exist)
    try:
        if not client.collection_exists(name) or count(client, name) == 0:
            raise KnowledgeBaseEmpty(f"Knowledge base has not been ingested ({name} is empty): run POST /api/v1/ingest")
        valid_brands = collection_brands(client, name)
        checks_ms = (time.perf_counter() - started) * 1000
    except SearchError:
        raise
    except Exception as e:
        raise VectorDatabaseUnavailable(f"Vector database unavailable: {e}") from e

    # Brand filter: an explicit request wins; otherwise detect brands named in the question
    if request.brand:
        unknown = [b for b in request.brand if b not in valid_brands]
        if unknown:
            raise InvalidBrand(f"Unknown brand {unknown}; valid brands: {sorted(valid_brands)}")
        brands, source = request.brand, "request"
    else:
        detected = [b for b in detect_brands(request.query, aliases) if b in valid_brands] if detect_brand else []
        brands, source = detected, ("detected" if detected else "none")

    conditions_must = [FieldCondition(key="brand", match=MatchAny(any=brands))] if brands else []
    conditions_not = [] if request.include_regulatory else [field_equals("content_type", "regulatory")]
    query_filter = Filter(must=conditions_must or None, must_not=conditions_not or None)

    t = time.perf_counter()
    query = embedder.embed_query(request.query)
    embed_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    try:
        points = client.query_points(
            name, query=query.vector.tolist(), query_filter=query_filter, limit=top_k, with_payload=True
        ).points
    except Exception as e:
        raise VectorDatabaseUnavailable(f"Vector database unavailable: {e}") from e
    search_ms = (time.perf_counter() - t) * 1000

    results = [
        SearchResult(
            rank=i,
            score=round(p.score, 4),
            chunk_id=p.payload["chunk_id"],
            brand=p.payload["brand"],
            model=p.payload["model"],
            source=p.payload["source"],
            page_start=p.payload["page_start"],
            page_end=p.payload["page_end"],
            citation=citation(p.payload["source"], p.payload["page_start"], p.payload["page_end"]),
            section=p.payload.get("section"),
            content_type=p.payload["content_type"],
            text=p.payload["text"],
        )
        for i, p in enumerate(points, start=1)
    ]
    response = SearchResponse(
        query=request.query,
        collection=name,
        filters=SearchFilters(brands=brands, brand_source=source, include_regulatory=request.include_regulatory),
        query_tokens=query.token_count,
        query_truncated=query.truncated,
        results=results,
        timings_ms={
            "checks": round(checks_ms, 1),  # collection exists / not empty / brand list
            "embed": round(embed_ms, 1),
            "search": round(search_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    logger.info("search", extra={"fields": {
        "query": request.query, "brands": brands, "brand_source": source,
        "include_regulatory": request.include_regulatory, "top_k": top_k,
        "results": [(r.chunk_id, r.score) for r in results], "timings_ms": response.timings_ms}})
    return response
