import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from redis import Redis

from app.core.config import Settings, get_settings
from app.database.qdrant import collection_problems, count, get_qdrant
from app.database.redis import get_redis
from app.models.health import CollectionInfo, HealthResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


def _error_text(e: Exception) -> str:
    return str(e) or type(e).__name__


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Qdrant or Redis is unreachable"}},
)
def health(
    qdrant: QdrantClient = Depends(get_qdrant),
    redis: Redis = Depends(get_redis),
    settings: Settings = Depends(get_settings),
):
    details: dict[str, str] = {}
    collection = None

    try:
        qdrant.get_collections()
        qdrant_status = "ok"
        # A missing collection is not an outage (just not ingested yet): reported, still 200
        name = settings.collection_name
        exists = qdrant.collection_exists(name)
        points = count(qdrant, name) if exists else None
        collection = CollectionInfo(name=name, exists=exists, points=points)
        if points:
            collection.problems = collection_problems(qdrant, name, settings.embedding_model_version,
                                                      settings.chunking, settings.contextual_headers)
            collection.compatible = not collection.problems
    except Exception as e:
        qdrant_status = "down"
        details["qdrant"] = _error_text(e)

    try:
        redis.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = "down"
        details["redis"] = _error_text(e)

    body = HealthResponse(
        status="ok" if not details else "degraded",
        qdrant=qdrant_status,
        redis=redis_status,
        collection=collection,
        llm="configured" if settings.xai_api_key and settings.xai_chat_model else "not_configured",
        details=details,
    )
    if details:
        logger.warning("Health check degraded", extra={"fields": {"details": details}})
    return JSONResponse(status_code=200 if not details else 503, content=body.model_dump())
