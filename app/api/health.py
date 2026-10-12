import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from redis import Redis

from app.database.qdrant import get_qdrant
from app.database.redis import get_redis
from app.models.health import HealthResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


def _error_text(e: Exception) -> str:
    return str(e) or type(e).__name__


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Qdrant or Redis is unreachable"}},
)
def health(qdrant: QdrantClient = Depends(get_qdrant), redis: Redis = Depends(get_redis)):
    details: dict[str, str] = {}

    try:
        qdrant.get_collections()
        qdrant_status = "ok"
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
        details=details,
    )
    if details:
        logger.warning("Health check degraded", extra={"fields": {"details": details}})
    return JSONResponse(status_code=200 if not details else 503, content=body.model_dump())
