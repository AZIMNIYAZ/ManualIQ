"""FastAPI entry point.  Run with:  uvicorn app.main:app --reload"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, ingest, search
from app.core.config import get_settings
from app.core.logging import RequestIdMiddleware, setup_logging
from app.database.qdrant import create_qdrant_client
from app.database.redis import create_redis_client
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Clients are created once and shared by all requests. The app starts even if
    # Qdrant or Redis is down; /health reports that instead.
    app.state.qdrant = create_qdrant_client(settings)
    app.state.redis = create_redis_client(settings)
    # The embedding model is loaded once; if it is missing, startup fails with a clear error
    app.state.embedder = get_embedding_service()
    logger.info("Startup complete", extra={"fields": {
        "qdrant_url": settings.qdrant_url, "collection": settings.collection_name,
        "embedding_model": app.state.embedder.model_version}})
    yield
    app.state.qdrant.close()
    app.state.redis.close()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()  # fails fast here if .env is invalid
    setup_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    return app


app = create_app()
