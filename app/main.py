"""FastAPI entry point.  Run with:  uvicorn app.main:app --reload"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.core.config import get_settings
from app.core.logging import RequestIdMiddleware, setup_logging
from app.database.qdrant import create_qdrant_client
from app.database.redis import create_redis_client

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Clients are created once and shared by all requests. The app starts even if
    # Qdrant or Redis is down; /health reports that instead.
    app.state.qdrant = create_qdrant_client(settings)
    app.state.redis = create_redis_client(settings)
    logger.info("Startup complete", extra={"fields": {"qdrant_url": settings.qdrant_url, "collection": settings.qdrant_collection}})
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
    # Step 6+: app.include_router(ingest.router, prefix="/api/v1") etc.
    return app


app = create_app()
