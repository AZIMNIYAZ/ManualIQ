"""FastAPI entry point.  Run with:  uvicorn app.main:app --reload"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import chat, health, ingest, search, sessions
from app.core.config import get_settings
from app.core.logging import RequestIdMiddleware, setup_logging
from app.database.qdrant import create_qdrant_client
from app.database.redis import create_redis_client
from app.services.embedding_service import get_embedding_service
from app.services.generation_service import GenerationError, create_generator, create_rewriter
from app.services.ingestion_service import IngestionError
from app.services.retrieval_service import SearchError
from app.services.session_service import SessionNotFound, SessionUnavailable, create_session_store

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
    # The LLM is optional at startup: without XAI_API_KEY, /search works and /chat returns 503
    app.state.generator = create_generator(settings)
    app.state.rewriter = create_rewriter(settings)  # follow-up questions; None without an API key
    app.state.sessions = create_session_store(app.state.redis, settings)
    logger.info("Startup complete", extra={"fields": {
        "qdrant_url": settings.qdrant_url, "collection": settings.collection_name,
        "embedding_model": app.state.embedder.model_version,
        "llm_model": settings.xai_chat_model if app.state.generator else None,
        "rewrite_model": app.state.rewriter.model if app.state.rewriter else None}})
    yield
    app.state.qdrant.close()
    app.state.redis.close()
    logger.info("Shutdown complete")


async def service_error(request: Request, exc: Exception) -> JSONResponse:
    """Known service errors carry their HTTP status; the body is {"error": message}."""
    logger.warning("Request failed", extra={"fields": {"error_type": type(exc).__name__, "error": str(exc)}})
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})


def create_app() -> FastAPI:
    settings = get_settings()  # fails fast here if .env is invalid
    setup_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    for error in (SearchError, GenerationError, IngestionError, SessionUnavailable, SessionNotFound):
        app.add_exception_handler(error, service_error)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    return app


app = create_app()
