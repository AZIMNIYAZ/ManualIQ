from fastapi import Request
from qdrant_client import QdrantClient

from app.core.config import Settings


def create_qdrant_client(settings: Settings) -> QdrantClient:
    # Creating the client does not connect yet, so this works even if Qdrant is down.
    return QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)


def get_qdrant(request: Request) -> QdrantClient:
    """FastAPI dependency: the shared client created at startup."""
    return request.app.state.qdrant
