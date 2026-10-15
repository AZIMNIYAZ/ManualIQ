from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.api.search import get_aliases
from app.core.config import Settings, get_settings
from app.database.qdrant import get_qdrant
from app.models.chat import ChatRequest, ChatResponse
from app.services.brand_detection import AliasTable
from app.services.chat_service import answer
from app.services.embedding_service import EmbeddingService, get_embedder
from app.services.generation_service import GenerationService, get_generator

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        409: {"description": "Knowledge base not ingested yet"},
        422: {"description": "Invalid request"},
        429: {"description": "LLM rate limit reached"},
        502: {"description": "LLM error (authentication, API error)"},
        503: {"description": "Generation not configured, or vector database unavailable"},
        504: {"description": "LLM timeout"},
    },
)
def chat(
    request: ChatRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    embedder: EmbeddingService = Depends(get_embedder),
    generator: GenerationService = Depends(get_generator),
    settings: Settings = Depends(get_settings),
    aliases: AliasTable = Depends(get_aliases),
):
    """Answers a question from the manuals with cited sources (one-shot: no conversation memory yet)."""
    return answer(qdrant, embedder, generator, settings, request, aliases)
