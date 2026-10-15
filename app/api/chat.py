from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.api.search import get_aliases
from app.core.config import Settings, get_settings
from app.database.qdrant import get_qdrant
from app.models.chat import ChatRequest, ChatResponse
from app.services.brand_detection import AliasTable
from app.services.chat_service import answer
from app.services.embedding_service import EmbeddingService, get_embedder
from app.services.generation_service import GenerationService, get_generator, get_rewriter
from app.services.session_service import SessionStore, get_session_store

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
    sessions: SessionStore = Depends(get_session_store),
    rewriter: GenerationService | None = Depends(get_rewriter),
):
    """Answers a question from the manuals with cited sources.

    Send the returned session_id with the next message to continue the conversation:
    follow-ups ("What about on ASUS?") are rewritten into standalone questions.
    """
    return answer(qdrant, embedder, generator, settings, request, aliases, sessions, rewriter)
