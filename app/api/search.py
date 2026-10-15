from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.database.qdrant import get_qdrant
from app.models.search import SearchRequest, SearchResponse
from app.services.brand_detection import AliasTable, alias_table_from_manifest
from app.services.embedding_service import EmbeddingService, get_embedder
from app.services.retrieval_service import search

router = APIRouter(prefix="/api/v1", tags=["search"])


def get_aliases(settings: Settings = Depends(get_settings)) -> AliasTable:
    return alias_table_from_manifest(settings.documents_dir)


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        409: {"description": "Knowledge base not ingested yet"},
        422: {"description": "Invalid request (blank query, top_k out of range, unknown brand)"},
        503: {"description": "Vector database unavailable"},
    },
)
def search_chunks(
    request: SearchRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    embedder: EmbeddingService = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
    aliases: AliasTable = Depends(get_aliases),
):
    """Vector search over the manuals (no LLM). Brands named in the query are filtered automatically."""
    return search(qdrant, embedder, settings, request, aliases)
