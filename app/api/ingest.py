from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.database.qdrant import get_qdrant
from app.models.ingestion import IngestionReport, IngestRequest
from app.services.embedding_service import EmbeddingService, get_embedder
from app.services.ingestion_service import IngestionError, ingest

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post(
    "/ingest",
    response_model=IngestionReport,
    responses={
        400: {"description": "recreate without the right confirm value"},
        409: {"description": "Ingestion already running, or collection built with incompatible settings"},
        422: {"description": "Manifest or documents invalid"},
        500: {"model": IngestionReport, "description": "Writing failed part-way (see collection_state)"},
        503: {"description": "Qdrant unavailable"},
    },
)
def ingest_documents(
    body: IngestRequest | None = None,
    qdrant: QdrantClient = Depends(get_qdrant),
    embedder: EmbeddingService = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
):
    """Builds the knowledge base from documents/manifest.json. Takes about 90 seconds."""
    body = body or IngestRequest()
    try:
        report = ingest(qdrant, embedder, settings, recreate=body.recreate, confirm=body.confirm)
    except IngestionError as e:
        return JSONResponse(status_code=e.status_code, content={"status": "failed", "error": str(e)})
    return JSONResponse(status_code=200 if report.status == "success" else 500, content=report.model_dump())
