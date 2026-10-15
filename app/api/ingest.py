from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.database.qdrant import get_qdrant
from app.models.ingestion import IngestionReport, IngestRequest
from app.services.embedding_service import EmbeddingService, get_embedder
from app.services.ingestion_service import IngestionWriteFailed, ingest

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post(
    "/ingest",
    response_model=IngestionReport,
    responses={
        400: {"description": "recreate without the right confirm value"},
        409: {"description": "Ingestion already running, or collection built with incompatible settings"},
        422: {"description": "Manifest or documents invalid"},
        500: {"description": "ingestion_write_failed: writing failed part-way; error.details holds the report"},
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
    report = ingest(qdrant, embedder, settings, recreate=body.recreate, confirm=body.confirm)
    if report.status != "success":
        raise IngestionWriteFailed(report.error or "Ingestion failed", details=report.model_dump(mode="json"))
    return report
