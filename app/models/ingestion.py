from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    recreate: bool = Field(False, description="Delete and rebuild the collection (needs confirm)")
    confirm: str | None = Field(None, description="Collection name, required when recreate is true")


class IngestionReport(BaseModel):
    status: Literal["success", "failed"]
    collection: str
    # updated: fully replaced and verified; unchanged: nothing was written; mixed: a write
    # failed part-way, so old and new points coexist and ingestion must be re-run
    collection_state: Literal["updated", "unchanged", "mixed"]
    recreated: bool = False
    documents_processed: int = 0
    pages_indexed: int = 0
    pages_excluded: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    stale_points_deleted: int = 0
    ingest_run: str | None = None
    versions: dict[str, str] = {}
    warnings: list[str] = []
    error: str | None = None
    timings_s: dict[str, float] = {}
