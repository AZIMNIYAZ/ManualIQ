from typing import Literal

from pydantic import BaseModel

ServiceStatus = Literal["ok", "down"]


class CollectionInfo(BaseModel):
    name: str
    exists: bool
    points: int | None  # None if the collection has not been created (not ingested yet)
    # Do the points match the current settings (embedding model, chunking, contextual headers)?
    compatible: bool | None = None  # None: nothing to check (missing or empty collection)
    problems: list[str] = []


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: ServiceStatus = "ok"
    qdrant: ServiceStatus
    redis: ServiceStatus
    collection: CollectionInfo | None = None  # None if Qdrant is down
    llm: Literal["configured", "not_configured"] = "not_configured"  # no API call: /chat needs it, /search does not
    details: dict[str, str] = {}
