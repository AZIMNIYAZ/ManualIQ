from typing import Literal

from pydantic import BaseModel

ServiceStatus = Literal["ok", "down"]


class CollectionInfo(BaseModel):
    name: str
    exists: bool
    points: int | None  # None if the collection has not been created (not ingested yet)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: ServiceStatus = "ok"
    qdrant: ServiceStatus
    redis: ServiceStatus
    collection: CollectionInfo | None = None  # None if Qdrant is down
    details: dict[str, str] = {}
