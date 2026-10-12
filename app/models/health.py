from typing import Literal

from pydantic import BaseModel

ServiceStatus = Literal["ok", "down"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: ServiceStatus = "ok"
    qdrant: ServiceStatus
    redis: ServiceStatus
    details: dict[str, str] = {}
