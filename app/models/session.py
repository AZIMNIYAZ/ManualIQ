from datetime import datetime
from typing import Literal

from pydantic import BaseModel

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    # assistant messages only: what this turn actually searched and cited
    rewritten_query: str | None = None
    brands: list[str] | None = None
    sources: list[str] | None = None  # citations of this turn ("netgear_rax50.pdf, p. 139")
    found_in_documents: bool | None = None


class SessionResponse(BaseModel):
    session_id: str
    messages: list[StoredMessage]
    expires_in_seconds: int
