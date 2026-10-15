from fastapi import APIRouter, Depends, Path, Response

from app.models.session import SESSION_ID_PATTERN, SessionResponse
from app.services.session_service import SessionStore, get_session_store

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

SessionId = Path(..., pattern=SESSION_ID_PATTERN)


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    responses={404: {"description": "Session does not exist or has expired"},
               503: {"description": "Session store (Redis) unavailable"}},
)
def get_session(session_id: str = SessionId, sessions: SessionStore = Depends(get_session_store)):
    """The conversation so far: each turn with its rewritten query, brands and citations."""
    messages, ttl = sessions.get(session_id)
    return SessionResponse(session_id=session_id, messages=messages, expires_in_seconds=ttl)


@router.delete("/{session_id}", status_code=204, responses={503: {"description": "Session store unavailable"}})
def delete_session(session_id: str = SessionId, sessions: SessionStore = Depends(get_session_store)):
    """Deletes the conversation. Also 204 if it did not exist."""
    sessions.delete(session_id)
    return Response(status_code=204)
