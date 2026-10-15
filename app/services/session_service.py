"""Conversation history in Redis.

Key session:{id} holds a list of JSON messages. A turn (user message + assistant
answer) is written in one MULTI/EXEC transaction, only after the answer succeeded:
RPUSH both, LTRIM to the last N messages, and EXPIRE (the TTL restarts on every turn,
which RPUSH alone would not do).

Limitation: two simultaneous requests on the same session could interleave their
turns. One user in one chat does not send overlapping requests, so no lock is used.
"""

import uuid

from fastapi import Request
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import Settings
from app.models.session import StoredMessage


class SessionUnavailable(Exception):
    status_code = 503


class SessionNotFound(Exception):
    status_code = 404


class SessionStore:
    def __init__(self, redis: Redis, prefix: str, ttl_seconds: int, max_messages: int):
        self.redis, self.prefix, self.ttl, self.max_messages = redis, prefix, ttl_seconds, max_messages

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    def history(self, session_id: str, last: int) -> list[StoredMessage]:
        """The last `last` messages, oldest first. Unknown or expired sessions are empty."""
        if last <= 0:
            return []
        try:
            raw = self.redis.lrange(self.key(session_id), -last, -1)
        except RedisError as e:
            raise SessionUnavailable(f"Session store unavailable: {e}") from e
        return [StoredMessage.model_validate_json(m) for m in raw]

    def append_turn(self, session_id: str, user: StoredMessage, assistant: StoredMessage) -> None:
        key = self.key(session_id)
        try:
            with self.redis.pipeline(transaction=True) as pipe:  # MULTI ... EXEC: all or nothing
                pipe.rpush(key, user.model_dump_json(), assistant.model_dump_json())
                pipe.ltrim(key, -self.max_messages, -1)
                pipe.expire(key, self.ttl)
                pipe.execute()
        except RedisError as e:
            raise SessionUnavailable(f"Session store unavailable: {e}") from e

    def get(self, session_id: str) -> tuple[list[StoredMessage], int]:
        """All stored messages and the seconds until the session expires."""
        key = self.key(session_id)
        try:
            with self.redis.pipeline(transaction=True) as pipe:
                pipe.lrange(key, 0, -1)
                pipe.ttl(key)
                raw, ttl = pipe.execute()
        except RedisError as e:
            raise SessionUnavailable(f"Session store unavailable: {e}") from e
        if not raw:
            raise SessionNotFound(f"Session {session_id!r} does not exist or has expired")
        return [StoredMessage.model_validate_json(m) for m in raw], ttl

    def delete(self, session_id: str) -> None:
        try:
            self.redis.delete(self.key(session_id))
        except RedisError as e:
            raise SessionUnavailable(f"Session store unavailable: {e}") from e


def create_session_store(redis: Redis, settings: Settings) -> SessionStore:
    return SessionStore(redis, settings.session_key_prefix, settings.session_ttl_seconds,
                        settings.session_max_stored_messages)


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.sessions
