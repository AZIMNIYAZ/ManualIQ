from fastapi import Request
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.core.config import Settings


def create_redis_client(settings: Settings) -> Redis:
    # Redis connects lazily on the first command, so this works even if Redis is down.
    # redis-py retries 10 times with backoff by default (~50 s when Redis is down);
    # one immediate retry keeps a failing call to about 2x the timeout.
    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
        socket_connect_timeout=settings.redis_timeout_seconds,
        socket_timeout=settings.redis_timeout_seconds,
        retry=Retry(NoBackoff(), retries=1),
    )


def get_redis(request: Request) -> Redis:
    """FastAPI dependency: the shared client created at startup."""
    return request.app.state.redis
