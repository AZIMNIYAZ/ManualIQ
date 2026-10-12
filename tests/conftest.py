import pytest
from fastapi.testclient import TestClient

from app.database.qdrant import get_qdrant
from app.database.redis import get_redis
from app.main import app


class FakeQdrant:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def get_collections(self):
        if self.fail:
            raise ConnectionError("Connection refused")
        return []


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def ping(self):
        if self.fail:
            raise ConnectionError("Connection refused")
        return True


@pytest.fixture
def client():
    """Real app with real Qdrant/Redis clients (needs `docker compose up -d`)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def make_client():
    """App with fake Qdrant/Redis, so tests run without Docker."""

    def _make(qdrant_fails: bool = False, redis_fails: bool = False) -> TestClient:
        app.dependency_overrides[get_qdrant] = lambda: FakeQdrant(fail=qdrant_fails)
        app.dependency_overrides[get_redis] = lambda: FakeRedis(fail=redis_fails)
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()
