import socket

import pytest


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", port)) == 0


services_running = pytest.mark.skipif(
    not (_port_open(6333) and _port_open(6379)),
    reason="Qdrant/Redis not running (docker compose up -d)",
)


# --- Fake dependencies: no Docker needed ---

def test_all_ok(make_client):
    r = make_client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["llm"] in ("configured", "not_configured")  # depends on your .env; never fails health
    del body["llm"]
    assert body == {
        "status": "ok", "api": "ok", "qdrant": "ok", "redis": "ok",
        "collection": {"name": "rag_350_60", "exists": False, "points": None},  # not ingested: still 200
        "details": {},
    }


def test_qdrant_down_returns_503(make_client):
    r = make_client(qdrant_fails=True).get("/health")
    body = r.json()
    assert r.status_code == 503
    assert body["status"] == "degraded"
    assert body["qdrant"] == "down" and body["redis"] == "ok"
    assert "Connection refused" in body["details"]["qdrant"]


def test_redis_down_returns_503(make_client):
    r = make_client(redis_fails=True).get("/health")
    body = r.json()
    assert r.status_code == 503
    assert body["redis"] == "down" and body["qdrant"] == "ok"
    assert "redis" in body["details"]


def test_request_id_header(make_client):
    r = make_client().get("/health")
    assert len(r.headers["X-Request-ID"]) > 0


def test_request_id_is_echoed(make_client):
    r = make_client().get("/health", headers={"X-Request-ID": "my-test-id"})
    assert r.headers["X-Request-ID"] == "my-test-id"


# --- Real services ---

@services_running
def test_health_with_real_services(client):
    r = client.get("/health")
    assert r.status_code == 200, r.json()
    assert r.json()["status"] == "ok"
