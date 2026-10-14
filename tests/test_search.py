import os

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.models.search import SearchRequest  # noqa: E402
from app.services.brand_detection import build_alias_table, detect_brands  # noqa: E402
from app.services.ingestion_service import ingest  # noqa: E402
from app.services.retrieval_service import InvalidBrand, KnowledgeBaseEmpty, citation, search  # noqa: E402
from tests.test_ingestion import FakeEmbedder, make_settings, needs_qdrant, qdrant, write_docs  # noqa: E402,F401

ALIASES = build_alias_table({
    "tplink": ["TP-Link", "TPLink", "TP Link", "Archer AX55", "Archer"],
    "netgear": ["Netgear", "Nighthawk", "RAX50"],
    "asus": ["ASUS", "RT-AX58U"],
    "dlink": ["D-Link", "DLink", "D Link", "DIR-X1560"],
    "tenda": ["Tenda", "RX12 Pro", "TX12 Pro"],
})


# ---------------------------------------------------------------- brand detection

@pytest.mark.parametrize("query, expected", [
    ("How do I reset my TP-Link router?", ["tplink"]),
    ("tplink port forwarding", ["tplink"]),
    ("Archer AX55 firmware upgrade", ["tplink"]),
    ("set up the guest network on my Nighthawk", ["netgear"]),
    ("Netgear Nighthawk RAX50 parental controls", ["netgear"]),  # deduplicated
    ("d-link dir-x1560 wall mounting", ["dlink"]),
    ("Is ASUS or Netgear easier to set up?", ["asus", "netgear"]),
    ("How do I reset my router?", []),
    ("Which link light shows the connection?", []),  # "link" alone is not D-Link
    ("see https://example.com/D-Link for details on my Netgear router", ["netgear"]),  # URL ignored
    ("I cannot open www.tplinkwifi.net or routerlogin.net", []),  # domains ignored
    ("My Linksys router keeps rebooting", []),  # unknown brand in the question: not an error
])
def test_detect_brands(query, expected):
    assert detect_brands(query, ALIASES) == expected


def test_citation_format():
    assert citation("netgear_rax50.pdf", 139, 139) == "netgear_rax50.pdf, p. 139"
    assert citation("netgear_rax50.pdf", 139, 140) == "netgear_rax50.pdf, pp. 139–140"


# ---------------------------------------------------------------- request validation

@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "   "}, {"query": "x" * 1001},
    {"query": "reset", "top_k": 0}, {"query": "reset", "top_k": 21}, {"query": "reset", "brand": []},
])
def test_invalid_requests(payload):
    with pytest.raises(ValidationError):
        SearchRequest(**payload)


def test_brand_is_normalised():
    assert SearchRequest(query="reset", brand=" NetGear ").brand == ["netgear"]
    assert SearchRequest(query="reset", brand=["tenda", "ASUS", "asus"]).brand == ["asus", "tenda"]


# ---------------------------------------------------------------- with Qdrant (rag_test_* only)

def write_docs_with_regulatory(folder):
    write_docs(folder)
    (folder / "legal.txt").write_text(
        "Regulatory information. This device complies with Part 15 of the FCC rules and with Industry Canada "
        "licence-exempt standards. The EU Declaration of Conformity is available online and the CE mark is "
        "shown on the label. Operation is subject to the conditions that this device may not cause harmful "
        "interference and must accept any interference received.", encoding="utf-8")
    import json
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["documents"].append({"file": "legal.txt", "document_id": "acme_legal", "brand": "acme", "model": "X1"})
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


TEST_ALIASES = build_alias_table({"acme": ["Acme"], "zeta": ["Zeta"]})


@pytest.fixture
def ingested(qdrant, tmp_path):
    write_docs_with_regulatory(tmp_path)
    settings = make_settings(tmp_path)
    assert ingest(qdrant, FakeEmbedder(), settings).status == "success"
    return qdrant, settings


@needs_qdrant
def test_empty_knowledge_base_is_409(qdrant, tmp_path):
    with pytest.raises(KnowledgeBaseEmpty, match="not been ingested"):
        search(qdrant, FakeEmbedder(), make_settings(tmp_path), SearchRequest(query="reset"), TEST_ALIASES)


@needs_qdrant
def test_regulatory_hidden_by_default_and_included_on_request(ingested):
    client, settings = ingested
    default = search(client, FakeEmbedder(), settings, SearchRequest(query="rules", top_k=20), TEST_ALIASES)
    assert default.results and all(r.content_type == "normal" for r in default.results)
    included = search(client, FakeEmbedder(), settings,
                      SearchRequest(query="rules", top_k=20, include_regulatory=True), TEST_ALIASES)
    assert {r.content_type for r in included.results} == {"normal", "regulatory"}


@needs_qdrant
def test_detected_brand_filters_results(ingested):
    client, settings = ingested
    r = search(client, FakeEmbedder(), settings, SearchRequest(query="Zeta firewall setup", top_k=20), TEST_ALIASES)
    assert r.filters.brands == ["zeta"] and r.filters.brand_source == "detected"
    assert {x.brand for x in r.results} == {"zeta"}


@needs_qdrant
def test_request_brand_overrides_detection(ingested):
    client, settings = ingested
    r = search(client, FakeEmbedder(), settings,
               SearchRequest(query="Zeta firewall setup", brand="acme", top_k=20), TEST_ALIASES)
    assert r.filters.brand_source == "request" and {x.brand for x in r.results} == {"acme"}


@needs_qdrant
def test_unknown_request_brand_is_422_with_valid_list(ingested):
    client, settings = ingested
    with pytest.raises(InvalidBrand, match=r"valid brands: \['acme', 'zeta'\]"):
        search(client, FakeEmbedder(), settings, SearchRequest(query="reset", brand="linksys"), TEST_ALIASES)


@needs_qdrant
def test_results_ranked_and_cited(ingested):
    client, settings = ingested
    r = search(client, FakeEmbedder(), settings, SearchRequest(query="guest network", top_k=5), TEST_ALIASES)
    assert len(r.results) == 5 and [x.rank for x in r.results] == [1, 2, 3, 4, 5]
    assert [x.score for x in r.results] == sorted((x.score for x in r.results), reverse=True)
    assert all(x.citation.startswith(x.source) for x in r.results)
    assert r.query_tokens > 0 and r.query_truncated is False
    assert set(r.timings_ms) == {"checks", "embed", "search", "total"}


# ---------------------------------------------------------------- API

@needs_qdrant
def test_api_search_and_errors(ingested):
    from app.api.search import get_aliases
    from app.core.config import get_settings
    from app.database.qdrant import get_qdrant
    from app.main import app
    from app.services.embedding_service import get_embedder

    client, settings = ingested
    app.dependency_overrides.update({
        get_qdrant: lambda: client, get_embedder: lambda: FakeEmbedder(),
        get_settings: lambda: settings, get_aliases: lambda: TEST_ALIASES,
    })
    try:
        api = TestClient(app)
        ok = api.post("/api/v1/search", json={"query": "Acme parental controls", "top_k": 3})
        assert ok.status_code == 200 and len(ok.json()["results"]) == 3
        assert ok.json()["filters"] == {"brands": ["acme"], "brand_source": "detected", "include_regulatory": False}
        assert api.post("/api/v1/search", json={"query": "reset", "brand": "linksys"}).status_code == 422
        assert api.post("/api/v1/search", json={"query": "  "}).status_code == 422
        assert api.post("/api/v1/search", json={"query": "reset", "top_k": 50}).status_code == 422
    finally:
        app.dependency_overrides.clear()
