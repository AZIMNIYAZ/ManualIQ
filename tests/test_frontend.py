"""The chat page served at / (calls POST /api/v1/chat from the browser)."""

from app.api.frontend import BRAND_PLACEHOLDER, render_page


def test_page_is_served_with_the_manuals_brands(make_client):
    r = make_client().get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    assert "<title>Router Manual Assistant</title>" in page and 'fetch("/api/v1/chat"' in page
    assert BRAND_PLACEHOLDER not in page
    for brand, label in [("tplink", "TP-Link Archer AX55"), ("netgear", "Netgear Nighthawk RAX50"),
                         ("asus", "ASUS RT-AX58U V2"), ("dlink", "D-Link DIR-X1560"), ("tenda", "Tenda AX3000")]:
        assert f'<option value="{brand}">{label}' in page


def test_page_escapes_api_text_before_showing_it():
    """Answers and excerpts come from the manuals and the model: shown only after escaping."""
    page = render_page("documents")
    assert "const escape = (s) =>" in page
    assert "escape(s.excerpt)" in page and "escape(s.citation)" in page and "escape(r.answer)" in page


def test_page_is_not_in_the_api_schema(make_client):
    assert "/" not in make_client().get("/openapi.json").json()["paths"]
