"""The chat page: a single HTML file that calls POST /api/v1/chat from the browser.

Served by the API itself (same origin: no CORS, no separate server, no build step). The xAI key
never reaches the browser. The brand list comes from documents/manifest.json, so it always
matches the manuals that are ingested.
"""

import html
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from app.core.config import Settings, get_settings
from app.services.document_loader import load_manifest

PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"
BRAND_PLACEHOLDER = "<!--BRAND_OPTIONS-->"

router = APIRouter(include_in_schema=False)


@lru_cache
def render_page(documents_dir: str) -> str:
    options = "".join(
        f'<option value="{html.escape(m.brand)}">{html.escape(m.aliases[0] if m.aliases else m.brand)} '
        f'{html.escape(m.model)}</option>'
        for m in load_manifest(documents_dir))
    return PAGE.read_text(encoding="utf-8").replace(BRAND_PLACEHOLDER, options)


@router.get("/", response_class=HTMLResponse)
def chat_page(settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return HTMLResponse(render_page(settings.documents_dir))
