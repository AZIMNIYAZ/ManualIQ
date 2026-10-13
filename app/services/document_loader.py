"""Loads the documents listed in documents/manifest.json.

Returns raw text, one Page per PDF page. Cleaning happens separately in
text_cleaner.py, so raw and cleaned text can be compared.

TXT and Markdown files have no pages, so their whole text is page 1.
"""

import json
import logging
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader

from app.models.document import Document, Manifest, ManifestEntry, Page

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# pypdf logs harmless font warnings ("fontTools is required...") for some manuals
logging.getLogger("pypdf").setLevel(logging.ERROR)


class DocumentLoadError(Exception):
    pass


def load_manifest(documents_dir: str | Path) -> list[ManifestEntry]:
    """Reads and validates the manifest: valid fields, unique IDs, files exist."""
    documents_dir = Path(documents_dir)
    path = documents_dir / MANIFEST_FILE
    if not path.exists():
        raise DocumentLoadError(f"Manifest not found: {path}")

    try:
        manifest = Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as e:
        raise DocumentLoadError(f"Invalid manifest {path}: {e}") from e

    entries = manifest.documents
    if not entries:
        raise DocumentLoadError(f"Manifest {path} lists no documents")

    ids = [e.document_id for e in entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise DocumentLoadError(f"Duplicate document_id in manifest: {duplicates}")

    for entry in entries:
        file_path = documents_dir / entry.file
        if not file_path.exists():
            raise DocumentLoadError(f"File listed in manifest not found: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(f"Unsupported file type: {entry.file} (supported: {sorted(SUPPORTED_EXTENSIONS)})")

    listed = {e.file for e in entries}
    unlisted = [
        p.name for p in documents_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.name not in listed
    ]
    if unlisted:
        logger.warning("Files in documents/ not in manifest (ignored)", extra={"fields": {"files": unlisted}})

    return entries


def _read_pages(path: Path) -> list[Page]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [Page(page_number=i, raw_text=page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    return [Page(page_number=1, raw_text=path.read_text(encoding="utf-8"))]


def load_document(entry: ManifestEntry, documents_dir: str | Path) -> Document:
    path = Path(documents_dir) / entry.file
    try:
        pages = _read_pages(path)
    except Exception as e:
        raise DocumentLoadError(f"Could not read {path}: {e}") from e

    doc = Document(
        document_id=entry.document_id,
        brand=entry.brand,
        model=entry.model,
        source=entry.file,
        pages=pages,
    )
    logger.info(
        "Loaded document",
        extra={"fields": {"document_id": doc.document_id, "pages": len(pages), "chars": doc.text_length}},
    )
    return doc


def load_all_documents(documents_dir: str | Path) -> list[Document]:
    return [load_document(entry, documents_dir) for entry in load_manifest(documents_dir)]
