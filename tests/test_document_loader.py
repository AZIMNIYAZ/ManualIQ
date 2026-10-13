import json
from pathlib import Path

import pytest

from app.services.document_loader import DocumentLoadError, load_all_documents, load_manifest

REAL_DOCS = Path("documents")


def write_manifest(folder: Path, entries: list[dict]) -> None:
    (folder / "manifest.json").write_text(json.dumps({"documents": entries}), encoding="utf-8")


def entry(file: str, document_id: str, brand: str = "acme") -> dict:
    return {"file": file, "document_id": document_id, "brand": brand, "model": "X1"}


def test_loads_txt_and_md_as_single_page(tmp_path):
    (tmp_path / "faq.txt").write_text("Reset the router by holding the button.", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Setup\n\nConnect the cable.", encoding="utf-8")
    write_manifest(tmp_path, [entry("faq.txt", "acme_faq"), entry("guide.md", "acme_guide")])

    docs = load_all_documents(tmp_path)

    assert [d.document_id for d in docs] == ["acme_faq", "acme_guide"]
    assert docs[0].brand == "acme" and docs[0].source == "faq.txt"
    assert len(docs[0].pages) == 1 and docs[0].pages[0].page_number == 1
    assert "holding the button" in docs[0].pages[0].text


def test_missing_manifest(tmp_path):
    with pytest.raises(DocumentLoadError, match="Manifest not found"):
        load_manifest(tmp_path)


def test_missing_file(tmp_path):
    write_manifest(tmp_path, [entry("nope.pdf", "acme_x")])
    with pytest.raises(DocumentLoadError, match="not found"):
        load_manifest(tmp_path)


def test_duplicate_document_id(tmp_path):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("text", encoding="utf-8")
    write_manifest(tmp_path, [entry("a.txt", "same_id"), entry("b.txt", "same_id")])
    with pytest.raises(DocumentLoadError, match="Duplicate"):
        load_manifest(tmp_path)


def test_invalid_document_id(tmp_path):
    (tmp_path / "a.txt").write_text("text", encoding="utf-8")
    write_manifest(tmp_path, [entry("a.txt", "Has Spaces")])
    with pytest.raises(DocumentLoadError, match="Invalid manifest"):
        load_manifest(tmp_path)


def test_unsupported_file_type(tmp_path):
    (tmp_path / "a.docx").write_text("text", encoding="utf-8")
    write_manifest(tmp_path, [entry("a.docx", "acme_x")])
    with pytest.raises(DocumentLoadError, match="Unsupported"):
        load_manifest(tmp_path)


@pytest.mark.skipif(not (REAL_DOCS / "manifest.json").exists(), reason="documents/ not set up")
def test_real_manuals_load():
    docs = load_all_documents(REAL_DOCS)
    assert len(docs) == 5
    assert len({d.brand for d in docs}) == 5
    for d in docs:
        assert len(d.pages) > 100, d.document_id
        assert [p.page_number for p in d.pages] == list(range(1, len(d.pages) + 1))
        assert d.text_length > 50_000, d.document_id
