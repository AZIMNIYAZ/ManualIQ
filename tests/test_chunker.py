import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.models.document import Document, Page  # noqa: E402
from app.services.chunker import chunk_documents  # noqa: E402
from app.services.document_loader import load_all_documents  # noqa: E402
from app.services.text_cleaner import clean_documents  # noqa: E402
from app.services.tokenizer import count_tokens  # noqa: E402

REAL_DOCS = Path("documents")


def page(n: int, text: str, status: str = "indexed", content_type: str = "normal") -> Page:
    return Page(page_number=n, raw_text=text, clean_text=text if status == "indexed" else None,
                status=status, content_type=content_type, reason=None if status == "indexed" else "toc")


def doc(*pages: Page) -> Document:
    return Document(document_id="acme_x1", brand="acme", model="X1", source="acme.pdf", pages=list(pages))


def procedure(name: str, steps: int, page_marker: str = "") -> str:
    lines = [f"Configure {name}", f"This section explains how to configure {name} on the router.", f"To set up {name}:"]
    lines += [f"{i}. Open the {name} page and set option {i} to the value you need{page_marker}." for i in range(1, steps + 1)]
    return "\n".join(lines)


# ---------------------------------------------------------------- guarantees

def test_no_chunk_exceeds_size_and_nothing_lost():
    d = doc(page(1, procedure("parental controls", 8)), page(2, procedure("guest network", 8)))
    chunks, _ = chunk_documents([d], chunk_size=80, chunk_overlap=0)
    assert len(chunks) > 2
    assert all(c.token_count <= 80 and count_tokens(c.text) == c.token_count for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for line in (procedure("parental controls", 8) + "\n" + procedure("guest network", 8)).splitlines():
        assert line in joined


def test_chunks_never_cross_excluded_pages_or_content_types():
    d = doc(
        page(1, procedure("wireless", 4)),
        page(2, procedure("firewall", 4)),
        page(3, "Contents ........ 4", status="excluded"),
        page(4, procedure("VPN", 4)),
        page(5, "This device complies with FCC rules and the EU Declaration of Conformity. " * 3,
             content_type="regulatory"),
    )
    chunks, report = chunk_documents([d], chunk_size=120, chunk_overlap=20)
    assert report.segments == 3
    for c in chunks:
        assert not (c.page_start <= 3 <= c.page_end)  # never spans the excluded page
        if c.page_end == 5 or c.page_start == 5:
            assert c.page_start == c.page_end == 5 and c.content_type == "regulatory"
        else:
            assert c.content_type == "normal"


def test_page_ranges_are_correct():
    """One long procedure continuing over pages 10-13, three steps per page."""
    steps = [f"{i}. Open the settings page and set option {i} to the value you need [page {10 + (i - 1) // 3}]."
             for i in range(1, 13)]
    pages = [page(10, "To set up parental controls:\n" + "\n".join(steps[0:3]))]
    pages += [page(10 + n, "\n".join(steps[3 * n:3 * n + 3])) for n in range(1, 4)]
    chunks, _ = chunk_documents([doc(*pages)], chunk_size=100, chunk_overlap=0)
    for c in chunks:
        pages_in_text = [int(n) for n in re.findall(r"\[page (\d+)\]", c.text)]
        if pages_in_text:
            assert c.page_start <= min(pages_in_text) and max(pages_in_text) <= c.page_end
    assert any(c.page_end > c.page_start for c in chunks)  # continuous chunking spans pages


def test_overlap_repeats_whole_units():
    d = doc(page(1, procedure("parental controls", 12)))
    with_overlap, report = chunk_documents([d], chunk_size=80, chunk_overlap=25)
    for prev, nxt in zip(with_overlap, with_overlap[1:]):
        first_line = nxt.text.splitlines()[0]
        assert first_line in prev.text.splitlines()  # starts with a whole line from the previous chunk
    assert 0 < report.avg_overlap_tokens <= 25

    no_overlap, report0 = chunk_documents([d], chunk_size=80, chunk_overlap=0)
    for prev, nxt in zip(no_overlap, no_overlap[1:]):
        assert nxt.text.splitlines()[0] not in prev.text.splitlines()
    assert report0.avg_overlap_tokens == 0


def test_cuts_land_on_steps_or_headings():
    d = doc(page(1, procedure("parental controls", 10) + "\n" + procedure("guest network", 10)))
    chunks, _ = chunk_documents([d], chunk_size=90, chunk_overlap=0)
    for c in chunks:
        first = c.text.splitlines()[0]
        assert re.match(r"^(\d+\. |Configure |To set up )", first), first


def test_heading_and_intro_never_end_a_chunk():
    d = doc(page(1, procedure("parental controls", 9) + "\n" + procedure("guest network", 9)))
    chunks, report = chunk_documents([d], chunk_size=90, chunk_overlap=0)
    for c in chunks[:-1]:
        last = c.text.splitlines()[-1]
        assert not last.startswith(("Configure ", "To set up ")), last
    assert report.ending_with_title == 0


def test_oversized_unit_is_split():
    long_line = " ".join(f"Sentence number {i} describes one more router setting in detail." for i in range(60))
    d = doc(page(1, long_line))
    chunks, report = chunk_documents([d], chunk_size=100, chunk_overlap=0)
    assert report.oversized_units_split == 1
    assert all(c.token_count <= 100 for c in chunks)
    assert "Sentence number 59" in chunks[-1].text


def test_wrapped_lines_and_bare_step_numbers_are_merged():
    text = ("To log in:\n1. Launch a web browser from a computer that is connected to the\nrouter network.\n"
            "Step 3\nTap the Cloud Service icon.\n" + procedure("filler", 6))
    chunks, _ = chunk_documents([doc(page(1, text))], chunk_size=300, chunk_overlap=0)
    assert "1. Launch a web browser from a computer that is connected to the router network." in chunks[0].text
    assert "Step 3 Tap the Cloud Service icon." in chunks[0].text


def test_metadata():
    d = doc(page(7, procedure("port forwarding", 4)))
    (c,), _ = chunk_documents([d], chunk_size=300, chunk_overlap=0)
    assert c.chunk_id == "acme_x1_0000" and c.brand == "acme" and c.model == "X1"
    assert (c.page_start, c.page_end) == (7, 7) and c.chunking == "300/0"
    assert c.section == "Configure port forwarding"


def test_deterministic():
    d = doc(*[page(n, procedure(f"feature{n}", 5)) for n in range(1, 6)])
    assert chunk_documents([d], 120, 30) == chunk_documents([d], 120, 30)


@pytest.mark.parametrize("size, overlap", [(0, 0), (501, 60), (350, 350), (350, -1)])
def test_invalid_settings_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([], size, overlap)


# ---------------------------------------------------------------- real manuals

@pytest.fixture(scope="module")
def cleaned_manuals():
    if not (REAL_DOCS / "manifest.json").exists():
        pytest.skip("documents/ not set up")
    docs, _ = clean_documents(load_all_documents(REAL_DOCS))
    return docs


@pytest.mark.parametrize("size, overlap", [(200, 40), (350, 0), (350, 60), (480, 80)])
def test_real_manuals_respect_limits(cleaned_manuals, size, overlap):
    chunks, report = chunk_documents(cleaned_manuals, size, overlap)
    assert max(c.token_count for c in chunks) <= size
    assert report.ending_with_title == 0
    good_cuts = sum(v for k, v in report.cut_types.items() if k != "mid_sentence")
    assert good_cuts / sum(report.cut_types.values()) >= 0.9
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_real_manuals_settings_differ(cleaned_manuals):
    """The point of continuous chunking: different settings give clearly different collections."""
    counts = [chunk_documents(cleaned_manuals, s, o)[1].chunks for s, o in [(200, 40), (350, 60), (480, 80)]]
    assert counts[0] > counts[1] * 1.3 and counts[1] > counts[2] * 1.15
