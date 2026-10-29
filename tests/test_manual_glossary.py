"""Step 12: the manuals' feature-name glossary for the retrieval fallback."""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.models.document import Document, Page  # noqa: E402
from app.services.manual_glossary import (GLOSSARY_FILE, GLOSSARY_VERSION, build_glossary, clean_entry,  # noqa: E402
                                          glossary_entries, glossary_text, load_glossary)

TOC = """Contents
Chapter 6  Wireless Settings                                    27
6. 2. Schedule Y our Wireless Function  . . . . . . . . . . . . 30
7.3 Add a client to the blacklist ......................... 58
Enable the network access control list, set the access rule, and
allow or block connected devices...............................43
A. Wired connection ..................................................12
iOS Devices ..........................................................134
iii
7.3 Add a client to the blacklist ......................... 58"""


def doc(brand: str = "tenda", toc: str = TOC) -> Document:
    return Document(document_id=f"{brand}_x", brand=brand, model="X", source="x.pdf", pages=[
        Page(page_number=3, raw_text=toc, status="excluded", reason="toc"),
        Page(page_number=58, raw_text="7.3 Not a contents page", clean_text="x", status="indexed"),
    ])


@pytest.mark.parametrize("line, expected", [
    ("3.6.2 URL Filter ..................41", "URL Filter"),
    ("1. 2. 1. T op Panel  . . . . . . . 4", "Top Panel"),
    ("Chapter 4  Set Up Internet Connection          13", "Set Up Internet Connection"),
    ("QoS Engine............................................................64", "QoS Engine"),
    ("Table of Contents", None),
    ("iv", None),
    ("iiiD-Link DIR-X1560 User Manual", None),
    ("3", None),
])
def test_clean_entry(line, expected):
    assert clean_entry(line) == expected


def test_entries_from_contents_pages_only_with_wraps_joined_and_no_repeats():
    entries = glossary_entries(doc())
    assert entries == [
        "Wireless Settings", "Schedule Your Wireless Function", "Add a client to the blacklist",
        "Enable the network access control list, set the access rule, and allow or block connected devices",
        "Wired connection", "iOS Devices",
    ]


def test_glossary_text_limits_to_detected_brands():
    glossary = build_glossary([doc("tenda"), doc("asus", "3.10.6 Wireless MAC Filter .... 85")])
    names = {"tenda": "Tenda AX3000", "asus": "ASUS RT-AX58U"}
    assert glossary_text(glossary, names, ["asus"]) == "ASUS RT-AX58U: Wireless MAC Filter"
    both = glossary_text(glossary, names, [])
    assert both.startswith("Tenda AX3000: Wireless Settings; ") and both.endswith("\nASUS RT-AX58U: Wireless MAC Filter")


def test_load_glossary_rejects_missing_or_old_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_glossary(str(tmp_path))
    (tmp_path / GLOSSARY_FILE).write_text(json.dumps({"glossary_version": "0.9", "manuals": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_glossary(str(tmp_path))


def test_committed_glossary_has_the_manuals_feature_names():
    """documents/glossary.json is current and contains each manual's own wording."""
    data = load_glossary("documents")
    glossary = data["manuals"]
    assert set(glossary) == set(data["names"]) == {"tplink", "netgear", "asus", "dlink", "tenda"}
    assert data["glossary_version"] == GLOSSARY_VERSION
    assert data["names"]["dlink"] == "D-Link DIR-X1560"
    assert "Wireless MAC Filter" in glossary["asus"] and "QoS Engine" in glossary["dlink"]
