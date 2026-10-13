import os
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # bge tokenizer vocabulary is used as the dictionary

from app.models.document import Document, Page  # noqa: E402
from app.services.document_loader import load_all_documents  # noqa: E402
from app.services.text_cleaner import clean_document  # noqa: E402

FILLER = (
    "Open a web browser and enter the address shown on the label of your router. "
    "Click Save to apply the settings and wait for the router to restart before you continue."
)
REAL_DOCS = Path("documents")


def make_doc(*page_texts: str) -> Document:
    return Document(
        document_id="acme_x1", brand="acme", model="X1", source="acme.pdf",
        pages=[Page(page_number=i, raw_text=t) for i, t in enumerate(page_texts, start=1)],
    )


def filler_page(i: int) -> str:
    # Each page differs, like real pages; identical pages would look like repeated headers.
    return f"Step {i}: configure option number {i} for feature {chr(65 + i % 26)}{i}. {FILLER}"


def clean_one(text: str, pages_before: int = 0) -> Page:
    """Cleans a single page placed after some filler pages (so it is not 'near the start')."""
    doc = make_doc(*[filler_page(i) for i in range(pages_before)], text,
                   *[filler_page(100 + i) for i in range(20)])
    cleaned, _ = clean_document(doc)
    return cleaned.pages[pages_before]


# ---------------------------------------------------------------- A. page decisions

def test_empty_page_excluded():
    page = clean_one("12")
    assert page.status == "excluded" and page.reason == "empty" and page.clean_text is None


def test_short_pages_kept():
    """Short pages hold real steps and chapter titles; excluding them would split procedures."""
    assert clean_one("Chapter 13 VPN Server", pages_before=5).status == "indexed"
    assert clean_one("Step 4 Check the diagnosis result and click Optimize.", pages_before=5).status == "indexed"


def test_chinese_page_excluded():
    page = clean_one("依據低功率電波輻射性電機管理辦法 取得審驗證明之低功率射頻器材 非經核准 公司 商號或使用者均不得擅自變更頻率 " * 3)
    assert page.status == "excluded" and page.reason == "non_english"


def test_french_page_excluded():
    french = (
        "L'émetteur exempt de licence contenu dans le présent appareil est conforme aux exigences "
        "applicables aux appareils radio exempts de licence. L'exploitation est autorisée aux deux "
        "conditions suivantes: l'appareil ne doit pas produire de brouillage, et l'utilisateur de "
        "l'appareil doit accepter tout brouillage radioélectrique subi, même si le brouillage est "
        "susceptible d'en compromettre le fonctionnement pour la sécurité des personnes."
    )
    page = clean_one(french, pages_before=5)
    assert page.status == "excluded" and page.reason == "non_english"


def test_english_settings_table_kept():
    """Few common words (it's a table), but clearly English: must not be dropped."""
    table = (
        "Factory settings Parameter Default value Login IP address 192.168.0.1 Password No login "
        "password by default LAN parameters IP address 192.168.0.1 Subnet mask 255.255.255.0 DHCP "
        "server Enabled IP address pool 192.168.0.100 to 192.168.0.200 Wireless network name "
        "Tenda_XXXXXX Wireless password None Smart connect Enabled Guest network Disabled "
        "Firewall Enabled Remote management Disabled Time zone Automatic"
    )
    page = clean_one(table, pages_before=5)
    assert page.status == "indexed" and page.content_type == "normal"


def test_toc_with_heading_excluded():
    toc = "Contents\n" + "\n".join(f"Section {i} Setting up feature {i} ........................ {i * 3}" for i in range(1, 12))
    page = clean_one(toc, pages_before=5)
    assert page.status == "excluded" and page.reason == "toc"


def test_toc_layout_only_is_flagged_not_excluded():
    """Dot leaders in the middle of the manual without a Contents heading: keep and flag."""
    listing = "\n".join(f"Feature {i} supported on this model .................. {i}" for i in range(1, 12))
    page = clean_one(listing, pages_before=25)
    assert page.status == "indexed" and "toc_single_signal" in page.flags


def test_one_regulatory_family_is_not_enough():
    """'FCC Part 15 Class B' in a spec table is one piece of evidence, not three."""
    page = clean_one(FILLER + " Electromagnetic emissions: FCC Part 15 Class B.", pages_before=5)
    assert page.content_type == "normal"


def test_regulatory_page_tagged_but_indexed():
    text = (FILLER + " This device complies with FCC rules. It also meets Industry Canada requirements "
            "and carries the CE mark under the EU Declaration of Conformity.")
    page = clean_one(text, pages_before=5)
    assert page.status == "indexed" and page.content_type == "regulatory"


def test_ce_does_not_match_inside_words():
    text = FILLER + " Each device, service and interface can be configured. Contact customer service."
    page = clean_one(text, pages_before=5)
    assert page.content_type == "normal"


# ---------------------------------------------------------------- B. text repairs

TOPICS = ["wireless", "parental", "guest", "firmware", "VPN", "USB", "IPv6", "DDNS"]


def test_headers_and_page_numbers_removed_but_content_kept():
    pages = [f"Acme X1 User Manual {n}\nTo set up {topic} settings, {FILLER}\n{n}"
             for n, topic in enumerate(TOPICS, start=1)]
    cleaned, report = clean_document(make_doc(*pages))
    text = cleaned.pages[3].clean_text
    assert "Acme X1 User Manual" not in text
    assert not text.strip().endswith("4")
    assert text.startswith("To set up firmware settings")
    assert report.header_lines["Acme X# User Manual #"] == 8


def test_page_left_empty_after_header_removal_is_excluded():
    pages = [f"Acme X1 User Manual {n}\nTo set up {topic} settings, {FILLER}\n{n}"
             for n, topic in enumerate(TOPICS, start=1)]
    pages[4] = "Acme X1 User Manual 5\n5"
    cleaned, _ = clean_document(make_doc(*pages))
    assert cleaned.pages[4].status == "excluded" and cleaned.pages[4].reason == "empty"


def test_bullets_replaced_only_at_line_start():
    page = clean_one(FILLER + "\n Ensure the router is powered on.\n⚫ Solid green: signal is good.\nPCB ○ ○ ○ ○")
    assert "- Ensure the router is powered on." in page.clean_text
    assert "- Solid green: signal is good." in page.clean_text
    assert "PCB ○ ○ ○ ○" in page.clean_text  # table symbols are meaningful, not bullets


def test_apple_logo_replaced_with_words():
    page = clean_one(FILLER + "\nGo to  > System Preferences > Network.")
    assert "Go to Apple menu > System Preferences > Network." in page.clean_text


def test_kerning_split_repaired_with_capital_kept():
    page = clean_one(FILLER + "\nUse T raceroute to test the connection. Y our router supports it.")
    assert "Use Traceroute to test the connection. Your router supports it." in page.clean_text


def test_split_word_repaired():
    # In a real manual "select" and "protect" appear unsplit many times
    usage = " Select a mode. Protect the router. Select Save. Protect your data."
    page = clean_one(FILLER + usage + "\nSelect the option and selec t Enable to protec t your network.")
    assert "select Enable" in page.clean_text and "protect your network" in page.clean_text


def test_single_trailing_letter_needs_evidence_in_manual():
    """'the e' must not become 'thee': the joined word never appears in the manual."""
    page = clean_one(FILLER + "\nSet the e mail address in the settings.")
    assert "the e mail" in page.clean_text


@pytest.mark.parametrize("phrase", [
    "Log in to the router",          # "log in" / "login" both exist
    "Click Set up to continue",      # "set up" / "setup"
    "class A, class B and class C",  # single letter + real word
    "Router B and then the server",
    "the router’s port settings",    # apostrophe: not "sport"
])
def test_real_word_pairs_not_joined(phrase):
    page = clean_one(f"{FILLER}\n{phrase}. The login page and the setup wizard open. Use the router port.")
    assert phrase in page.clean_text


def test_hyphenated_line_break():
    page = clean_one(FILLER + " The configuration is saved.\nChange the config-\nuration and use the drop-\ndown list.")
    assert "Change the configuration" in page.clean_text
    assert "drop-down list" in page.clean_text  # "dropdown" not known: hyphen kept


# ---------------------------------------------------------------- guarantees

def test_raw_text_and_page_numbers_preserved():
    doc = make_doc("Too short", FILLER, FILLER)
    cleaned, _ = clean_document(doc)
    assert [p.page_number for p in cleaned.pages] == [1, 2, 3]
    assert [p.raw_text for p in cleaned.pages] == [p.raw_text for p in doc.pages]
    assert cleaned.pages[0].status == "excluded" and cleaned.pages[1].status == "indexed"


def test_cleaning_is_deterministic():
    doc = make_doc(*[f"Acme X1 User Manual {n}\nT o selec t a mode, {FILLER}\n{n}" for n in range(1, 9)])
    first = clean_document(doc)
    second = clean_document(doc)
    assert first[0] == second[0] and first[1] == second[1]


@pytest.mark.skipif(not (REAL_DOCS / "manifest.json").exists(), reason="documents/ not set up")
def test_real_manuals():
    docs = {d.document_id: d for d in load_all_documents(REAL_DOCS)}
    results = {did: clean_document(d) for did, d in docs.items()}

    # deterministic on real data
    tplink = docs["tplink_archer_ax55"]
    assert clean_document(tplink) == results["tplink_archer_ax55"]

    def page(did, n):
        return results[did][0].pages[n - 1]

    # valuable pages kept as normal content
    assert page("tenda_ax3000", 154).status == "indexed"  # factory settings table
    assert page("tenda_ax3000", 154).content_type == "normal"
    assert page("netgear_rax50", 159).content_type == "normal"  # factory defaults (mentions FCC Part 15 Class B)
    # known exclusions
    assert page("tplink_archer_ax55", 120).reason == "non_english"
    assert page("asus_rt_ax58u", 3).reason == "toc"
    assert page("dlink_dir_x1560", 165).reason == "non_english"
    # no private-use characters left in indexed text
    for cleaned, report in results.values():
        assert report.repair_counts["private_use_chars_left"] == 0
