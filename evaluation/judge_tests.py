"""Planted-error tests for the LLM judge ("can it catch a bad answer?").

Real answers were almost all grounded, so human grading alone cannot show whether the judge
detects ungrounded answers (Cohen's kappa is undefined without negatives). Here a real,
grounded answer gets exactly one deliberate error, made by code (not by an LLM), so the
correct verdict is known:

  wrong_menu        a menu path segment is changed           -> grounded must be false
  changed_value     an IP address or number is changed       -> grounded must be false
  other_brand_step  a step from another brand's manual added -> grounded false (brand mixing expected)
  invented_claim    a plausible but unsupported note added   -> grounded must be false
  wrong_citation    a [n] is moved to a different source      -> citations_correct must be false
"""

import re

MUTATIONS = ("wrong_menu", "changed_value", "other_brand_step", "invented_claim", "wrong_citation")

# Real menu paths from each manual, used as the "other brand" step
FOREIGN_STEPS = {
    "tplink": "Go to Advanced > System > Reboot and tick the Enable box of Reboot Schedule.",
    "netgear": "Select ADVANCED > Administration > Router Update and click the Check button.",
    "asus": "From the navigation panel, go to Advanced Settings > Firewall > General and select Yes.",
    "dlink": "In the Management menu, click System Admin, then click Reboot the Device.",
    "tenda": "Choose More > System Settings > Backup & Restore and click Backup.",
}
INVENTED_CLAIM = "Note: restart the router twice after saving, otherwise the change is not applied."

MENU_RE = re.compile(r"\b([A-Z][\w&/()-]*(?: [\w&/()-]+){0,3}) > ([\w&/()-]+(?: [\w&/()-]+){0,3})")
IP_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.)(\d{1,3})\b")
NUMBER_RE = re.compile(r"(?<![\d.])(\d{1,4})(\s?(?:seconds?|minutes?|hours?|characters?|clients?|devices?|Mbps))", re.I)
SOURCE_RE = re.compile(r"^\[(\d{1,2})\] Brand: (\w+) \|.*?, pp?\. (\d+)(?:[–-](\d+))?", re.M)
CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def wrong_menu(answer: str) -> str | None:
    m = MENU_RE.search(answer)
    if not m:
        return None
    return answer[:m.start(2)] + "Wireless Bridge Settings" + answer[m.end(2):]


def changed_value(answer: str) -> str | None:
    m = IP_RE.search(answer)
    if m:
        last = int(m.group(2))
        return answer[:m.start(2)] + str((last + 37) % 250 + 2) + answer[m.end(2):]
    m = NUMBER_RE.search(answer)
    if m:
        return answer[:m.start(1)] + str(int(m.group(1)) * 3 + 1) + answer[m.end(1):]
    return None


def _first_citation(answer: str) -> str:
    m = CITATION_RE.search(answer)
    return f" [{m.group(1)}]" if m else ""


def other_brand_step(answer: str, brand: str | None) -> str | None:
    if not brand:  # only meaningful for a single-brand answer
        return None
    foreign = next((b for b in FOREIGN_STEPS if b != brand), None)
    if foreign is None:
        return None
    return answer.rstrip() + f"\n\nThen, {FOREIGN_STEPS[foreign][0].lower()}{FOREIGN_STEPS[foreign][1:]}" + _first_citation(answer)


def invented_claim(answer: str) -> str:
    return answer.rstrip() + "\n\n" + INVENTED_CLAIM + _first_citation(answer)


def sources(context: str) -> dict[int, tuple[str, int, int]]:
    """[n] -> (brand, first page, last page), from the context headers."""
    return {int(m.group(1)): (m.group(2), int(m.group(3)), int(m.group(4) or m.group(3)))
            for m in SOURCE_RE.finditer(context)}


def wrong_citation(answer: str, context: str) -> str | None:
    """Moves every citation of the most-cited source to an uncited source from another manual
    or from pages that do not overlap (so the new source cannot support the same text)."""
    cited = [int(n) for n in CITATION_RE.findall(answer)]
    src = sources(context)
    if not cited:
        return None
    main = max(set(cited), key=cited.count)
    if main not in src:
        return None
    b, first, last = src[main]
    unrelated = [n for n, (b2, f2, l2) in sorted(src.items())
                 if n not in cited and (b2 != b or l2 < first or f2 > last)]
    if not unrelated:
        return None
    return CITATION_RE.sub(lambda m: f"[{unrelated[0]}]" if int(m.group(1)) == main else m.group(0), answer)


def mutate(kind: str, answer: str, brand: str | None, context: str) -> str | None:
    """The answer with one planted error, or None if this answer offers nothing to change."""
    if kind == "wrong_menu":
        out = wrong_menu(answer)
    elif kind == "changed_value":
        out = changed_value(answer)
    elif kind == "other_brand_step":
        out = other_brand_step(answer, brand)
    elif kind == "invented_claim":
        out = invented_claim(answer)
    elif kind == "wrong_citation":
        out = wrong_citation(answer, context)
    else:
        raise ValueError(kind)
    return out if out and out != answer else None


def detected(kind: str, verdict) -> bool:
    if kind == "wrong_citation":
        return verdict.citations_correct is False
    return verdict.grounded is False
