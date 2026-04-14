"""
data/cleaner.py — Clean extracted PDF/HTML text for use as ML features.

Removes:
  - Price table rows (numeric-heavy lines)
  - Repeated headers / footers ("continued on page X", copyright lines)
  - Lines that are mostly ticker codes / symbols
  - Blank lines and excessive whitespace

Keeps:
  - Sentences with real words (narrative content)
  - Paragraphs describing market events, supply/demand, geopolitics

Public API:
    clean_report_text(raw_text)  -> str   (full cleaned text)
    extract_narrative(raw_text)  -> str   (narrative sentences only, best for ML)
"""
from __future__ import annotations

import re


# ── Patterns to remove ────────────────────────────────────────────────────────

# Lines that are mostly numbers, prices, bid/ask ranges
_PRICE_ROW_RE = re.compile(
    r"^[\w\s\(\)/\-\.]*"          # optional label
    r"\s+[A-Z0-9]{4,8}"           # ticker code
    r"\s+[\d\.\-\+\s/–]+$",       # numbers only after ticker
    re.IGNORECASE,
)

# Lines with too many numbers relative to words
_NUMERIC_HEAVY_RE = re.compile(r"^[^a-zA-Z]*$")  # no letters at all

# Repeated boilerplate
_BOILERPLATE_PATTERNS = [
    re.compile(r"\(continued on page \d+\)", re.IGNORECASE),
    re.compile(r"www\.spglobal\.com", re.IGNORECASE),
    re.compile(r"©\s*\d{4}\s*by\s*S&P", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"^\s*page \d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),                    # lone page numbers
    re.compile(r"^bid\s+ask\s+mid", re.IGNORECASE),
    re.compile(r"^mid\s+change", re.IGNORECASE),
    re.compile(r"^\$\s*/\s*b(arrel)?", re.IGNORECASE),
    re.compile(r"^pga page \d+", re.IGNORECASE),
    re.compile(r"platts\s+connect", re.IGNORECASE),
    re.compile(r"^\s*\(?\d{1,3}\s*\$\s*/\s*[bm]\)?", re.IGNORECASE),  # "(110 $/b)"
]

# Section headers that mark the start of price tables (skip until next narrative)
_TABLE_SECTION_HEADERS = re.compile(
    r"^(middle east|north sea|west africa|latin america|"
    r"key benchmarks|forward dated brent|brent/wti spreads|"
    r"crude oil differentials|market table|price table|"
    r"bid\s+ask\s+mid|contents)\s*$",
    re.IGNORECASE,
)

# Narrative section markers — resume keeping lines after these
_NARRATIVE_MARKERS = re.compile(
    r"(market commentary|analysis|outlook|summary|"
    r"supply|demand|opec|geopolit|refinery|pipeline|"
    r"disruption|sanction|production|export|import)",
    re.IGNORECASE,
)


def _is_price_table_line(line: str) -> bool:
    """Return True if this line looks like a price table row."""
    stripped = line.strip()
    if not stripped:
        return False

    # Hard cap — very long lines are never price table rows (they're narrative)
    if len(stripped) > 300:
        return False

    # No letters at all → pure numbers/symbols
    if _NUMERIC_HEAVY_RE.match(stripped):
        return True

    # Count digits vs letters
    digits = sum(1 for c in stripped if c.isdigit())
    letters = sum(1 for c in stripped if c.isalpha())
    total = len(stripped)
    if total > 10 and digits / total > 0.45:
        return True

    # Matches price row pattern — only apply to short lines to avoid backtracking
    if len(stripped) <= 150 and _PRICE_ROW_RE.match(stripped):
        return True

    return False


def _is_boilerplate(line: str) -> bool:
    """Return True if this line is a boilerplate header/footer."""
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(line):
            return True
    return False


def _alpha_ratio(text: str) -> float:
    """Fraction of alphabetic characters in text."""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


# ── Public API ────────────────────────────────────────────────────────────────

def clean_report_text(raw_text: str) -> str:
    """
    Remove price tables, boilerplate, HTML tags, and numeric noise from extracted text.
    Returns cleaned text suitable for storage in body_text.
    """
    if not raw_text:
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", raw_text)
    # Decode common HTML entities
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"'))

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned: list[str] = []
    in_table_section = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (will re-add paragraph breaks later)
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        # Boilerplate always removed
        if _is_boilerplate(stripped):
            continue

        # Detect table section headers → enter table-skip mode
        if _TABLE_SECTION_HEADERS.match(stripped):
            in_table_section = True
            continue

        # Narrative markers → exit table-skip mode
        if in_table_section and _NARRATIVE_MARKERS.search(stripped):
            in_table_section = False

        if in_table_section:
            continue

        # Price table rows removed even outside table sections
        if _is_price_table_line(stripped):
            continue

        cleaned.append(stripped)

    # Collapse multiple blank lines
    result_lines: list[str] = []
    prev_blank = False
    for line in cleaned:
        if line == "":
            if not prev_blank:
                result_lines.append("")
            prev_blank = True
        else:
            result_lines.append(line)
            prev_blank = False

    return "\n".join(result_lines).strip()


def extract_narrative(raw_text: str) -> str:
    """
    Extract only narrative sentences from report text.
    More aggressive than clean_report_text — keeps only lines that look like
    real English sentences (>40% alphabetic, >5 words).

    Best used as ML feature input (body_snippet in train.py).
    """
    if not raw_text:
        return ""

    cleaned = clean_report_text(raw_text)
    lines = cleaned.split("\n")
    narrative: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Must be mostly alphabetic
        if _alpha_ratio(stripped) < 0.40:
            continue
        # Must have at least 5 words
        words = stripped.split()
        if len(words) < 5:
            continue
        narrative.append(stripped)

    return " ".join(narrative).strip()


def is_narrative_text(text: str, min_alpha_ratio: float = 0.40) -> bool:
    """
    Return True if text looks like narrative (not a price table).
    Used by train.py to decide whether to include body_text as a feature.
    """
    if not text or len(text) < 50:
        return False
    return _alpha_ratio(text) >= min_alpha_ratio
