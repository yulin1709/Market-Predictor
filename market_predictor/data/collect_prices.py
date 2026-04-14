"""
data/collect_prices.py - Extract benchmark prices from S&P report tables.

Source of truth:
    S&P Global report text saved in articles.body_text.

This parser is tuned for report pages that contain a "Key benchmarks" table
with rows like:
    Dubai (May) PCAAT00 102.54 102.58 102.55 -14.450

It uses:
    - Mid column -> price
    - Change column -> converted to pct_change_24h as an absolute move

Fallback:
    If Change is missing, pct_change_24h is computed from consecutive Mid values.
"""
import os
import re
import sqlite3
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(APP_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "").strip().lower() in {"1", "true", "yes"}

TARGET_SYMBOLS = {
    "Dubai": [
        "dubai (may)",
        "dubai (jun)",
        "dubai (jul)",
        "dubai",
    ],
    "Brent": [
        "brent (dated)",
        "dated north sea light",
        "dated brent",
        "brent",
    ],
    "WTI": [
        "wti fob usgc",
        "wti",
    ],
    "Oman": [
        "oman (may)",
        "oman (jun)",
        "oman",
    ],
}

SECTION_START_PATTERNS = [
    "key benchmarks",
    "middle east",
]

SECTION_END_PATTERNS = [
    "market commentary",
    "contents",
    "platts espo",
    "forward dated brent",
    "brent/wti spreads and efps",
]

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            date             TEXT NOT NULL,
            symbol           TEXT NOT NULL,
            price            REAL,
            pct_change_24h   REAL,
            PRIMARY KEY (date, symbol)
        )
        """
    )
    ensure_columns(
        conn,
        "prices",
        {
            "date": "TEXT",
            "symbol": "TEXT",
            "price": "REAL",
            "pct_change_24h": "REAL",
        },
    )
    ensure_columns(
        conn,
        "articles",
        {
            "headline": "TEXT",
            "body_text": "TEXT",
            "published_at": "TEXT",
        },
    )
    conn.commit()


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing_tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if table not in existing_tables:
        return

    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, col_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    existing_tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if table not in existing_tables:
        return set()
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2212", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def candidate_lines(text: str) -> list[str]:
    return [line.strip() for line in normalize_text(text).split("\n") if line.strip()]


def extract_pdf_text_from_path(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PDF extraction requires 'pypdf' in the environment running this script."
        ) from e

    with open(pdf_path, "rb") as f:
        reader = PdfReader(BytesIO(f.read()))

    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(text.strip())
    return "\n".join(pages).strip()


def load_report_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT id, headline, body_text, published_at
        FROM articles
        WHERE COALESCE(body_text, '') <> ''
        ORDER BY published_at ASC
        """
    ).fetchall()


def in_relevant_section(line: str) -> bool:
    lowered = line.lower()
    return any(pattern in lowered for pattern in SECTION_START_PATTERNS)


def section_ended(line: str) -> bool:
    lowered = line.lower()
    return any(pattern in lowered for pattern in SECTION_END_PATTERNS)


def extract_relevant_lines(text: str) -> list[str]:
    lines = candidate_lines(text)
    collected: list[str] = []
    active = False

    for line in lines:
        lowered = line.lower()
        if in_relevant_section(line):
            active = True
            continue
        if active and section_ended(line):
            break
        if active:
            # Skip obvious headers
            if lowered in {"bid ask mid change", "mid change", "110 ($/b)"}:
                continue
            collected.append(line)

    if collected:
        return collected
    return lines


def row_to_symbol(label: str, code: str) -> str | None:
    normalized_label = re.sub(r"\s+", " ", label.lower()).strip()
    normalized_code = code.strip().upper()

    if normalized_code.startswith("DBD"):
        return None
    if "spread vs dubai" in normalized_label:
        return None

    if normalized_code in TARGET_SYMBOLS:
        return normalized_code

    for symbol, aliases in TARGET_SYMBOLS.items():
        if normalized_label in aliases:
            return symbol
        if any(alias in normalized_label for alias in aliases):
            return symbol
    return None


def parse_numeric(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except Exception:
        return None


def normalize_symbol_name(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    for symbol, aliases in TARGET_SYMBOLS.items():
        if normalized == symbol.lower():
            return symbol
        if normalized in aliases:
            return symbol
    return None


def report_parser_mode(source_name: str) -> str:
    name = (source_name or "").lower()
    if "marketscan" in name or "arab gulf" in name:
        return "marketscan"
    if "marketwire" in name or "crude oil" in name or "world oil" in name:
        return "marketwire"
    return "generic"


def build_gemini_prompt(source_name: str, text: str) -> str:
    mode = report_parser_mode(source_name)
    base_rules = """Extract benchmark rows from this S&P commodity report text.

Return JSON only with this exact schema:
{
  "benchmarks": [
    {
      "symbol": "Dubai|Brent|WTI|Oman",
      "price": 0.0,
      "change_abs": 0.0,
      "evidence": "short source snippet"
    }
  ]
}

Rules:
- Only use these symbols: Dubai, Brent, WTI, Oman.
- Use the Mid column as price.
- Use the Change column as change_abs when present.
- Prefer benchmark tables over commentary prose.
- Do not invent values.
- If a benchmark is missing, omit it.
- Keep evidence short and copied from the nearby row text.
"""

    if mode == "marketscan":
        extra = """This report is likely an Arab Gulf Marketscan style report.
- Focus on benchmark tables and market table rows, not prose summaries.
- Ignore narrative sections about LPG, gasoline, naphtha, jet kero, gasoil, and fuel oil unless they contain benchmark rows.
"""
    elif mode == "marketwire":
        extra = """This report is likely a Crude Oil Marketwire style report.
- Focus on sections like Key benchmarks, Middle East, Forward Dated Brent, or other crude benchmark tables.
- Ignore commentary paragraphs unless they restate benchmark row values clearly.
"""
    else:
        extra = """Focus on the most structured benchmark table in the document."""

    return f"""{base_rules}
{extra}
Source name: {source_name or "unknown"}

Report text:
{text[:18000]}
"""


def parse_table_row(line: str) -> dict | None:
    compact = re.sub(r"\s+", " ", line).strip()

    number = r"[+-]?\d+(?:\.\d+)?"
    range_token = rf"{number}(?:\s*-\s*{number})"

    # label + code + bid-ask range + mid + change
    match = re.match(
        rf"^(?P<label>.+?) (?P<code>[A-Z0-9]{{6,8}}) "
        rf"(?P<range>{range_token}) "
        rf"(?P<mid>{number}) "
        rf"(?P<change>{number})$",
        compact,
    )
    if not match:
        # label + code + mid + change
        match = re.match(
            rf"^(?P<label>.+?) (?P<code>[A-Z0-9]{{6,8}}) "
            rf"(?P<mid>{number}) "
            rf"(?P<change>{number})$",
            compact,
        )
    if not match:
        # label + code + range + mid
        match = re.match(
            rf"^(?P<label>.+?) (?P<code>[A-Z0-9]{{6,8}}) "
            rf"(?P<range>{range_token}) "
            rf"(?P<mid>{number})$",
            compact,
        )
    if not match:
        # label + code + mid
        match = re.match(
            rf"^(?P<label>.+?) (?P<code>[A-Z0-9]{{6,8}}) "
            rf"(?P<mid>{number})$",
            compact,
        )
    if not match:
        return None

    data = match.groupdict()
    symbol = row_to_symbol(data["label"], data["code"])
    if not symbol:
        return None

    price = parse_numeric(data["mid"])
    if price is None:
        return None

    change = parse_numeric(data.get("change") or "")
    pct_change_24h = None
    if change is not None:
        prior_price = price - change
        if prior_price not in (0, None):
            pct_change_24h = (change / prior_price) * 100

    return {
        "symbol": symbol,
        "label": data["label"],
        "code": data["code"],
        "price": price,
        "pct_change_24h": pct_change_24h,
    }


def gemini_extract_prices_from_text(text: str, source_name: str = "") -> list[dict]:
    if not GEMINI_API_KEY:
        return []

    prompt = build_gemini_prompt(source_name, text)

    url = GEMINI_URL_TEMPLATE.format(model=GEMINI_MODEL)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = requests.post(
            f"{url}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=60,
            verify=not DISABLE_SSL_VERIFY,
        )
        resp.raise_for_status()
        data = resp.json()
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text_out)
    except Exception:
        return []

    rows: list[dict] = []
    seen_symbols: set[str] = set()
    for item in parsed.get("benchmarks", []):
        symbol = normalize_symbol_name(item.get("symbol"))
        price = parse_numeric(str(item.get("price", "")))
        change_abs = parse_numeric(str(item.get("change_abs", "")))
        if not symbol or price is None or symbol in seen_symbols:
            continue

        pct_change_24h = None
        if change_abs is not None:
            prior_price = price - change_abs
            if prior_price not in (0, None):
                pct_change_24h = (change_abs / prior_price) * 100

        rows.append(
            {
                "symbol": symbol,
                "label": symbol,
                "code": "",
                "price": price,
                "pct_change_24h": pct_change_24h,
            }
        )
        seen_symbols.add(symbol)
    return rows


def extract_prices_from_text(text: str, source_name: str = "", force_gemini: bool = False) -> list[dict]:
    if force_gemini:
        return gemini_extract_prices_from_text(text, source_name=source_name)

    rows: list[dict] = []
    seen_symbols: set[str] = set()

    for line in extract_relevant_lines(text):
        parsed = parse_table_row(line)
        if not parsed:
            continue
        if parsed["symbol"] in seen_symbols:
            continue
        seen_symbols.add(parsed["symbol"])
        rows.append(parsed)

    if rows:
        return rows

    return gemini_extract_prices_from_text(text, source_name=source_name)


def build_price_rows(conn: sqlite3.Connection, force_gemini: bool = False) -> list[dict]:
    per_symbol: dict[str, list[dict]] = defaultdict(list)
    inspected_reports = 0

    for row in load_report_rows(conn):
        inspected_reports += 1
        published_at = str(row["published_at"] or "").strip()
        if not published_at:
            continue

        try:
            report_date = pd.to_datetime(published_at).strftime("%Y-%m-%d")
        except Exception:
            continue

        text = str(row["body_text"] or "")
        source_name = str(row["headline"] or "")
        extracted_rows = extract_prices_from_text(
            text,
            source_name=source_name,
            force_gemini=force_gemini,
        )
        if not extracted_rows:
            continue

        for extracted in extracted_rows:
            per_symbol[extracted["symbol"]].append(
                {
                    "date": report_date,
                    "symbol": extracted["symbol"],
                    "price": float(extracted["price"]),
                    "pct_change_24h": extracted["pct_change_24h"],
                    "source_article_id": row["id"],
                }
            )

    all_rows: list[dict] = []
    for symbol, rows in per_symbol.items():
        df = pd.DataFrame(rows).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if "pct_change_24h" not in df.columns:
            df["pct_change_24h"] = None

        computed_change = df["price"].pct_change().shift(-1) * 100
        df["pct_change_24h"] = df["pct_change_24h"].where(df["pct_change_24h"].notna(), computed_change)
        all_rows.extend(
            df[["date", "symbol", "price", "pct_change_24h"]].to_dict("records")
        )
    if not all_rows:
        print(f"[prices] Inspected {inspected_reports} stored reports with non-empty body_text.")
    return all_rows


def build_price_rows_from_pdf(pdf_path: str, report_date: str = "", force_gemini: bool = False) -> list[dict]:
    text = extract_pdf_text_from_path(pdf_path)
    source_name = Path(pdf_path).name
    extracted_rows = extract_prices_from_text(
        text,
        source_name=source_name,
        force_gemini=force_gemini,
    )
    if not extracted_rows:
        return []

    if not report_date:
        report_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    rows = []
    for extracted in extracted_rows:
        rows.append(
            {
                "date": report_date,
                "symbol": extracted["symbol"],
                "price": float(extracted["price"]),
                "pct_change_24h": extracted["pct_change_24h"],
            }
        )
    return rows


def infer_report_date_from_filename(pdf_path: str) -> str:
    name = Path(pdf_path).stem
    patterns = [
        r"(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<d>\d{2})",
        r"(?P<d>\d{2})[-_](?P<m>\d{2})[-_](?P<y>\d{4})",
        r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        parts = match.groupdict()
        try:
            return f"{int(parts['y']):04d}-{int(parts['m']):02d}-{int(parts['d']):02d}"
        except Exception:
            continue
    return ""


def build_price_rows_from_pdf_dir(pdf_dir: str, force_gemini: bool = False) -> list[dict]:
    all_rows: list[dict] = []
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))

    for pdf_path in pdf_paths:
        report_date = infer_report_date_from_filename(str(pdf_path))
        if not report_date:
            print(f"[prices] Skipping {pdf_path.name}: could not infer report date from filename.")
            continue

        rows = build_price_rows_from_pdf(
            str(pdf_path),
            report_date=report_date,
            force_gemini=force_gemini,
        )
        if not rows:
            print(f"[prices] No benchmark rows found in {pdf_path.name}")
            continue

        print(f"[prices] Parsed {len(rows)} benchmark rows from {pdf_path.name} ({report_date})")
        all_rows.extend(rows)

    deduped: dict[tuple[str, str], dict] = {}
    for row in all_rows:
        deduped[(row["date"], row["symbol"])] = row
    return list(deduped.values())


def main(pdf_path: str = "", pdf_dir: str = "", report_date: str = "", force_gemini: bool = False) -> None:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if pdf_dir:
        rows = build_price_rows_from_pdf_dir(pdf_dir, force_gemini=force_gemini)
    elif pdf_path:
        rows = build_price_rows_from_pdf(pdf_path, report_date, force_gemini=force_gemini)
    else:
        rows = build_price_rows(conn, force_gemini=force_gemini)
    if not rows:
        print("[prices] No benchmark prices found in stored report tables.")
        print("[prices] The stored report text may be empty or the table format may differ.")
        if not GEMINI_API_KEY:
            print("[prices] Gemini fallback is disabled. Set GEMINI_API_KEY or GOOGLE_API_KEY to enable LLM repair parsing.")
        conn.close()
        return

    inserted = 0
    price_columns = table_columns(conn, "prices")
    has_ticker = "ticker" in price_columns
    conn.execute("DELETE FROM prices WHERE symbol IS NULL OR TRIM(COALESCE(symbol, '')) = ''")
    for row in rows:
        if has_ticker:
            conn.execute(
                """
                INSERT OR REPLACE INTO prices
                (date, symbol, ticker, price, pct_change_24h)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["date"],
                    row["symbol"],
                    row["symbol"],
                    row["price"],
                    row["pct_change_24h"],
                ),
            )
        else:
            conn.execute(
                """
                INSERT OR REPLACE INTO prices
                (date, symbol, price, pct_change_24h)
                VALUES (?, ?, ?, ?)
                """,
                (row["date"], row["symbol"], row["price"], row["pct_change_24h"]),
            )
        inserted += 1
    conn.commit()

    print(f"[prices] Stored {inserted} price rows extracted from report tables.")
    print("\n[prices] Summary:")
    for row in conn.execute(
        """
        SELECT symbol, COUNT(*), MIN(date), MAX(date)
        FROM prices
        WHERE symbol IS NOT NULL AND TRIM(COALESCE(symbol, '')) <> ''
        GROUP BY symbol
        """
    ).fetchall():
        print(f"  {row[0]}: {row[1]} days ({row[2]} -> {row[3]})")

    conn.close()
    print("\n[prices] Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str, default="", help="Parse prices from a local PDF file.")
    parser.add_argument("--pdf-dir", type=str, default="", help="Parse prices from all PDF files in a local directory.")
    parser.add_argument("--report-date", type=str, default="", help="Override report date (YYYY-MM-DD).")
    parser.add_argument("--force-gemini", action="store_true", help="Use Gemini parsing instead of rule-based parsing.")
    args = parser.parse_args()
    main(
        pdf_path=args.pdf_path.strip(),
        pdf_dir=args.pdf_dir.strip(),
        report_date=args.report_date.strip(),
        force_gemini=args.force_gemini,
    )
