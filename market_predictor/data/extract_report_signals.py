"""
data/extract_report_signals.py - Extract benchmark rows and market signals from
S&P PDF reports for forecasting workflows.

This is a standalone PDF-driven pipeline inspired by the newsletter summary
workflow, but tailored for structured extraction instead of prose summaries.

Features:
- Parse one PDF or a directory of PDFs
- Gemini-first extraction with JSON-only output
- Rule-based benchmark fallback if Gemini fails
- Save per-report JSON artifacts under output/report_extractions/
- Optionally write benchmark prices into SQLite prices table
- Optionally write report-level extracted signals into SQLite

Usage:
    python data/extract_report_signals.py --pdf-path "C:\\path\\report.pdf"
    python data/extract_report_signals.py --pdf-dir "C:\\path\\reports"
    python data/extract_report_signals.py --pdf-dir "C:\\path\\reports" --skip-db
"""
import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(APP_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "report_extractions")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY", "").strip()
    or os.getenv("GEMINI_API_KEY", "").strip()
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "").strip().lower() in {"1", "true", "yes"}

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

ALLOWED_SYMBOLS = {"Dubai", "Brent", "WTI", "Oman"}

TARGET_SYMBOLS = {
    "Dubai": ["dubai (may)", "dubai (jun)", "dubai (jul)", "dubai"],
    "Brent": ["brent (dated)", "dated brent", "brent"],
    "WTI": ["wti fob usgc", "wti"],
    "Oman": ["oman (may)", "oman (jun)", "oman"],
}


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS report_extractions (
            id              TEXT PRIMARY KEY,
            file_name       TEXT NOT NULL,
            report_date     TEXT,
            report_type     TEXT,
            benchmark_json  TEXT,
            signal_json     TEXT,
            raw_json        TEXT,
            created_at      TEXT NOT NULL
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
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def extract_pdf_text_from_path(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("PDF extraction requires 'pypdf'.") from e

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


def infer_report_type(name: str) -> str:
    lowered = (name or "").lower()
    if "marketscan" in lowered or "arab gulf" in lowered:
        return "marketscan"
    if "marketwire" in lowered or "crude oil" in lowered or "world oil" in lowered:
        return "marketwire"
    return "generic"


def infer_report_date(name: str, override: str = "") -> str:
    if override:
        return override
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


def normalize_symbol_name(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    for symbol, aliases in TARGET_SYMBOLS.items():
        if normalized == symbol.lower():
            return symbol
        if normalized in aliases:
            return symbol
    return None


def parse_numeric(value) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def build_gemini_prompt(source_name: str, report_type: str, text: str) -> str:
    if report_type == "marketscan":
        extra = """This is an Arab Gulf Marketscan-style report.
- Focus on benchmark tables and market developments relevant to crude and refined products.
- Extract only benchmark rows for Dubai, Brent, WTI, and Oman if present.
- Extract concise forecast-relevant signals from the report."""
    elif report_type == "marketwire":
        extra = """This is a Crude Oil Marketwire-style report.
- Focus on benchmark tables such as Key benchmarks, Middle East, and crude market sections.
- Extract only benchmark rows for Dubai, Brent, WTI, and Oman if present.
- Extract concise forecast-relevant signals from Sweet Crude, Sour Crude, and Global topics."""
    else:
        extra = """Focus on the most structured benchmark table and forecast-relevant narrative."""

    return f"""Extract structured forecasting data from this S&P commodity report.

Return JSON only with this exact schema:
{{
  "report_date": "YYYY-MM-DD or empty string",
  "report_type": "{report_type}",
  "benchmarks": [
    {{
      "symbol": "Dubai|Brent|WTI|Oman",
      "price": 0.0,
      "change_abs": 0.0,
      "evidence": "short source snippet"
    }}
  ],
  "signals": [
    {{
      "category": "Sweet Crude|Sour Crude|Global|LPG|Gasoline|Naphtha|Jet Kero|Gasoil|Fuel Oil|Other",
      "signal": "short factual signal",
      "impact": "bullish|bearish|neutral"
    }}
  ]
}}

Rules:
- Only use these symbols for benchmarks: Dubai, Brent, WTI, Oman.
- Use Mid as price when table data exists.
- Use Change as change_abs when table data exists.
- Do not invent values.
- Omit missing benchmarks.
- Signals must be factual, concise, and useful for forecasting.
- Prefer table evidence and explicit market commentary over generic summaries.

{extra}

Source name: {source_name}
Report text:
{text[:22000]}
"""


def gemini_extract(source_name: str, report_type: str, text: str) -> dict:
    if not GOOGLE_API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")

    url = GEMINI_URL_TEMPLATE.format(model=GEMINI_MODEL)
    payload = {
        "contents": [{"parts": [{"text": build_gemini_prompt(source_name, report_type, text)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        f"{url}?key={GOOGLE_API_KEY}",
        json=payload,
        timeout=90,
        verify=not DISABLE_SSL_VERIFY,
    )
    resp.raise_for_status()
    data = resp.json()
    text_out = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text_out)


def rule_based_benchmarks(text: str) -> list[dict]:
    rows = []
    seen_symbols = set()
    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(
            r"^(?P<label>.+?) (?P<code>[A-Z0-9]{6,8}) "
            r"(?P<bid>-?\d+(?:\.\d+)?) "
            r"(?P<ask>-?\d+(?:\.\d+)?) "
            r"(?P<mid>-?\d+(?:\.\d+)?) "
            r"(?P<change>-?\d+(?:\.\d+)?)$",
            re.sub(r"\s+", " ", line),
        )
        if not match:
            continue
        data = match.groupdict()
        symbol = normalize_symbol_name(data["label"])
        if not symbol or symbol in seen_symbols:
            continue
        price = parse_numeric(data["mid"])
        change_abs = parse_numeric(data["change"])
        if price is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "price": price,
                "change_abs": change_abs,
                "evidence": line[:160],
            }
        )
        seen_symbols.add(symbol)
    return rows


def validate_payload(payload: dict, source_name: str, report_type: str, inferred_report_date: str) -> dict:
    benchmarks = []
    seen_symbols = set()
    for item in payload.get("benchmarks", []):
        symbol = normalize_symbol_name(item.get("symbol"))
        price = parse_numeric(item.get("price"))
        change_abs = parse_numeric(item.get("change_abs"))
        evidence = str(item.get("evidence", "")).strip()
        if not symbol or symbol in seen_symbols or price is None:
            continue
        pct_change_24h = None
        if change_abs is not None:
            prior_price = price - change_abs
            if prior_price not in (0, None):
                pct_change_24h = (change_abs / prior_price) * 100
        benchmarks.append(
            {
                "symbol": symbol,
                "price": price,
                "change_abs": change_abs,
                "pct_change_24h": pct_change_24h,
                "evidence": evidence,
            }
        )
        seen_symbols.add(symbol)

    signals = []
    for item in payload.get("signals", []):
        category = str(item.get("category", "Other")).strip() or "Other"
        signal = str(item.get("signal", "")).strip()
        impact = str(item.get("impact", "neutral")).strip().lower()
        if not signal:
            continue
        if impact not in {"bullish", "bearish", "neutral"}:
            impact = "neutral"
        signals.append(
            {
                "category": category,
                "signal": signal,
                "impact": impact,
            }
        )

    report_date = str(payload.get("report_date", "")).strip() or inferred_report_date
    return {
        "report_date": report_date,
        "report_type": str(payload.get("report_type", report_type)).strip() or report_type,
        "benchmarks": benchmarks,
        "signals": signals,
        "source_name": source_name,
    }


def extract_report(
    pdf_path: str,
    report_date_override: str = "",
    report_type_override: str = "",
    force_rule_only: bool = False,
) -> dict:
    source_name = Path(pdf_path).name
    report_type = report_type_override or infer_report_type(source_name)
    inferred_report_date = infer_report_date(source_name, override=report_date_override)
    text = extract_pdf_text_from_path(pdf_path)
    if not text:
        raise RuntimeError(f"No text extracted from PDF: {pdf_path}")

    if force_rule_only:
        payload = {
            "report_date": inferred_report_date,
            "report_type": report_type,
            "benchmarks": rule_based_benchmarks(text),
            "signals": [],
        }
    else:
        try:
            payload = gemini_extract(source_name, report_type, text)
        except Exception:
            payload = {
                "report_date": inferred_report_date,
                "report_type": report_type,
                "benchmarks": rule_based_benchmarks(text),
                "signals": [],
            }

    validated = validate_payload(payload, source_name, report_type, inferred_report_date)
    if not validated["benchmarks"]:
        fallback = validate_payload(
            {
                "report_date": inferred_report_date,
                "report_type": report_type,
                "benchmarks": rule_based_benchmarks(text),
                "signals": validated["signals"],
            },
            source_name,
            report_type,
            inferred_report_date,
        )
        if fallback["benchmarks"]:
            validated["benchmarks"] = fallback["benchmarks"]

    validated["metadata"] = {
        "file_name": source_name,
        "report_type": report_type,
        "report_date": validated["report_date"],
        "text_length": len(text),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "model": GEMINI_MODEL if not force_rule_only and GOOGLE_API_KEY else "rule-based",
    }
    return validated


def write_json_output(result: dict, pdf_path: str) -> str:
    safe_name = Path(pdf_path).stem.replace(" ", "_")
    folder = os.path.join(OUTPUT_DIR, safe_name)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(folder, f"{safe_name}_forecast_extract_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    return out_path


def upsert_prices(conn: sqlite3.Connection, result: dict) -> int:
    columns = table_columns(conn, "prices")
    has_ticker = "ticker" in columns
    report_date = result["report_date"]
    if not report_date:
        return 0

    inserted = 0
    for item in result["benchmarks"]:
        if has_ticker:
            conn.execute(
                """
                INSERT OR REPLACE INTO prices
                (date, symbol, ticker, price, pct_change_24h)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report_date,
                    item["symbol"],
                    item["symbol"],
                    item["price"],
                    item["pct_change_24h"],
                ),
            )
        else:
            conn.execute(
                """
                INSERT OR REPLACE INTO prices
                (date, symbol, price, pct_change_24h)
                VALUES (?, ?, ?, ?)
                """,
                (
                    report_date,
                    item["symbol"],
                    item["price"],
                    item["pct_change_24h"],
                ),
            )
        inserted += 1
    return inserted


def upsert_report_extraction(conn: sqlite3.Connection, result: dict) -> None:
    file_name = result["metadata"]["file_name"]
    report_id = f"{file_name}:{result['report_date'] or 'unknown'}"
    conn.execute(
        """
        INSERT OR REPLACE INTO report_extractions
        (id, file_name, report_date, report_type, benchmark_json, signal_json, raw_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report_id,
            file_name,
            result["report_date"],
            result["report_type"],
            json.dumps(result["benchmarks"], ensure_ascii=True),
            json.dumps(result["signals"], ensure_ascii=True),
            json.dumps(result, ensure_ascii=True),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def process_one(
    pdf_path: str,
    conn: sqlite3.Connection | None,
    report_date_override: str,
    report_type_override: str,
    skip_db: bool,
    force_rule_only: bool,
) -> None:
    result = extract_report(
        pdf_path,
        report_date_override=report_date_override,
        report_type_override=report_type_override,
        force_rule_only=force_rule_only,
    )
    out_path = write_json_output(result, pdf_path)
    print(f"[extract] {Path(pdf_path).name}")
    print(f"  report_type={result['report_type']}")
    print(f"  report_date={result['report_date'] or '<unknown>'}")
    print(f"  benchmarks={len(result['benchmarks'])} signals={len(result['signals'])}")
    print(f"  json={out_path}")

    if conn is not None and not skip_db:
        inserted = upsert_prices(conn, result)
        upsert_report_extraction(conn, result)
        conn.commit()
        print(f"  db_prices={inserted} db_report=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str, default="", help="Process one PDF file.")
    parser.add_argument("--pdf-dir", type=str, default="", help="Process all PDFs in a directory.")
    parser.add_argument("--report-date", type=str, default="", help="Override report date for single PDF runs.")
    parser.add_argument(
        "--report-type",
        type=str,
        default="",
        choices=["", "marketwire", "marketscan", "generic"],
        help="Override inferred report type.",
    )
    parser.add_argument("--skip-db", action="store_true", help="Do not write extracted data into SQLite.")
    parser.add_argument("--force-rule-only", action="store_true", help="Skip Gemini and use rule-based extraction only.")
    args = parser.parse_args()

    if not args.pdf_path and not args.pdf_dir:
        raise SystemExit("Use --pdf-path or --pdf-dir")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = None if args.skip_db else sqlite3.connect(DB_PATH)
    if conn is not None:
        init_db(conn)

    try:
        if args.pdf_path:
            process_one(
                pdf_path=args.pdf_path,
                conn=conn,
                report_date_override=args.report_date,
                report_type_override=args.report_type,
                skip_db=args.skip_db,
                force_rule_only=args.force_rule_only,
            )
        else:
            pdf_paths = sorted(Path(args.pdf_dir).glob("*.pdf"))
            if not pdf_paths:
                raise SystemExit(f"No PDF files found in {args.pdf_dir}")
            for pdf_path in pdf_paths:
                process_one(
                    pdf_path=str(pdf_path),
                    conn=conn,
                    report_date_override="",
                    report_type_override=args.report_type,
                    skip_db=args.skip_db,
                    force_rule_only=args.force_rule_only,
                )
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
