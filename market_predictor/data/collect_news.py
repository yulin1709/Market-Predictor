"""
data/collect_news.py — Pull energy reports from S&P Global News & Insights API.

Pipeline:
  1. GET /news-insights/v1/search/packages  → list of package IDs + titles
  2. GET /news-insights/v1/content/{id}     → envelope with metadata
  3. If PDF bytes available → pdf_handler.extract_text_from_bytes()
  4. cleaner.clean_report_text()            → remove price tables / boilerplate
  5. Store headline + cleaned body_text in articles table

Fallback:
  If no PDF → use JSON body field from /content/{id} response.

Queries target the two most useful daily reports:
  - Crude Oil Marketwire  (COM)
  - Arab Gulf Marketscan  (AGM)

Usage:
    python market_predictor/data/collect_news.py
    python market_predictor/data/collect_news.py --days 30
    python market_predictor/data/collect_news.py --content-id <uuid>
    python market_predictor/data/collect_news.py --refresh-empty-bodies
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import UTC, date, datetime, timedelta
from io import BytesIO

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.auth import api_get_response, get_headers
from data.pdf_handler import extract_text_from_bytes, fetch_and_extract
from data.cleaner import clean_report_text, is_narrative_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(APP_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
SEARCH_URL = "https://api.ci.spglobal.com/news-insights/v1/search/packages"
CONTENT_URL = "https://api.ci.spglobal.com/news-insights/v1/content/{id}"

DEBUG = os.getenv("DEBUG_NEWS_EXTRACTION", "").strip().lower() in {"1", "true", "yes"}
START_DATE = "2022-01-01"

# Report queries — target the daily Platts benchmark reports.
# Extended to pull more article types for richer training data.
# Keep --max-pages 2-3 for daily runs to avoid paginating all history.
QUERIES = [
    "Crude Oil Marketwire",
    "Arab Gulf Marketscan",
    "Gulf Arab Marketscan",
    "refinery outage",
    "tanker shipping",
    "EIA inventory",
    "demand outlook",
    "geopolitical oil",
    "OPEC production",
]

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _norm_id(value: str) -> str:
    return (value or "").strip().lower()


def _is_uuid(value: str) -> bool:
    return bool(UUID_RE.fullmatch(_norm_id(value)))


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_coerce_text(i) for i in value if i).strip()
    if isinstance(value, dict):
        for k in ("text", "value", "content", "body", "summary", "html"):
            t = _coerce_text(value.get(k))
            if t:
                return t
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _in_window(published: str, start: str, end: str) -> bool:
    if not published:
        return True
    # Use UTC date from the timestamp to avoid MYT/UTC timezone mismatches
    pub_date = str(published)[:10]
    return start <= pub_date <= end


# ── DB helpers ────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id           TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            headline     TEXT NOT NULL,
            body_text    TEXT,
            source       TEXT,
            published_at TEXT,
            url          TEXT
        )
    """)
    _ensure_columns(conn, "articles", {
        "title": "TEXT", "headline": "TEXT", "body_text": "TEXT",
        "source": "TEXT", "published_at": "TEXT",
        "fetched_at": "TEXT", "url": "TEXT",
        "report_name": "TEXT", "content_id": "TEXT",
    })
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection, table: str, cols: dict[str, str]) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, col_type in cols.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


def _already_stored(conn: sqlite3.Connection, article_id: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM articles WHERE id = ?", (_norm_id(article_id),)
    ).fetchone())


def _get_existing(conn: sqlite3.Connection, article_id: str):
    return conn.execute(
        "SELECT id, headline, body_text, source, published_at, url FROM articles WHERE id = ?",
        (_norm_id(article_id),),
    ).fetchone()


def _upsert(conn: sqlite3.Connection, article: dict) -> str:
    try:
        conn.execute(
            "INSERT INTO articles (id, title, headline, body_text, source, "
            "published_at, fetched_at, url, report_name, content_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                article["id"], article["headline"], article["headline"],
                article.get("body_text", ""),
                article.get("source", "S&P Global Platts"),
                article.get("published_at", ""),
                article.get("fetched_at") or _now(),
                article.get("url", ""),
                article.get("report_name", ""),
                article.get("content_id", ""),
            ),
        )
        conn.commit()
        return "inserted"
    except sqlite3.IntegrityError:
        return "duplicate"


def _update(conn: sqlite3.Connection, article: dict) -> bool:
    existing = _get_existing(conn, article["id"])
    if not existing:
        return False
    _, old_hl, old_body, old_src, old_pub, old_url = existing
    new_body = article.get("body_text") or old_body or ""
    new_hl = article.get("headline") or old_hl or article["id"]
    if new_body == (old_body or "") and new_hl == (old_hl or ""):
        return False
    conn.execute(
        "UPDATE articles SET headline=?, title=?, body_text=?, source=?, "
        "published_at=?, fetched_at=?, url=?, report_name=?, content_id=? WHERE id=?",
        (
            new_hl, new_hl, new_body,
            article.get("source") or old_src or "S&P Global Platts",
            article.get("published_at") or old_pub or "",
            _now(), article.get("url") or old_url or "",
            article.get("report_name", ""),
            article.get("content_id", ""),
            article["id"],
        ),
    )
    conn.commit()
    return True


# ── Body extraction ───────────────────────────────────────────────────────────

def _extract_body_from_content(article_id: str) -> tuple[str, str, str, str]:
    """
    Fetch /content/{id} and extract body text.

    Returns (headline, body_text, published_at, url).
    Body text is cleaned via cleaner.clean_report_text().

    Strategy:
      1. If response is PDF bytes → pdf_handler + cleaner
      2. If JSON with body field → use body field + cleaner
      3. If JSON with documentUrl → fetch PDF from that URL + cleaner
      4. Fallback → empty string
    """
    url = CONTENT_URL.format(id=article_id)
    try:
        resp = api_get_response(url, timeout=30, retries=1, accept="application/json")
    except Exception as exc:
        print(f"  [content] ERROR fetching {article_id}: {exc}")
        return "", "", "", ""

    ct = resp.headers.get("Content-Type", "").lower()

    # ── Case 1: Direct PDF response ───────────────────────────────────────────
    if "application/pdf" in ct or resp.content[:4] == b"%PDF":
        raw_text = extract_text_from_bytes(resp.content)
        body = clean_report_text(raw_text)
        if DEBUG:
            print(f"  [content] PDF direct: {len(body)} chars after cleaning")
        return article_id, body, "", url

    # ── Case 2: JSON response ─────────────────────────────────────────────────
    if "application/json" in ct:
        try:
            data = resp.json()
        except Exception:
            return "", "", "", ""

        # Unwrap envelope
        envelope = data.get("envelope", data)
        if isinstance(envelope.get("data"), dict):
            inner = envelope["data"]
        else:
            inner = envelope

        headline = _coerce_text(
            inner.get("headline") or inner.get("title")
            or envelope.get("package", {}).get("title")
            or ""
        )
        published = (
            inner.get("updatedDate") or inner.get("publishedDate")
            or envelope.get("properties", {}).get("coverDate")
            or envelope.get("properties", {}).get("updatedDate")
            or ""
        )
        doc_url = (
            inner.get("documentUrl") or inner.get("downloadUrl")
            or data.get("documentUrl") or ""
        )

        # Try JSON body field first
        body_raw = _coerce_text(
            inner.get("body") or inner.get("bodyText")
            or inner.get("content") or inner.get("summary") or ""
        )

        if body_raw and len(body_raw) > 100:
            body = clean_report_text(body_raw)
            if DEBUG:
                print(f"  [content] JSON body: {len(body)} chars after cleaning")
            return headline, body, str(published)[:19], doc_url

        # Try PDF from documentUrl
        if doc_url and "plattsconnect" not in doc_url:
            raw_text = fetch_and_extract(doc_url, headers=get_headers())
            if raw_text:
                body = clean_report_text(raw_text)
                if DEBUG:
                    print(f"  [content] PDF from documentUrl: {len(body)} chars after cleaning")
                return headline, body, str(published)[:19], doc_url

        if DEBUG:
            print(f"  [content] No body found for {article_id}")
        return headline, "", str(published)[:19], doc_url

    # ── Case 3: Plain text ────────────────────────────────────────────────────
    if "text" in ct:
        body = clean_report_text(resp.text[:8000])
        return article_id, body, "", url

    return "", "", "", ""


# ── Per-article fetch ─────────────────────────────────────────────────────────

def fetch_content_item(conn: sqlite3.Connection, article_id: str) -> int:
    """Fetch and store a single article by content ID. Returns 1 if stored."""
    article_id = _norm_id(article_id)
    existing = _get_existing(conn, article_id)

    headline, body, published, url = _extract_body_from_content(article_id)

    if not headline:
        headline = (existing[1] if existing else "") or article_id
    if not published and existing:
        published = existing[4] or ""
    if not url and existing:
        url = existing[5] or ""

    article = {
        "id": article_id,
        "headline": headline,
        "body_text": body,
        "source": "S&P Global Platts",
        "published_at": published,
        "fetched_at": _now(),
        "url": url,
    }

    body_len = len(body)
    if not existing:
        _upsert(conn, article)
        print(f"  [content] Stored: {headline[:80]} (body={body_len} chars)")
        return 1

    if _update(conn, article):
        print(f"  [content] Updated: {headline[:80]} (body={body_len} chars)")
        return 1

    return 0


# ── Batch refresh ─────────────────────────────────────────────────────────────

def refresh_empty_bodies(conn: sqlite3.Connection, limit: int | None = None,
                         workers: int = 5,
                         report_filter: list[str] | None = None) -> int:
    """
    Re-fetch body text for articles that have empty body_text.

    Args:
        conn:          SQLite connection
        limit:         max articles to process (None = all)
        workers:       number of parallel fetch threads (default 5)
        report_filter: only process articles whose headline contains one of these
                       strings (e.g. ['Crude Oil Marketwire', 'Arab Gulf Marketscan'])
                       None = process all empty articles
    """
    sql = """
        SELECT id, headline FROM articles
        WHERE TRIM(COALESCE(body_text, '')) = ''
        ORDER BY published_at DESC, id DESC
    """
    params: tuple = ()
    if limit:
        sql += " LIMIT ?"
        params = (limit,)

    rows = conn.execute(sql, params).fetchall()

    # Filter to UUID IDs only
    valid = [(r[0], r[1]) for r in rows if _is_uuid(r[0])]

    # Optional: only fetch articles from high-value report types
    if report_filter:
        def _matches(headline: str) -> bool:
            hl = (headline or "").lower()
            return any(f.lower() in hl for f in report_filter)
        valid = [(aid, hl) for aid, hl in valid if _matches(hl)]

    skipped = len(rows) - len(valid)
    if skipped:
        print(f"  [refresh] Skipped {skipped} (non-UUID or filtered out)")
    if not valid:
        print("  [refresh] No empty bodies to fill")
        return 0

    print(f"  [refresh] Filling {len(valid)} empty bodies (workers={workers})...")

    # Parallel fetch — each thread fetches body text independently
    # DB writes are serialised back on the main thread to avoid SQLite contention
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(article_id: str) -> tuple[str, str, str, str, str]:
        """Returns (article_id, headline, body, published, url)."""
        headline, body, published, url = _extract_body_from_content(article_id)
        return article_id, headline, body, published, url

    refreshed = 0
    total = len(valid)
    ids = [aid for aid, _ in valid]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, aid): aid for aid in ids}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                article_id, headline, body, published, url = future.result()
            except Exception as exc:
                print(f"  [refresh] ERROR: {exc}")
                continue

            if not body:
                continue

            # Serialised DB write
            existing = _get_existing(conn, article_id)
            old_hl = existing[1] if existing else ""
            article = {
                "id": article_id,
                "headline": headline or old_hl or article_id,
                "body_text": body,
                "source": "S&P Global Platts",
                "published_at": published or (existing[4] if existing else ""),
                "url": url or (existing[5] if existing else ""),
            }
            if existing:
                if _update(conn, article):
                    refreshed += 1
                    print(f"  [refresh] {done}/{total} OK {article['headline'][:60]} ({len(body)} chars)")
            else:
                _upsert(conn, article)
                refreshed += 1
                print(f"  [refresh] {done}/{total} + {article['headline'][:60]} ({len(body)} chars)")

    print(f"  [refresh] Done. Refreshed {refreshed} / {total}")
    return refreshed


# ── Search + ingest ───────────────────────────────────────────────────────────

def fetch_query(conn: sqlite3.Connection, query: str,
                start_date: str, end_date: str,
                fetch_full_body: bool = True,
                max_pages: int | None = None) -> int:
    """Search packages and ingest articles. Returns count of new articles."""
    new_count = 0
    page = 1
    page_size = 100
    consecutive_old = 0  # stop early if we see many results older than start_date

    while True:
        print(f"  '{query}' page {page}...")
        try:
            resp = api_get_response(
                SEARCH_URL,
                params={"q": query, "pageSize": page_size, "page": page},
                timeout=30, retries=1,
            )
            data = resp.json()
        except Exception as exc:
            print(f"  ERROR: {exc}")
            break

        # Extract results list
        results = data.get("results", [])
        if isinstance(results, dict):
            for key in ("Story", "story", "stories", "items"):
                if isinstance(results.get(key), list):
                    results = results[key]
                    break
        if not results:
            break

        for item in results:
            art_id = _norm_id(item.get("id") or "")
            if not _is_uuid(art_id):
                continue

            published = item.get("updatedDate") or item.get("publishedDate") or ""
            pub_date = str(published)[:10]

            if not _in_window(published, start_date, end_date):
                if pub_date and pub_date < start_date:
                    consecutive_old += 1
                    # Stop paginating after 5 consecutive out-of-window results
                    if consecutive_old >= 5:
                        return new_count
                continue
            consecutive_old = 0  # reset on in-window result

            # Build headline from title or query + date
            headline = _coerce_text(
                item.get("title") or item.get("headline") or item.get("noteName") or ""
            )
            if not headline:
                mime = _coerce_text(item.get("mimeType")) or "package"
                day = str(published)[:10]
                headline = f"{query} [{mime}] {day}".strip()

            doc_url = item.get("documentUrl") or item.get("url") or ""

            # Fetch full body if needed
            body = ""
            exists = _already_stored(conn, art_id)
            if fetch_full_body:
                existing = _get_existing(conn, art_id) if exists else None
                existing_body = existing[2] if existing else ""
                if not exists or not (existing_body or "").strip():
                    _, body, fetched_pub, fetched_url = _extract_body_from_content(art_id)
                    if fetched_pub and not published:
                        published = fetched_pub
                    if fetched_url and not doc_url:
                        doc_url = fetched_url
                    time.sleep(0.15)

            article = {
                "id": art_id,
                "headline": headline,
                "body_text": body,
                "source": "S&P Global Platts",
                "published_at": str(published)[:19],
                "fetched_at": _now(),
                "url": doc_url,
                "report_name": query,
                "content_id": art_id,
            }

            status = _upsert(conn, article)
            if status == "inserted":
                new_count += 1
                body_len = len(body)
                print(f"  + {headline[:70]} (body={body_len} chars)")
            elif status == "duplicate":
                _update(conn, article)

        # Pagination
        meta = data.get("metadata", {})
        total_pages = meta.get("total_pages") or meta.get("totalPages") or meta.get("pageCount") or 0
        if max_pages and page >= max_pages:
            break
        if (total_pages and page >= total_pages) or len(results) < page_size:
            break
        page += 1
        time.sleep(0.3)

    return new_count


# ── Main entry point ──────────────────────────────────────────────────────────

def run_main(
    days: int | None = None,
    content_id: str = "",
    search_only: bool = False,
    max_pages: int | None = None,
    refresh_empty_bodies_only: bool = False,
    limit: int | None = None,
    workers: int = 5,
    report_filter: list[str] | None = None,
    extra_queries: list[str] | None = None,
) -> None:
    # end_date is tomorrow to catch timezone edge cases (UTC vs MYT)
    end_date = (date.today() + timedelta(days=1)).isoformat()
    start_date = (
        (date.today() - timedelta(days=days)).isoformat() if days else START_DATE
    )

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    init_db(conn)

    if refresh_empty_bodies_only:
        refresh_empty_bodies(conn, limit=limit, workers=workers,
                             report_filter=report_filter)
        conn.close()
        return

    if content_id:
        n = fetch_content_item(conn, _norm_id(content_id))
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        print(f"\n[news] Done. {n} article stored. Total in DB: {total}")
        conn.close()
        return

    queries_to_run = list(QUERIES)
    if extra_queries:
        for q in extra_queries:
            if q not in queries_to_run:
                queries_to_run.append(q)

    total_new = 0
    for query in queries_to_run:
        print(f"\n[news] Query: '{query}'  ({start_date} -> {end_date})")
        n = fetch_query(
            conn, query, start_date, end_date,
            fetch_full_body=not search_only,
            max_pages=max_pages,
        )
        print(f"  -> {n} new articles")
        total_new += n
        time.sleep(0.5)

    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"\n[news] Done. {total_new} new articles added. Total in DB: {total}")
    conn.close()


# Keep backward-compatible alias
main = run_main

current_timestamp = _now  # backward compat for backfill_reports.py


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest S&P Global Platts reports.")
    parser.add_argument("--days", type=int, default=None,
                        help="Fetch articles from last N days (default: all history)")
    parser.add_argument("--content-id", type=str, default="",
                        help="Fetch a single article by content UUID")
    parser.add_argument("--search-only", action="store_true",
                        help="Only search, do not fetch full body text")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit number of search result pages per query")
    parser.add_argument("--refresh-empty-bodies", action="store_true",
                        help="Re-fetch body text for articles with empty body_text")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of articles to refresh")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel fetch threads for --refresh-empty-bodies (default 5)")
    parser.add_argument("--reports-only", action="store_true",
                        help="Only refresh Crude Oil Marketwire and Arab Gulf Marketscan articles")
    parser.add_argument("--queries", type=str, nargs="+", default=None,
                        help="Additional search queries to run (appended to default QUERIES list)")
    args = parser.parse_args()

    report_filter = None
    if args.reports_only:
        report_filter = ["Crude Oil Marketwire", "Arab Gulf Marketscan",
                         "Gulf Arab Marketscan", "Oilgram"]

    run_main(
        days=args.days,
        content_id=args.content_id.strip(),
        search_only=args.search_only,
        max_pages=args.max_pages,
        refresh_empty_bodies_only=args.refresh_empty_bodies,
        limit=args.limit,
        workers=args.workers,
        report_filter=report_filter,
        extra_queries=args.queries,
    )
