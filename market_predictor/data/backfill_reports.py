"""
Batch backfill runner for S&P report ingestion.

This script orchestrates:
1. report/package discovery
2. empty body refresh in batches
3. benchmark price extraction
4. article labeling

Usage:
    python data/backfill_reports.py --days 180 --max-pages 5
    python data/backfill_reports.py --days 365 --max-pages 8 --refresh-batch 100 --refresh-rounds 10
    python data/backfill_reports.py --days 180 --max-pages 5 --skip-labels
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from collect_news import DB_PATH, current_timestamp, init_db, main as collect_news_main, refresh_empty_bodies
from collect_prices import main as collect_prices_main
from align_and_label import main as align_and_label_main


def count_empty_bodies(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE TRIM(COALESCE(body_text, '')) = ''"
    ).fetchone()
    return int(row[0] if row else 0)


def count_non_empty_bodies(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE TRIM(COALESCE(body_text, '')) <> ''"
    ).fetchone()
    return int(row[0] if row else 0)


def run_refresh_rounds(refresh_batch: int, refresh_rounds: int,
                       workers: int = 5,
                       report_filter: list[str] | None = None) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    init_db(conn)

    for round_idx in range(1, refresh_rounds + 1):
        before_empty = count_empty_bodies(conn)
        before_filled = count_non_empty_bodies(conn)
        if before_empty == 0:
            print(f"[backfill] No empty bodies remain before round {round_idx}.")
            break

        print(
            f"[backfill] Refresh round {round_idx}/{refresh_rounds} "
            f"(empty={before_empty}, filled={before_filled}, batch={refresh_batch}, workers={workers})"
        )
        refreshed = refresh_empty_bodies(conn, limit=refresh_batch,
                                         workers=workers,
                                         report_filter=report_filter)
        after_empty = count_empty_bodies(conn)
        after_filled = count_non_empty_bodies(conn)
        print(
            f"[backfill] Round {round_idx} complete: refreshed={refreshed}, "
            f"empty_now={after_empty}, filled_now={after_filled}"
        )
        if refreshed == 0 or after_empty >= before_empty:
            print("[backfill] Refresh progress stalled; stopping further rounds.")
            break

    conn.close()


def main(
    days: int,
    max_pages: int | None,
    refresh_batch: int,
    refresh_rounds: int,
    skip_prices: bool,
    skip_labels: bool,
    workers: int = 5,
    reports_only: bool = False,
) -> None:
    print(f"[backfill] Started at {current_timestamp()}")
    print(
        f"[backfill] Discovery window: days={days}, max_pages={max_pages or 'all'}, "
        f"refresh_batch={refresh_batch}, refresh_rounds={refresh_rounds}, workers={workers}"
    )

    report_filter = None
    if reports_only:
        report_filter = ["Crude Oil Marketwire", "Arab Gulf Marketscan",
                         "Gulf Arab Marketscan", "Oilgram"]
        print(f"[backfill] Report filter: {report_filter}")

    collect_news_main(days=days, max_pages=max_pages, search_only=False)
    run_refresh_rounds(refresh_batch=refresh_batch, refresh_rounds=refresh_rounds,
                       workers=workers, report_filter=report_filter)

    if not skip_prices:
        collect_prices_main()
    if not skip_labels:
        align_and_label_main()

    print(f"[backfill] Finished at {current_timestamp()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--refresh-batch", type=int, default=100)
    parser.add_argument("--refresh-rounds", type=int, default=5)
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel fetch threads (default 5)")
    parser.add_argument("--reports-only", action="store_true",
                        help="Only refresh Marketwire/Marketscan articles")
    args = parser.parse_args()
    main(
        days=args.days,
        max_pages=args.max_pages,
        refresh_batch=args.refresh_batch,
        refresh_rounds=args.refresh_rounds,
        skip_prices=args.skip_prices,
        skip_labels=args.skip_labels,
        workers=args.workers,
        reports_only=args.reports_only,
    )
