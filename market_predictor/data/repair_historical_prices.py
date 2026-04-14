"""
data/repair_historical_prices.py — Repair historical price data in the predictions table.

Fixes two known data quality issues:
  Fix 1: Rows where entry_price == exit_price (same-day fetch bug)
  Fix 2: Brent rows that share the same price as Dubai Crude on the same date
         (caused by the old BZ=F ticker being used for both)

Usage:
    # Preview what would be changed (safe, no writes)
    python market_predictor/data/repair_historical_prices.py --dry-run

    # Apply repairs
    python market_predictor/data/repair_historical_prices.py

The script is idempotent — rows already marked data_quality='repaired_ok' are skipped.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
SAME_PRICE_TOLERANCE = 0.0001


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for col in ("entry_price_source", "exit_price_source", "data_quality"):
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass


def _find_same_day_price_rows(conn: sqlite3.Connection) -> list[tuple]:
    """Rows where entry_price == exit_price (within tolerance) — same-day fetch bug."""
    return conn.execute("""
        SELECT id, commodity, prediction_date, entry_price, exit_price
        FROM predictions
        WHERE entry_price IS NOT NULL
          AND exit_price IS NOT NULL
          AND ABS(exit_price - entry_price) < ?
          AND COALESCE(data_quality, '') != 'repaired_ok'
    """, (SAME_PRICE_TOLERANCE,)).fetchall()


def _find_shared_price_rows(conn: sqlite3.Connection) -> list[tuple]:
    """
    Brent rows that share the exact same price as a Dubai Crude row on the same date.
    This was caused by both using BZ=F (Brent futures) as the yfinance ticker.
    """
    return conn.execute("""
        SELECT b.id, b.commodity, b.prediction_date, b.entry_price, b.exit_price
        FROM predictions b
        JOIN predictions d
          ON b.prediction_date = d.prediction_date
         AND ABS(COALESCE(b.entry_price, -1) - COALESCE(d.entry_price, -1)) < ?
        WHERE b.commodity LIKE '%Brent%'
          AND d.commodity LIKE '%Dubai%'
          AND b.entry_price IS NOT NULL
          AND COALESCE(b.data_quality, '') != 'repaired_ok'
    """, (SAME_PRICE_TOLERANCE,)).fetchall()


def _clear_prices(conn: sqlite3.Connection, row_id: int, reason: str, dry_run: bool) -> None:
    """Clear entry/exit prices and mark for re-fetch."""
    if dry_run:
        print(f"  [DRY-RUN] Would clear prices for id={row_id} (reason: {reason})")
        return
    conn.execute("""
        UPDATE predictions
        SET entry_price = NULL,
            exit_price  = NULL,
            actual_move = NULL,
            outcome     = NULL,
            pnl_usd     = NULL,
            data_quality = ?
        WHERE id = ?
    """, (f"cleared_{reason}", row_id))


def _refetch_row(conn: sqlite3.Connection, row_id: int, commodity: str,
                 pred_date: str, dry_run: bool) -> None:
    """Re-fetch entry and exit prices for a single row using the two-tier fetcher."""
    from data.fetch_price import get_closing_price, get_next_trading_close

    entry_result = get_closing_price(commodity, pred_date)
    exit_result = get_next_trading_close(commodity, pred_date)

    if not entry_result or not exit_result:
        print(f"  [repair] id={row_id}: price fetch failed for {commodity} on {pred_date}")
        if not dry_run:
            conn.execute(
                "UPDATE predictions SET data_quality = 'repair_fetch_failed' WHERE id = ?",
                (row_id,),
            )
        return

    entry_price = entry_result["price"]
    exit_price = exit_result["price"]

    # Guard: don't write if still same-day
    if abs(exit_price - entry_price) < SAME_PRICE_TOLERANCE:
        print(f"  [repair] id={row_id}: re-fetched prices still identical ({entry_price}) — skipping")
        if not dry_run:
            conn.execute(
                "UPDATE predictions SET data_quality = 'repair_still_same_price' WHERE id = ?",
                (row_id,),
            )
        return

    from data.backfill_actuals import classify_outcome
    from data.ticker_config import DEFAULT_LOT_SIZE

    # Need signal and direction to recompute outcome
    row = conn.execute(
        "SELECT signal, direction FROM predictions WHERE id = ?", (row_id,)
    ).fetchone()
    if not row:
        return
    signal, direction = row

    actual_move = round((exit_price - entry_price) / entry_price * 100, 2)
    outcome = classify_outcome(signal, direction, actual_move)
    direction_mult = 1 if direction == "rise" else (-1 if direction == "fall" else 0)
    pnl_usd = round((exit_price - entry_price) * DEFAULT_LOT_SIZE * direction_mult, 2)

    if dry_run:
        print(
            f"  [DRY-RUN] id={row_id} {commodity} {pred_date}: "
            f"${entry_price} ({entry_result['source']}) -> ${exit_price} ({exit_result['source']}) "
            f"= {actual_move:+.2f}% ({outcome}) P&L: ${pnl_usd:+,.0f}"
        )
        return

    conn.execute("""
        UPDATE predictions
        SET entry_price        = ?,
            entry_price_source = ?,
            exit_price         = ?,
            exit_price_source  = ?,
            actual_move        = ?,
            outcome            = ?,
            pnl_usd            = ?,
            data_quality       = 'repaired_ok'
        WHERE id = ?
    """, (
        entry_price, entry_result["source"],
        exit_price, exit_result["source"],
        actual_move, outcome, pnl_usd,
        row_id,
    ))
    print(
        f"  [repair] id={row_id} {commodity} {pred_date}: "
        f"${entry_price} ({entry_result['source']}) -> ${exit_price} ({exit_result['source']}) "
        f"= {actual_move:+.2f}% ({outcome}) P&L: ${pnl_usd:+,.0f}"
    )


def repair(dry_run: bool = False) -> None:
    conn = sqlite3.connect(DB_PATH)
    _ensure_columns(conn)

    # Fix 1: same-day price rows
    same_day_rows = _find_same_day_price_rows(conn)
    print(f"\n[repair] Fix 1 — Same-day price rows: {len(same_day_rows)} found")
    for row_id, commodity, pred_date, entry_price, exit_price in same_day_rows:
        print(f"  id={row_id} {commodity} {pred_date}: entry={entry_price} exit={exit_price}")
        _clear_prices(conn, row_id, "same_day", dry_run)
        _refetch_row(conn, row_id, commodity, pred_date, dry_run)

    # Fix 2: Brent rows sharing Dubai prices
    shared_rows = _find_shared_price_rows(conn)
    print(f"\n[repair] Fix 2 — Brent/Dubai shared price rows: {len(shared_rows)} found")
    for row_id, commodity, pred_date, entry_price, exit_price in shared_rows:
        print(f"  id={row_id} {commodity} {pred_date}: entry={entry_price} exit={exit_price}")
        _clear_prices(conn, row_id, "shared_dubai_price", dry_run)
        _refetch_row(conn, row_id, commodity, pred_date, dry_run)

    if not dry_run:
        conn.commit()
        print("\n[repair] Committed all changes.")
    else:
        print("\n[repair] Dry-run complete — no changes written.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair historical price data in predictions table.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing to DB")
    args = parser.parse_args()
    repair(dry_run=args.dry_run)
