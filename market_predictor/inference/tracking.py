"""
inference/tracking.py — Trade simulation logic for the Prediction Tracker.

Target user: Fundamental traders and market analysts who act on HIGH-impact
news events only. This is NOT a technical trading system.

Key design decisions:
- Only HIGH signal trades are taken (fundamental analysts act on significant news)
- Trade frequency capped at ~8 per month (2 per week) — quality over quantity
- Direction labels: Bullish / Bearish (not LONG/SHORT — fundamental language)
- Cut-loss: if a trade hits stop-loss threshold, it is flagged as cut-loss
- P&L = (exit - entry) * lot_size for Bullish, (entry - exit) * lot_size for Bearish
- Neutral zone: |return| < NEUTRAL_THRESHOLD → neither win nor loss
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

import pandas as pd

from data.db_path import DB_PATH as _DB_PATH; DB_PATH = _DB_PATH

# Trading parameters
DEFAULT_LOT_SIZE = 1000       # barrels or MMBtu per lot
NEUTRAL_THRESHOLD = 0.20      # % — moves smaller than this are "neutral" (noise)
HIGH_CONFIDENCE_THRESHOLD = 0.55   # 55% — matches trade_filter.py; model accuracy at ≥50% is ~52%

# Frequency cap: max trades per calendar month
MAX_TRADES_PER_MONTH = 8

# Cut-loss threshold: if loss exceeds this %, flag as cut-loss
CUT_LOSS_THRESHOLD_PCT = 2.0  # 2% adverse move = cut-loss signal


@dataclass
class TradeResult:
    trade_direction: str    # Bullish / Bearish / Hold
    pnl_usd: float
    return_pct: float
    outcome: str            # win / loss / cut-loss / neutral / pending / hold


def direction_to_label(direction: str) -> str:
    """Convert rise/fall/flat to Bullish/Bearish/Neutral for fundamental analyst language."""
    if direction == "rise":
        return "Bullish"
    if direction == "fall":
        return "Bearish"
    return "Neutral"


def get_trade_direction(signal: str, direction: str, confidence: float,
                        impact_score: int = 5) -> str:
    """
    Determine trade direction from signal and direction.

    Rules for fundamental analysts:
    - Only HIGH signal → trade
    - HIGH + bullish → Bullish
    - HIGH + bearish → Bearish
    - MEDIUM/LOW or neutral → Hold
    - Confidence below threshold → Hold
    Note: impact_score filter is applied at signal generation time (trade_filter.py),
    not here, to avoid filtering out historical predictions with missing impact data.
    """
    if confidence <= 1.0:
        conf_pct = confidence * 100
    else:
        conf_pct = confidence

    if conf_pct < HIGH_CONFIDENCE_THRESHOLD * 100:
        return "Hold"

    if signal == "HIGH":
        if direction in ("rise", "bullish"):
            return "Bullish"
        elif direction in ("fall", "bearish"):
            return "Bearish"

    return "Hold"


def compute_trade_pnl(
    entry_price: float,
    exit_price: float,
    trade_direction: str,
    lot_size: int = DEFAULT_LOT_SIZE,
    cut_loss_price: float | None = None,
) -> tuple[float, float, str]:
    """
    Compute P&L and outcome for a trade.
    If cut_loss_price is provided and hit, caps the loss at the cut-loss level.
    """
    if trade_direction == "Hold":
        return 0.0, 0.0, "hold"

    if not entry_price or not exit_price or entry_price <= 0:
        return 0.0, 0.0, "pending"

    price_diff = exit_price - entry_price
    return_pct = (price_diff / entry_price) * 100

    if trade_direction == "Bullish":
        effective_return = return_pct
        # Apply cut-loss cap: if price fell below cut_loss_price, cap exit at cut_loss_price
        if cut_loss_price and exit_price < cut_loss_price:
            effective_exit = cut_loss_price
            pnl_usd = (effective_exit - entry_price) * lot_size
            effective_return = (effective_exit - entry_price) / entry_price * 100
            return round(pnl_usd, 2), round(return_pct, 4), "cut-loss"
        pnl_usd = price_diff * lot_size
    elif trade_direction == "Bearish":
        effective_return = -return_pct
        # Apply cut-loss cap: if price rose above cut_loss_price, cap exit at cut_loss_price
        if cut_loss_price and exit_price > cut_loss_price:
            effective_exit = cut_loss_price
            pnl_usd = -(effective_exit - entry_price) * lot_size
            effective_return = -(effective_exit - entry_price) / entry_price * 100
            return round(pnl_usd, 2), round(return_pct, 4), "cut-loss"
        pnl_usd = -price_diff * lot_size
    else:
        return 0.0, 0.0, "hold"

    if abs(effective_return) < NEUTRAL_THRESHOLD:
        outcome = "neutral"
    elif effective_return > 0:
        outcome = "win"
    elif effective_return <= -CUT_LOSS_THRESHOLD_PCT:
        outcome = "cut-loss"
    else:
        outcome = "loss"

    return round(pnl_usd, 2), round(return_pct, 4), outcome


def _apply_monthly_cap(rows: list[tuple]) -> list[tuple]:
    """
    Apply monthly trade frequency cap (MAX_TRADES_PER_MONTH).
    Within each month, keep only the highest-confidence HIGH-signal trades.
    Returns filtered list of rows.
    """
    from collections import defaultdict
    monthly: dict[str, list] = defaultdict(list)
    for row in rows:
        row_id, signal, direction, confidence, entry_price, exit_price, pred_date = row
        if signal == "HIGH" and entry_price is not None:
            month_key = str(pred_date)[:7]  # YYYY-MM
            monthly[month_key].append(row)

    kept = []
    for month_key, month_rows in monthly.items():
        month_rows.sort(key=lambda r: r[3] or 0, reverse=True)
        kept.extend(month_rows[:MAX_TRADES_PER_MONTH])

    # Include non-HIGH rows and rows without entry_price (they'll be HOLD/pending)
    high_ids = {r[0] for r in kept}
    non_high = [r for r in rows if r[1] != "HIGH" or r[4] is None]
    return kept + non_high


def recompute_all_trades(conn: sqlite3.Connection) -> int:
    """
    Recompute trade_direction, pnl_usd, return_pct, and outcome for all
    resolved predictions using fundamental analyst trading logic.

    - Respects trade_status='TRADE' already set in DB (from trade_filter.py)
    - Falls back to monthly cap logic only for rows without trade_status
    - Adds columns if missing. Returns count of rows updated.
    """
    for col, col_type in [
        ("trade_direction", "TEXT"),
        ("return_pct",      "REAL"),
        ("trade_outcome",   "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass

    # Enable WAL so background collect_news.py doesn't block us
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass

    rows = conn.execute("""
        SELECT id, signal, direction, confidence, entry_price, exit_price, prediction_date,
               COALESCE(cut_loss_price, NULL) as cut_loss_price,
               COALESCE(trade_status, '') as trade_status
        FROM predictions
        ORDER BY prediction_date ASC
    """).fetchall()

    # Build capped_ids from DB trade_status first (authoritative)
    db_trade_ids = {r[0] for r in rows if r[8] == 'TRADE'}

    # For rows without trade_status, fall back to monthly cap logic
    rows_without_status = [(r[0],r[1],r[2],r[3],r[4],r[5],r[6]) for r in rows if r[8] != 'TRADE']
    capped_fallback = _apply_monthly_cap(rows_without_status)
    capped_fallback_ids = {r[0] for r in capped_fallback if r[1] == "HIGH" and r[4] is not None}

    capped_ids = db_trade_ids | capped_fallback_ids

    updated = 0
    for row in rows:
        row_id, signal, direction, confidence, entry_price, exit_price, pred_date, cut_loss_price, trade_status = row

        # Normalise direction
        _dmap = {"rise": "bullish", "fall": "bearish", "flat": "neutral",
                 "LONG": "bullish", "SHORT": "bearish", "HOLD": "neutral", "Hold": "neutral"}
        direction_norm = _dmap.get(direction or "", direction or "neutral").lower()
        if direction != direction_norm:
            conn.execute("UPDATE predictions SET direction=? WHERE id=?", (direction_norm, row_id))

        # Trade direction: TRADE rows or capped HIGH → Bullish/Bearish; everything else → Hold
        if row_id in capped_ids:
            trade_dir = get_trade_direction(signal or "LOW", direction_norm, confidence or 0)
        else:
            trade_dir = "Hold"

        if exit_price is not None and entry_price is not None:
            pnl, ret_pct, trade_outcome = compute_trade_pnl(
                entry_price, exit_price, trade_dir, cut_loss_price=cut_loss_price
            )
        else:
            pnl, ret_pct, trade_outcome = 0.0, 0.0, "pending"

        # Only update pnl_usd for actual trades — don't zero out Hold rows that have
        # a valid pnl_usd already set by backfill_actuals
        if trade_dir != "Hold":
            conn.execute("""
                UPDATE predictions
                SET trade_direction = ?, return_pct = ?, trade_outcome = ?, pnl_usd = ?
                WHERE id = ?
            """, (trade_dir, ret_pct, trade_outcome, pnl, row_id))
        else:
            conn.execute("""
                UPDATE predictions
                SET trade_direction = ?, return_pct = ?, trade_outcome = ?
                WHERE id = ?
            """, (trade_dir, ret_pct, trade_outcome, row_id))
        updated += 1

    conn.commit()
    return updated


def load_trading_df(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all predictions with trading columns as a DataFrame."""
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return pd.read_sql_query("""
        SELECT id, prediction_date, commodity, ticker, signal, direction,
               confidence, expected_move, headline,
               entry_price, exit_price, price_unit,
               actual_move, outcome, pnl_usd,
               trade_direction, return_pct, trade_outcome
        FROM predictions
        ORDER BY prediction_date DESC
    """, conn)
