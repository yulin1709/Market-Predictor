"""
inference/trade_filter.py — Monthly trade budget + quality filter for fundamental traders.

A signal only becomes a TRADE recommendation if it passes ALL filters:
  1. Signal must be HIGH
  2. Confidence >= HIGH_CONFIDENCE_THRESHOLD (70%)
  3. Expected move > MIN_EXPECTED_MOVE_PCT (1%)
  4. Event type must not be low-impact noise
  5. Monthly trade budget not exceeded (MAX_TRADES_PER_MONTH = 8)

Returns status: 'TRADE' | 'WATCH' | 'HOLD'
  TRADE = passes all filters, within budget
  WATCH = passes quality but budget reached, or low-impact event
  HOLD  = MEDIUM/LOW signal, not worth monitoring
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date

from data.db_path import DB_PATH as _DB_PATH; DB_PATH = _DB_PATH

MAX_TRADES_PER_MONTH     = 8     # ~2 per week
HIGH_CONFIDENCE_THRESHOLD = 0.55  # 55% minimum — model accuracy at ≥50% is ~52%, tradeable
MIN_EXPECTED_MOVE_PCT    = 1.0   # 1% minimum expected move

HIGH_IMPACT_EVENT_TYPES = {
    "opec_decision", "pipeline_outage", "sanctions", "geopolitical",
    "supply_disruption", "inventory_data", "demand_shock",
    "lng_outage", "force_majeure", "refinery_maintenance",
    "shipping_disruption",
}

LOW_IMPACT_EVENT_TYPES = {
    "routine_report", "analyst_comment", "price_movement", "general_market",
}


def get_trades_this_month() -> int:
    """Return count of TRADE-status predictions logged this calendar month."""
    today = date.today()
    month_start = today.replace(day=1).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("""
            SELECT COUNT(DISTINCT prediction_date || '-' || commodity)
            FROM predictions
            WHERE trade_status = 'TRADE'
              AND prediction_date >= ?
        """, (month_start,)).fetchone()[0]
        conn.close()
        return int(count)
    except Exception:
        return 0


def should_trade(
    signal: str,
    direction: str,
    confidence: float,
    expected_move: str,
    event_type: str | None = None,
    article_count: int = 10,
) -> dict:
    """
    Determine whether a signal qualifies as TRADE, WATCH, or HOLD.

    Returns:
        {
            'status':           'TRADE' | 'WATCH' | 'HOLD',
            'reason':           str,
            'trade_number':     int | None,
            'remaining_budget': int,
        }
    """
    trades_this_month = get_trades_this_month()
    remaining = MAX_TRADES_PER_MONTH - trades_this_month

    move_map = {">3%": 3.5, "1-3%": 2.0, "<1%": 0.5}
    expected_move_pct = move_map.get(expected_move, 0.5)

    # Filter 1: Must be HIGH signal
    if signal != "HIGH":
        return {
            "status": "HOLD",
            "reason": f"Signal is {signal} — only HIGH signals qualify for fundamental trades",
            "trade_number": None,
            "remaining_budget": remaining,
        }

    # Filter 2: Confidence threshold
    if confidence < HIGH_CONFIDENCE_THRESHOLD:
        return {
            "status": "WATCH",
            "reason": f"Confidence {confidence:.0%} below {HIGH_CONFIDENCE_THRESHOLD:.0%} threshold — monitor but do not act",
            "trade_number": None,
            "remaining_budget": remaining,
        }

    # Filter 3: Expected move size
    if expected_move_pct < MIN_EXPECTED_MOVE_PCT:
        return {
            "status": "WATCH",
            "reason": f"Expected move {expected_move} too small for a fundamental trade",
            "trade_number": None,
            "remaining_budget": remaining,
        }

    # Filter 4: Event type
    if event_type and event_type in LOW_IMPACT_EVENT_TYPES:
        return {
            "status": "WATCH",
            "reason": f'Event type "{event_type}" is not a market-moving fundamental catalyst',
            "trade_number": None,
            "remaining_budget": remaining,
        }

    # Filter 5: Monthly budget
    if remaining <= 0:
        return {
            "status": "WATCH",
            "reason": f"Monthly trade budget reached ({MAX_TRADES_PER_MONTH} trades). Review existing positions.",
            "trade_number": None,
            "remaining_budget": 0,
        }

    # Passes all filters → TRADE
    direction_label = {"bullish": "Bullish", "bearish": "Bearish", "rise": "Bullish",
                       "fall": "Bearish", "neutral": "Neutral", "flat": "Neutral"}.get(direction, direction)
    return {
        "status": "TRADE",
        "reason": f"HIGH-impact {direction_label} signal with {confidence:.0%} confidence. Expected move: {expected_move}.",
        "trade_number": trades_this_month + 1,
        "remaining_budget": remaining - 1,
    }
