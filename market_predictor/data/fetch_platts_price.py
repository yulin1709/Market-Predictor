"""
data/fetch_platts_price.py — Fetch Platts benchmark prices from the local prices table.

The prices table is populated by collect_prices.py which parses S&P Global Platts
report body text (already stored in articles.db) and extracts benchmark rows.

This module queries that local table — no additional API calls needed.

Public functions:
    get_platts_price(commodity, date)           -> dict | None
    get_platts_next_close(commodity, from_date) -> dict | None

Both return:
    {"price": float, "source": "platts", "date": "YYYY-MM-DD", "symbol": str}
or None if no price found.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
_MAX_SEARCH_DAYS = 7

# Map commodity display names to the symbol values stored in the prices table
_COMMODITY_TO_PRICES_SYMBOL: dict[str, str] = {
    "Dubai Crude (PCAAT00)": "Dubai",
    "Dubai Crude":           "Dubai",
    "Brent Dated (PCAAS00)": "Brent",
    "Brent":                 "Brent",
    "WTI (PCACG00)":         "WTI",
    "WTI":                   "WTI",
    "LNG / Nat Gas (JKM)":   "LNG",
    "LNG":                   "LNG",
}


def _query_prices_table(symbol: str, date_str: str) -> float | None:
    """Look up a price from the local prices table for a given symbol and date."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT price FROM prices WHERE symbol = ? AND date = ? AND price IS NOT NULL",
            (symbol, date_str),
        ).fetchone()
        conn.close()
        if row and row[0]:
            return round(float(row[0]), 4)
    except Exception as exc:
        print(f"[fetch_platts] DB error for {symbol} on {date_str}: {exc}")
    return None


def get_platts_price(commodity: str, date: str) -> dict | None:
    """
    Return the Platts closing price for `commodity` on `date` (YYYY-MM-DD).

    Queries the local prices table (populated by collect_prices.py).
    Walks forward up to _MAX_SEARCH_DAYS to handle weekends / holidays.

    Returns:
        {"price": float, "source": "platts", "date": str, "symbol": str}
        or None if no price found.
    """
    symbol = _COMMODITY_TO_PRICES_SYMBOL.get(commodity)
    if not symbol:
        return None

    start_dt = datetime.strptime(date, "%Y-%m-%d")
    for offset in range(_MAX_SEARCH_DAYS):
        query_date = start_dt + timedelta(days=offset)
        date_str = query_date.strftime("%Y-%m-%d")
        price = _query_prices_table(symbol, date_str)
        if price is not None:
            return {"price": price, "source": "platts", "date": date_str, "symbol": symbol}

    return None


def get_platts_next_close(commodity: str, from_date: str) -> dict | None:
    """
    Return the Platts closing price for the FIRST trading day STRICTLY AFTER `from_date`.

    Returns:
        {"price": float, "source": "platts", "date": str, "symbol": str}
        or None if no price found within _MAX_SEARCH_DAYS.
    """
    symbol = _COMMODITY_TO_PRICES_SYMBOL.get(commodity)
    if not symbol:
        return None

    # Always start from the NEXT day
    start_dt = datetime.strptime(from_date, "%Y-%m-%d") + timedelta(days=1)

    for offset in range(_MAX_SEARCH_DAYS):
        query_date = start_dt + timedelta(days=offset)
        date_str = query_date.strftime("%Y-%m-%d")

        if date_str == from_date:
            continue

        price = _query_prices_table(symbol, date_str)
        if price is not None:
            return {"price": price, "source": "platts", "date": date_str, "symbol": symbol}

    return None
