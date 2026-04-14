"""
data/fetch_price.py — Two-tier price fetching: Platts first, yfinance fallback.

This is the ONLY module in the project that fetches price data.
All other modules (pipeline.py, backfill_actuals.py) must import from here.

Return type for all public functions is a price dict:
    {
        "price":  float,
        "source": "platts" | "yfinance",
        "date":   "YYYY-MM-DD",
        "symbol": str,
    }
or None if no price could be fetched from either source.

Functions:
    get_closing_price(commodity, date)           -> dict | None
    get_next_trading_close(commodity, from_date) -> dict | None
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DEFAULT_LOT_SIZE = 1000  # barrels or MMBtu per lot


# ── yfinance helpers (fallback) ───────────────────────────────────────────────

def _yf_price_on_date(ticker_yf: str, date_str: str) -> float | None:
    """Fetch a single day's closing price from yfinance. Returns None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    query_date = datetime.strptime(date_str, "%Y-%m-%d")
    next_date = query_date + timedelta(days=1)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(
                ticker_yf,
                start=date_str,
                end=next_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )
        if df is not None and not df.empty and "Close" in df.columns:
            price = float(df["Close"].iloc[0])
            if price > 0:
                return round(price, 4)
    except Exception:
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_closing_price(commodity: str, date: str) -> dict | None:
    """
    Return the closing price for `commodity` on `date` (YYYY-MM-DD).

    Tries Platts first, falls back to yfinance.
    Walks forward up to 5 calendar days to handle weekends / holidays.

    Returns:
        {"price": float, "source": str, "date": str, "symbol": str}
        or None if no data found.
    """
    try:
        from data.ticker_config import COMMODITY_CONFIG, get_platts_symbol
    except ImportError as exc:
        print(f"[fetch_price] Import error: {exc}")
        return None

    cfg = COMMODITY_CONFIG.get(commodity)
    if not cfg:
        print(f"[fetch_price] Unknown commodity: {commodity}")
        return None

    start_dt = datetime.strptime(date, "%Y-%m-%d")

    for offset in range(5):
        query_date = start_dt + timedelta(days=offset)
        date_str = query_date.strftime("%Y-%m-%d")

        # Tier 1: Platts
        try:
            from data.fetch_platts_price import get_platts_price
            result = get_platts_price(commodity, date_str)
            if result:
                return result
        except Exception as exc:
            print(f"[fetch_price] Platts unavailable for {commodity} on {date_str}: {exc}")

        # Tier 2: yfinance
        ticker_yf = cfg.get("ticker_yf", "")
        if ticker_yf:
            price = _yf_price_on_date(ticker_yf, date_str)
            if price is not None:
                return {
                    "price":  price,
                    "source": "yfinance",
                    "date":   date_str,
                    "symbol": ticker_yf,
                }

    return None


def get_next_trading_close(commodity: str, from_date: str) -> dict | None:
    """
    Return the closing price for the FIRST trading day STRICTLY AFTER `from_date`.
    Never returns a price dated the same as from_date.

    Tries Platts first, falls back to yfinance.
    Walks forward up to 5 calendar days to handle weekends / holidays.

    Returns:
        {"price": float, "source": str, "date": str, "symbol": str}
        or None if no data found.
    """
    try:
        from data.ticker_config import COMMODITY_CONFIG
    except ImportError as exc:
        print(f"[fetch_price] Import error: {exc}")
        return None

    cfg = COMMODITY_CONFIG.get(commodity)
    if not cfg:
        print(f"[fetch_price] Unknown commodity: {commodity}")
        return None

    # CRITICAL: always start from the NEXT day
    start_dt = datetime.strptime(from_date, "%Y-%m-%d") + timedelta(days=1)

    for offset in range(7):  # up to 7 days forward — handles long weekends + holidays
        query_date = start_dt + timedelta(days=offset)
        date_str = query_date.strftime("%Y-%m-%d")

        # Safety: never return same-day price
        if date_str == from_date:
            continue

        # Tier 1: Platts
        try:
            from data.fetch_platts_price import get_platts_next_close
            result = get_platts_next_close(commodity, from_date)
            if result:
                return result
        except Exception:
            pass

        # Tier 2: yfinance
        ticker_yf = cfg.get("ticker_yf", "")
        if ticker_yf:
            price = _yf_price_on_date(ticker_yf, date_str)
            if price is not None:
                return {
                    "price":  price,
                    "source": "yfinance",
                    "date":   date_str,
                    "symbol": ticker_yf,
                }

    return None
