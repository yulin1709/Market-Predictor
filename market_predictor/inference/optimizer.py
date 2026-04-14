"""
inference/optimizer.py — Entry Optimizer core calculations.

Generates trade setups from model predictions + price data.
No additional APIs required — uses existing model outputs and prices table.

Public API:
    build_setup(sig, entry_price, historical_df)  -> TradeSetup | None
    build_all_setups(signals, conn)               -> list[TradeSetup]
    summary_stats(setups)                         -> dict
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

# Entry zone half-width as % of price
ENTRY_ZONE_PCT = 0.003          # ±0.3%

# Target move by signal level (min%, max%, midpoint%)
TARGET_MOVE: dict[str, tuple[float, float, float]] = {
    "HIGH":   (2.0, 4.0, 3.0),
    "MEDIUM": (1.0, 2.0, 1.5),
    "LOW":    (0.3, 1.0, 0.5),
}

# Default stop-loss % by signal level
STOP_LOSS_PCT: dict[str, float] = {
    "HIGH":   1.0,
    "MEDIUM": 0.8,
    "LOW":    0.5,
}

# Minimum confidence (0-1) to generate a setup — below this → NO TRADE
# Raised for fundamental analysts who only act on high-conviction signals
MIN_CONFIDENCE = 0.50

# Minimum R:R to flag as "best setup"
MIN_RR_BEST = 1.5

# EV threshold to flag as high-EV
EV_POSITIVE_THRESHOLD = 0.0

from data.db_path import DB_PATH as _DB_PATH; DB_PATH = _DB_PATH


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TradeSetup:
    commodity: str
    ticker: str
    signal: str                  # HIGH / MEDIUM / LOW
    direction: str               # rise / fall / flat
    trade_direction: str         # LONG / SHORT / NO TRADE

    entry_price: float
    entry_low: float             # entry zone lower bound
    entry_high: float            # entry zone upper bound

    target_price: float
    target_move_pct: float       # expected move %
    target_move_min: float
    target_move_max: float

    stop_loss: float
    stop_loss_pct: float

    risk_usd: float              # per lot
    reward_usd: float            # per lot
    risk_reward: float           # reward / risk

    confidence: float            # 0.0 – 1.0 (raw model)
    confidence_pct: float        # display value (scaled 35-95)

    win_rate: float              # historical win rate for similar setups
    expected_value: float        # EV = win_rate*reward - (1-win_rate)*risk (per lot)

    rating: str                  # strong / moderate / cautious / skip
    rationale: str

    lot_size: int = 1000
    headline: str = ""
    prediction_date: str = ""

    # derived
    is_best: bool = field(init=False)

    def __post_init__(self):
        self.is_best = (
            self.trade_direction != "NO TRADE"
            and self.risk_reward >= MIN_RR_BEST
            and self.expected_value > EV_POSITIVE_THRESHOLD
        )


# ── Historical win rate ───────────────────────────────────────────────────────

def _historical_win_rate(commodity: str, signal: str, direction: str,
                          conn: sqlite3.Connection) -> float:
    """
    Compute historical win rate for the same commodity + signal + direction
    from the predictions table.

    Falls back to signal-level defaults if insufficient history.
    """
    defaults = {"HIGH": 0.55, "MEDIUM": 0.48, "LOW": 0.40}
    try:
        rows = conn.execute("""
            SELECT outcome
            FROM predictions
            WHERE commodity = ?
              AND signal = ?
              AND direction = ?
              AND outcome IN ('correct', 'incorrect')
        """, (commodity, signal, direction)).fetchall()

        if len(rows) < 5:
            # Not enough history — use signal-level default
            return defaults.get(signal, 0.45)

        correct = sum(1 for r in rows if r[0] == "correct")
        return round(correct / len(rows), 3)
    except Exception:
        return defaults.get(signal, 0.45)


def _recent_volatility(commodity: str, conn: sqlite3.Connection,
                        lookback: int = 10) -> float:
    """
    Estimate recent price volatility as avg |actual_move| over last N resolved predictions.
    Used to widen/narrow entry zone. Returns 0.003 (0.3%) as default.
    """
    try:
        rows = conn.execute("""
            SELECT actual_move FROM predictions
            WHERE commodity = ?
              AND actual_move IS NOT NULL
            ORDER BY prediction_date DESC
            LIMIT ?
        """, (commodity, lookback)).fetchall()
        if not rows:
            return 0.003
        avg_abs = sum(abs(r[0]) for r in rows) / len(rows) / 100
        # Clamp to [0.001, 0.01]
        return max(0.001, min(avg_abs * 0.3, 0.01))
    except Exception:
        return 0.003


# ── Core builder ─────────────────────────────────────────────────────────────

def build_setup(
    sig: dict,
    entry_price: float,
    conn: sqlite3.Connection,
    lot_size: int = 1000,
    stop_pct_override: Optional[float] = None,
) -> Optional[TradeSetup]:
    """
    Build a TradeSetup from a commodity signal dict and current price.

    Args:
        sig:               commodity signal dict from compute_today_signals()
        entry_price:       current market price in USD
        conn:              open SQLite connection (for historical win rate)
        lot_size:          barrels or MMBtu per lot
        stop_pct_override: override stop-loss % (0-100 scale), or None for default

    Returns:
        TradeSetup, or None if direction is flat / confidence too low.
    """
    commodity = sig["commodity"]
    ticker = sig["ticker"]
    signal = sig["signal"]
    direction = sig["direction"]
    confidence_pct = float(sig.get("scaled_conf", 35.0))
    confidence = confidence_pct / 100.0

    # Only generate setups for HIGH signal — fundamental analysts act on significant news only
    if signal != "HIGH":
        return None  # MEDIUM/LOW → no setup for fundamental analysts

    # Determine trade direction (Bullish/Bearish language)
    if direction == "rise":
        trade_dir = "Bullish"
    elif direction == "fall":
        trade_dir = "Bearish"
    else:
        return None  # flat → no trade

    if confidence < MIN_CONFIDENCE:
        return None  # below threshold → skip

    # Entry zone (±volatility-adjusted)
    zone_pct = _recent_volatility(commodity, conn)
    entry_low = round(entry_price * (1 - zone_pct), 4)
    entry_high = round(entry_price * (1 + zone_pct), 4)

    # Target price
    move_min, move_max, move_mid = TARGET_MOVE.get(signal, TARGET_MOVE["LOW"])
    # Scale target by confidence: higher confidence → aim for upper end of range
    conf_factor = min((confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE), 1.0)
    target_move_pct = move_min + (move_max - move_min) * conf_factor

    if trade_dir == "Bullish":
        target_price = round(entry_price * (1 + target_move_pct / 100), 4)
    else:
        target_price = round(entry_price * (1 - target_move_pct / 100), 4)

    # Stop loss
    sl_pct = (stop_pct_override / 100.0) if stop_pct_override else STOP_LOSS_PCT.get(signal, 1.0) / 100.0
    if trade_dir == "Bullish":
        stop_loss = round(entry_price * (1 - sl_pct), 4)
    else:
        stop_loss = round(entry_price * (1 + sl_pct), 4)

    # Risk / reward per lot
    risk_usd = round(abs(entry_price - stop_loss) * lot_size, 2)
    reward_usd = round(abs(target_price - entry_price) * lot_size, 2)
    risk_reward = round(reward_usd / risk_usd, 2) if risk_usd > 0 else 0.0

    # Historical win rate
    win_rate = _historical_win_rate(commodity, signal, direction, conn)

    # Expected Value per lot
    ev = round(win_rate * reward_usd - (1 - win_rate) * risk_usd, 2)

    # Rating
    if confidence >= 0.80 and risk_reward >= 2.0:
        rating = "strong"
    elif confidence >= 0.65 and risk_reward >= 1.5:
        rating = "moderate"
    elif confidence >= MIN_CONFIDENCE:
        rating = "cautious"
    else:
        rating = "skip"

    rationale = _build_rationale(signal, trade_dir, confidence_pct, risk_reward, win_rate, rating)

    return TradeSetup(
        commodity=commodity,
        ticker=ticker,
        signal=signal,
        direction=direction,
        trade_direction=trade_dir,
        entry_price=round(entry_price, 4),
        entry_low=entry_low,
        entry_high=entry_high,
        target_price=target_price,
        target_move_pct=round(target_move_pct, 2),
        target_move_min=move_min,
        target_move_max=move_max,
        stop_loss=stop_loss,
        stop_loss_pct=round(sl_pct * 100, 2),
        risk_usd=risk_usd,
        reward_usd=reward_usd,
        risk_reward=risk_reward,
        confidence=confidence,
        confidence_pct=confidence_pct,
        win_rate=win_rate,
        expected_value=ev,
        rating=rating,
        rationale=rationale,
        lot_size=lot_size,
        headline=sig.get("top_headline", ""),
        prediction_date=sig.get("prediction_date", ""),
    )


def _build_rationale(signal: str, trade_dir: str, conf_pct: float,
                      rr: float, win_rate: float, rating: str) -> str:
    wr_pct = int(win_rate * 100)
    sentiment = "Bullish" if trade_dir == "Bullish" else "Bearish"
    if rating == "strong":
        return (f"HIGH impact — {sentiment} signal at {conf_pct:.0f}% confidence. "
                f"R:R {rr:.1f}:1, {wr_pct}% historical win rate. Strong fundamental setup.")
    if rating == "moderate":
        return (f"HIGH impact — {sentiment} signal at {conf_pct:.0f}% confidence. "
                f"R:R {rr:.1f}:1, {wr_pct}% historical win rate. Acceptable setup.")
    if rating == "cautious":
        return (f"HIGH impact — {sentiment} signal at {conf_pct:.0f}% confidence. "
                f"R:R {rr:.1f}:1. Proceed with reduced position size.")
    return f"Confidence too low ({conf_pct:.0f}%). Wait for stronger fundamental signal."


# ── Batch builder ─────────────────────────────────────────────────────────────

def build_all_setups(
    signals: list[dict],
    conn: sqlite3.Connection,
    lot_size: int = 1000,
    stop_pct_override: Optional[float] = None,
) -> list[TradeSetup]:
    """
    Build trade setups for all commodity signals that have a current price.

    Fetches entry price via the two-tier fetcher (Platts → yfinance).
    Skips commodities where price is unavailable.
    """
    from datetime import date as _date
    today = _date.today().isoformat()

    setups: list[TradeSetup] = []
    for sig in signals:
        commodity = sig["commodity"]
        try:
            from data.fetch_price import get_closing_price
            price_result = get_closing_price(commodity, today)
            if not price_result:
                continue
            entry_price = price_result["price"]
        except Exception:
            continue

        setup = build_setup(sig, entry_price, conn, lot_size=lot_size,
                            stop_pct_override=stop_pct_override)
        if setup:
            setups.append(setup)

    return setups


# ── Summary stats ─────────────────────────────────────────────────────────────

def summary_stats(setups: list[TradeSetup]) -> dict:
    """Compute summary statistics across all setups."""
    if not setups:
        return {
            "total": 0, "best_count": 0,
            "avg_rr": 0.0, "avg_ev": 0.0,
            "high_conf_pct": 0.0, "avg_win_rate": 0.0,
        }
    total = len(setups)
    best = sum(1 for s in setups if s.is_best)
    avg_rr = round(sum(s.risk_reward for s in setups) / total, 2)
    avg_ev = round(sum(s.expected_value for s in setups) / total, 2)
    high_conf = sum(1 for s in setups if s.confidence_pct >= 70)
    avg_wr = round(sum(s.win_rate for s in setups) / total * 100, 1)
    return {
        "total": total,
        "best_count": best,
        "avg_rr": avg_rr,
        "avg_ev": avg_ev,
        "high_conf_pct": round(high_conf / total * 100, 0),
        "avg_win_rate": avg_wr,
    }


# ── Historical setups from predictions table ──────────────────────────────────

def load_historical_setups(conn: sqlite3.Connection,
                            days: int = 30) -> pd.DataFrame:
    """
    Load past predictions and reconstruct trade setup columns for display.
    Used in the historical setups table in the dashboard.
    """
    from datetime import date, timedelta
    since = (date.today() - timedelta(days=days)).isoformat()

    try:
        df = pd.read_sql_query(
            """
            SELECT
                prediction_date, commodity, signal, direction,
                confidence, entry_price, exit_price,
                actual_move, outcome, pnl_usd,
                trade_direction, return_pct, trade_outcome,
                headline, entry_price_source
            FROM predictions
            WHERE prediction_date >= ?
              AND entry_price IS NOT NULL
            ORDER BY prediction_date DESC
            """,
            conn,
            params=(since,),
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # Reconstruct target / stop / R:R from stored prices
    def _target(row):
        sig = row["signal"]
        _, _, mid = TARGET_MOVE.get(sig, TARGET_MOVE["LOW"])
        if row["direction"] == "rise":
            return round(row["entry_price"] * (1 + mid / 100), 4)
        return round(row["entry_price"] * (1 - mid / 100), 4)

    def _stop(row):
        sl = STOP_LOSS_PCT.get(row["signal"], 1.0) / 100.0
        if row["direction"] == "rise":
            return round(row["entry_price"] * (1 - sl), 4)
        return round(row["entry_price"] * (1 + sl), 4)
    df["target_price"] = df.apply(_target, axis=1)
    df["stop_loss"] = df.apply(_stop, axis=1)
    df["risk_usd"] = (df["entry_price"] - df["stop_loss"]).abs() * 1000
    df["reward_usd"] = (df["target_price"] - df["entry_price"]).abs() * 1000
    df["risk_reward"] = (df["reward_usd"] / df["risk_usd"].replace(0, float("nan"))).round(2)
    df["confidence_pct"] = (df["confidence"] * 100).round(1)

    return df
