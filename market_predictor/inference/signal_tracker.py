"""
inference/signal_tracker.py — Honest trading signal tracker for fundamental analysts.

Data integrity rules:
- ALL signals labelled [LIVE] or [BACKTEST]
- LIVE = logged on the day, entry = next-day open (simulated via open price)
- BACKTEST = walk-forward: model trained on data up to T, signal for T+1 only
- Confidence = dynamic from model output, never hardcoded
- Entry price = open of next candle after signal (not the close that triggered it)
- Slippage = 0.15% per trade applied to entry and exit
- Spread cost deducted from P&L per commodity
- Metrics flagged with ⚠️ if n < 30 trades
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.db_path import DB_PATH as _DB_PATH; DB_PATH = _DB_PATH

# Cost model per commodity (USD per barrel / MMBtu)
SPREAD_COST: dict[str, float] = {
    "Dubai Crude (PCAAT00)": 0.05,
    "Brent Dated (PCAAS00)": 0.04,
    "WTI (PCACG00)":         0.04,
    "LNG / Nat Gas (JKM)":   0.02,
}
SLIPPAGE_PCT = 0.0015   # 0.15% per trade
DEFAULT_LOT  = 1000     # barrels / MMBtu
LOW_SAMPLE_THRESHOLD = 30


@dataclass
class SignalRow:
    row_id: int
    date: str
    commodity: str
    catalyst: str           # headline / event description
    event_type: str
    signal: str             # BUY / SELL / HOLD
    direction: str          # bullish / bearish / neutral
    confidence_pct: float   # dynamic from model, 0-100
    entry_price: Optional[float]
    exit_price: Optional[float]
    return_pct_net: Optional[float]   # after slippage + spread
    pnl_usd_net: Optional[float]      # after costs
    outcome: str            # win / loss / cut-loss / neutral / pending
    signal_type: str        # LIVE / BACKTEST
    impact_score: int
    trade_status: str       # TRADE / WATCH / HOLD


@dataclass
class SummaryMetrics:
    n_trades: int = 0
    n_live: int = 0
    n_backtest: int = 0
    win_rate_pct: float = 0.0
    win_rate_low_sample: bool = False
    profit_factor: float = 0.0
    profit_factor_low_sample: bool = False
    max_drawdown_usd: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    total_pnl_usd: float = 0.0
    monthly_pnl: dict = field(default_factory=dict)   # {YYYY-MM: float}
    note: str = ""
    mixed_warning: str = ""   # non-empty if live+backtest are mixed


def _apply_costs(
    entry: float,
    exit_: float,
    direction: str,
    commodity: str,
    lot: int = DEFAULT_LOT,
) -> tuple[float, float, float]:
    """
    Apply slippage and spread to entry/exit prices.
    Returns (net_entry, net_exit, cost_usd).
    """
    spread = SPREAD_COST.get(commodity, 0.05)
    slip = entry * SLIPPAGE_PCT

    if direction in ("bullish", "rise", "BUY"):
        net_entry = entry + slip          # buy at slightly higher price
        net_exit  = exit_ - slip          # sell at slightly lower price
    else:
        net_entry = entry - slip          # short at slightly lower price
        net_exit  = exit_ + slip          # cover at slightly higher price

    cost_usd = (spread * 2 + slip * 2) * lot   # round-trip cost
    return net_entry, net_exit, cost_usd


def load_signals(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load all predictions with signal_type labelling.
    LIVE = prediction_date >= model training cutoff date (from model_info.pkl).
    BACKTEST = prediction_date < training cutoff.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass

    # Determine training cutoff from model_info
    training_cutoff = _get_training_cutoff()

    df = pd.read_sql_query("""
        SELECT
            id,
            prediction_date,
            commodity,
            COALESCE(headline, '—')     AS catalyst,
            COALESCE(event_type, 'other') AS event_type,
            signal,
            direction,
            confidence,
            entry_price,
            exit_price,
            actual_move,
            pnl_usd,
            return_pct,
            COALESCE(trade_outcome, outcome) AS outcome,
            COALESCE(trade_status, 'HOLD')   AS trade_status,
            COALESCE(impact_score, 2)        AS impact_score,
            COALESCE(cut_loss_price, NULL)   AS cut_loss_price
        FROM predictions
        ORDER BY prediction_date DESC
    """, conn)

    if df.empty:
        return df

    # Label LIVE vs BACKTEST
    df["signal_type"] = df["prediction_date"].apply(
        lambda d: "LIVE" if str(d) >= training_cutoff else "BACKTEST"
    )

    # Map direction to BUY/SELL/HOLD signal column
    def _to_signal(row):
        if row["trade_status"] == "HOLD":
            return "HOLD"
        d = str(row["direction"] or "").lower()
        if d in ("bullish", "rise"):
            return "BUY"
        if d in ("bearish", "fall"):
            return "SELL"
        return "HOLD"

    df["signal_label"] = df.apply(_to_signal, axis=1)

    # Normalise confidence to 0-100
    df["confidence_pct"] = df["confidence"].apply(
        lambda x: round(float(x) * 100, 1) if (x is not None and float(x) <= 1.0)
        else (round(float(x), 1) if x is not None else 0.0)
    )

    # Apply costs to compute net return and net P&L
    def _net_pnl(row):
        ep = row.get("entry_price")
        xp = row.get("exit_price")
        direction = str(row.get("direction", "")).lower()
        commodity = row.get("commodity", "")
        if not ep or not xp or ep <= 0:
            return None, None
        net_e, net_x, cost = _apply_costs(ep, xp, direction, commodity)
        if direction in ("bullish", "rise", "BUY"):
            raw_pnl = (net_x - net_e) * DEFAULT_LOT
        else:
            raw_pnl = (net_e - net_x) * DEFAULT_LOT
        raw_pnl -= cost
        net_ret = (net_x - net_e) / net_e * 100 if direction in ("bullish", "rise") \
                  else (net_e - net_x) / net_e * 100
        return round(raw_pnl, 2), round(net_ret, 4)

    df[["pnl_net", "return_net"]] = df.apply(
        lambda r: pd.Series(_net_pnl(r)), axis=1
    )

    return df


def compute_summary(df: pd.DataFrame, signal_type_filter: str = "ALL") -> SummaryMetrics:
    """
    Compute summary metrics. Only uses TRADE-status rows.
    signal_type_filter: 'ALL' | 'LIVE' | 'BACKTEST'
    """
    if df.empty:
        return SummaryMetrics(note="No data.")

    trades = df[df["trade_status"] == "TRADE"].copy()

    if signal_type_filter != "ALL":
        trades = trades[trades["signal_type"] == signal_type_filter]

    resolved = trades[trades["outcome"].isin(["win", "loss", "cut-loss", "neutral"])].copy()
    n = len(resolved)

    if n == 0:
        return SummaryMetrics(note="No resolved trades yet.")

    wins   = resolved[resolved["outcome"] == "win"]
    losses = resolved[resolved["outcome"].isin(["loss", "cut-loss"])]
    decided = len(wins) + len(losses)

    win_rate = round(len(wins) / decided * 100, 1) if decided > 0 else 0.0

    gross_profit = wins["pnl_net"].sum() if not wins.empty else 0.0
    gross_loss   = abs(losses["pnl_net"].sum()) if not losses.empty else 0.0
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    pnl_series = resolved.sort_values("prediction_date")["pnl_net"].fillna(0)
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = round(float(drawdown.min()), 2)

    avg_win  = round(wins["return_net"].mean(), 4) if not wins.empty else 0.0
    avg_loss = round(losses["return_net"].mean(), 4) if not losses.empty else 0.0
    total_pnl = round(resolved["pnl_net"].sum(), 2)

    # Monthly P&L
    resolved["month"] = resolved["prediction_date"].astype(str).str[:7]
    monthly = resolved.groupby("month")["pnl_net"].sum().round(2).to_dict()

    # Mixed warning
    n_live = len(resolved[resolved["signal_type"] == "LIVE"])
    n_bt   = len(resolved[resolved["signal_type"] == "BACKTEST"])
    mixed_warn = ""
    if signal_type_filter == "ALL" and n_live > 0 and n_bt > 0:
        mixed_warn = (
            f"⚠️ Metrics include {n_live} LIVE and {n_bt} BACKTEST signals. "
            "Live and backtest performance should be evaluated separately."
        )

    return SummaryMetrics(
        n_trades=n,
        n_live=n_live,
        n_backtest=n_bt,
        win_rate_pct=win_rate,
        win_rate_low_sample=decided < LOW_SAMPLE_THRESHOLD,
        profit_factor=pf,
        profit_factor_low_sample=n < LOW_SAMPLE_THRESHOLD,
        max_drawdown_usd=max_dd,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        total_pnl_usd=total_pnl,
        monthly_pnl=monthly,
        mixed_warning=mixed_warn,
    )


def _get_training_cutoff() -> str:
    """Return the model training cutoff date (ISO string). Signals after this = LIVE."""
    try:
        import joblib
        save_dir = Path(__file__).resolve().parent.parent / "models" / "saved"
        info = joblib.load(save_dir / "model_info.pkl")
        cutoff = info.get("test_cutoff") or info.get("trained_at", "")
        if cutoff:
            return str(cutoff)[:10]
    except Exception:
        pass
    # Fallback: treat anything in the last 30 days as LIVE
    return (date.today() - timedelta(days=30)).isoformat()
