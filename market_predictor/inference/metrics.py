"""
inference/metrics.py — Compute both ML model metrics and trading performance metrics.

TWO separate metric sets:
  A. Model Performance (offline ML evaluation) — from training/test split
  B. Trading Performance (backtest simulation) — from predictions table

These must NEVER be mixed. Model accuracy != trading win rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── A. Model Performance ─────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    """Offline ML evaluation metrics from training/test split."""
    model_type: str = "unknown"
    train_samples: int = 0
    test_samples: int = 0
    test_cutoff: str = ""
    overall_accuracy: float = 0.0
    high_precision: float = 0.0
    high_recall: float = 0.0
    medium_precision: float = 0.0
    low_precision: float = 0.0
    confusion_matrix: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    note: str = ""


def load_model_metrics() -> ModelMetrics:
    """Load model performance metrics from saved model_info.pkl."""
    try:
        import joblib
        save_dir = Path(__file__).resolve().parent.parent / "models" / "saved"
        info = joblib.load(save_dir / "model_info.pkl")

        cr = info.get("classification_report", {})
        high = cr.get("HIGH", cr.get("rise", {}))  # support both magnitude and direction models
        medium = cr.get("MEDIUM", {})
        low = cr.get("LOW", {})

        # test_samples: total samples evaluated across all CV folds
        test_samples = (
            info.get("test_samples")
            or sum(f.get("test_size", 0) for f in info.get("cv_folds", []))
            or info.get("n_folds", 0)
            or 0
        )

        return ModelMetrics(
            model_type=info.get("model_type", "unknown"),
            train_samples=info.get("train_samples") or 0,
            test_samples=int(test_samples),
            test_cutoff=info.get("test_cutoff", info.get("split_mode", "")),
            overall_accuracy=round((cr.get("accuracy") or 0) * 100, 1),
            high_precision=round((high.get("precision") or 0) * 100, 1),
            high_recall=round((high.get("recall") or 0) * 100, 1),
            medium_precision=round((medium.get("precision") or 0) * 100, 1),
            low_precision=round((low.get("precision") or 0) * 100, 1),
            confusion_matrix=info.get("confusion_matrix", []),
            classes=info.get("classes", info.get("labels", ["HIGH", "LOW", "MEDIUM"])),
            note="Evaluated via walk-forward CV (no data leakage). test_samples = total across all folds.",
        )
    except Exception as e:
        return ModelMetrics(note=f"Model metrics unavailable: {e}")


# ── B. Trading Performance ────────────────────────────────────────────────────

@dataclass
class TradingMetrics:
    """Backtest simulation metrics from predictions table.
    Designed for fundamental traders and market analysts."""
    total_predictions: int = 0
    total_trades: int = 0        # Bullish + Bearish (excludes Hold)
    wins: int = 0
    losses: int = 0
    cut_losses: int = 0          # trades that hit cut-loss threshold (>2% adverse)
    neutrals: int = 0
    holds: int = 0
    pending: int = 0
    win_rate: float = 0.0        # wins / (wins + losses + cut_losses)
    total_pnl: float = 0.0
    avg_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    # Risk metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None   # downside-only risk
    profit_factor: float = 0.0             # gross profit / gross loss
    value_at_risk_95: float = 0.0          # 95% VaR (worst 5% of trades)
    avg_trades_per_month: float = 0.0
    note: str = ""


def compute_trading_metrics(df: pd.DataFrame) -> TradingMetrics:
    """
    Compute trading performance metrics for fundamental analysts.
    Uses Bullish/Bearish direction labels. Only HIGH-signal trades counted.
    """
    if df.empty:
        return TradingMetrics(note="No predictions available.")

    for col in ("trade_direction", "trade_outcome", "pnl_usd", "return_pct"):
        if col not in df.columns:
            df[col] = None

    total = len(df)
    # Handle both old (LONG/SHORT/HOLD) and new (Bullish/Bearish/Hold) direction labels
    _bullish_vals = {"Bullish", "LONG"}
    _bearish_vals = {"Bearish", "SHORT"}
    _hold_vals = {"Hold", "HOLD"}
    trades = df[df["trade_direction"].isin(_bullish_vals | _bearish_vals)]
    resolved = trades[trades["trade_outcome"].isin(["win", "loss", "cut-loss", "neutral"])]
    wins_df      = resolved[resolved["trade_outcome"] == "win"]
    losses_df    = resolved[resolved["trade_outcome"].isin(["loss", "cut-loss"])]
    cut_loss_df  = resolved[resolved["trade_outcome"] == "cut-loss"]
    neutrals_df  = resolved[resolved["trade_outcome"] == "neutral"]
    holds        = df[df["trade_direction"].isin(_hold_vals)]
    pending      = df[df["trade_outcome"] == "pending"]

    win_count      = len(wins_df)
    loss_count     = len(losses_df)
    cut_loss_count = len(cut_loss_df)
    decided        = win_count + loss_count  # excludes neutral

    win_rate   = round(win_count / decided * 100, 1) if decided > 0 else 0.0
    total_pnl  = round(resolved["pnl_usd"].sum(), 2) if not resolved.empty else 0.0
    avg_return = round(resolved["pnl_usd"].mean(), 2) if not resolved.empty else 0.0

    # Effective return = direction-adjusted return (positive = profitable regardless of direction)
    def _effective_return(row):
        ret = row.get("return_pct", 0) or 0
        td = str(row.get("trade_direction", ""))
        if td in ("Bearish", "SHORT"):
            return -ret  # SHORT profits when price falls
        return ret

    if not resolved.empty:
        resolved = resolved.copy()
        resolved["effective_return"] = resolved.apply(_effective_return, axis=1)
        avg_win  = round(wins_df.apply(_effective_return, axis=1).mean(), 4) if not wins_df.empty else 0.0
        avg_loss = round(losses_df.apply(_effective_return, axis=1).mean(), 4) if not losses_df.empty else 0.0
    else:
        avg_win = avg_loss = 0.0
    best_pnl   = round(resolved["pnl_usd"].max(), 2) if not resolved.empty else 0.0
    worst_pnl  = round(resolved["pnl_usd"].min(), 2) if not resolved.empty else 0.0

    # Profit factor
    gross_profit = wins_df["pnl_usd"].sum() if not wins_df.empty else 0.0
    gross_loss   = abs(losses_df["pnl_usd"].sum()) if not losses_df.empty else 0.0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Risk metrics — use pnl_usd for meaningful dollar-based risk metrics
    sharpe = sortino = var_95 = None
    if not resolved.empty and len(resolved) > 2:
        pnl_series = resolved["pnl_usd"].dropna()
        if len(pnl_series) >= 5 and pnl_series.std() > 0:
            # Normalise to % of average entry price for annualisation
            avg_entry = resolved["entry_price"].dropna().mean() or 1.0
            pnl_pct = pnl_series / (avg_entry * 1000) * 100  # % return per trade
            daily_rf = 0.05 / 252
            excess = pnl_pct - daily_rf
            if excess.std() > 0:
                sharpe = round(float(excess.mean() / excess.std() * (252 ** 0.5)), 2)
            downside = excess[excess < 0]
            if len(downside) > 1 and downside.std() > 0:
                sortino = round(float(excess.mean() / downside.std() * (252 ** 0.5)), 2)
        # 95% VaR on pnl_usd
        if len(pnl_series) >= 5:
            var_95 = round(float(np.percentile(pnl_series, 5)), 2)

    # Avg trades per month
    avg_per_month = 0.0
    if not resolved.empty and "prediction_date" in resolved.columns:
        months = resolved["prediction_date"].astype(str).str[:7].nunique()
        avg_per_month = round(len(resolved) / max(months, 1), 1)

    return TradingMetrics(
        total_predictions=total,
        total_trades=len(trades),
        wins=win_count,
        losses=loss_count,
        cut_losses=cut_loss_count,
        neutrals=len(neutrals_df),
        holds=len(holds),
        pending=len(pending),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_return_pct=avg_return,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        best_trade_pnl=best_pnl,
        worst_trade_pnl=worst_pnl,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        profit_factor=profit_factor,
        value_at_risk_95=var_95 or 0.0,
        avg_trades_per_month=avg_per_month,
        note=(
            "For fundamental analysts. Only HIGH-signal trades counted. "
            "Win rate = wins / (wins + losses + cut-losses). "
            "Cut-loss = adverse move > 2%. Hold = MEDIUM/LOW signal or low confidence."
        ),
    )
