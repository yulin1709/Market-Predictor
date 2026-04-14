"""
inference/evaluation.py — Pure ML classification evaluation.

Evaluates: "Did the model correctly predict the market movement direction?"

This is COMPLETELY SEPARATE from trading simulation (tracking.py).
No P&L, no LONG/SHORT, no lot sizes.

Logic:
  predicted_direction: from predictions.direction column (rise/fall/flat)
  actual_direction:    derived from (exit_price - entry_price) / entry_price
                       > +1%  → rise (+1)
                       < -1%  → fall (-1)
                       else   → flat (0)
  is_correct:          predicted == actual
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from data.db_path import DB_PATH as _DB_PATH; DB_PATH = _DB_PATH

# Threshold for classifying actual movement
MOVE_THRESHOLD = 0.01   # 1% — moves smaller than this are "flat"


def _map_direction(direction: str) -> int:
    """Map direction string to integer: rise=+1, fall=-1, flat=0."""
    if direction == "rise":
        return 1
    if direction == "fall":
        return -1
    return 0


def _classify_actual(return_pct: float | None) -> int | None:
    """Classify actual price movement into +1/0/-1."""
    if return_pct is None:
        return None
    if return_pct > MOVE_THRESHOLD * 100:
        return 1
    if return_pct < -MOVE_THRESHOLD * 100:
        return -1
    return 0


def load_evaluation_df(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load predictions and compute evaluation columns.

    Returns DataFrame with columns:
        prediction_date, commodity, signal, direction, confidence,
        entry_price, exit_price, return_pct,
        predicted_int, actual_int, actual_direction,
        is_correct, status (evaluated/pending)
    """
    df = pd.read_sql_query("""
        SELECT
            id, prediction_date, commodity, signal, direction, confidence,
            entry_price, exit_price, actual_move
        FROM predictions
        ORDER BY prediction_date DESC
    """, conn)

    if df.empty:
        return df

    # Compute return_pct from real prices where available
    df["return_pct"] = df.apply(
        lambda r: round((r["exit_price"] - r["entry_price"]) / r["entry_price"] * 100, 4)
        if (pd.notna(r["entry_price"]) and pd.notna(r["exit_price"]) and r["entry_price"] > 0)
        else (r["actual_move"] if pd.notna(r.get("actual_move")) else None),
        axis=1,
    )

    # Predicted direction as integer
    df["predicted_int"] = df["direction"].apply(
        lambda d: _map_direction(str(d)) if pd.notna(d) else 0
    )

    # Actual direction from price
    df["actual_int"] = df["return_pct"].apply(_classify_actual)

    # Human-readable actual direction
    _int_to_str = {1: "rise", -1: "fall", 0: "flat", None: None}
    df["actual_direction"] = df["actual_int"].apply(lambda x: _int_to_str.get(x))

    # Correctness
    df["is_correct"] = df.apply(
        lambda r: (r["predicted_int"] == r["actual_int"])
        if (r["actual_int"] is not None and pd.notna(r["actual_int"]))
        else None,
        axis=1,
    )

    # Status
    df["status"] = df["is_correct"].apply(
        lambda x: "evaluated" if x is not None else "pending"
    )

    return df


def compute_accuracy_metrics(df: pd.DataFrame) -> dict:
    """
    Compute classification accuracy metrics from evaluation DataFrame.

    Returns dict with:
        total_evaluated, total_pending, accuracy_pct,
        high_precision_pct, high_recall_pct,
        by_commodity (dict), by_signal (dict),
        confusion_matrix (dict of dicts)
    """
    if df.empty:
        return {
            "total_evaluated": 0, "total_pending": 0,
            "accuracy_pct": 0.0, "high_precision_pct": 0.0,
            "high_recall_pct": 0.0, "by_commodity": {}, "by_signal": {},
            "confusion_matrix": {},
        }

    evaluated = df[df["status"] == "evaluated"].copy()
    pending_count = (df["status"] == "pending").sum()

    if evaluated.empty:
        return {
            "total_evaluated": 0, "total_pending": int(pending_count),
            "accuracy_pct": 0.0, "high_precision_pct": 0.0,
            "high_recall_pct": 0.0, "by_commodity": {}, "by_signal": {},
            "confusion_matrix": {},
        }

    total_eval = len(evaluated)
    correct = evaluated["is_correct"].sum()
    accuracy = round(correct / total_eval * 100, 1) if total_eval > 0 else 0.0

    # HIGH signal precision: of all HIGH predictions, how many were correct?
    high_preds = evaluated[evaluated["signal"] == "HIGH"]
    high_precision = (
        round(high_preds["is_correct"].sum() / len(high_preds) * 100, 1)
        if len(high_preds) > 0 else 0.0
    )

    # HIGH recall: of all actual large moves, how many did HIGH signal catch?
    # Actual large move = |return_pct| > 3%
    actual_large = evaluated[evaluated["return_pct"].abs() > 3.0]
    high_caught = actual_large[actual_large["signal"] == "HIGH"]
    high_recall = (
        round(len(high_caught) / len(actual_large) * 100, 1)
        if len(actual_large) > 0 else 0.0
    )

    # Accuracy by commodity
    by_commodity = {}
    for comm, grp in evaluated.groupby("commodity"):
        n = len(grp)
        c = grp["is_correct"].sum()
        by_commodity[comm] = {"correct": int(c), "total": n, "accuracy": round(c / n * 100, 1)}

    # Accuracy by signal
    by_signal = {}
    for sig, grp in evaluated.groupby("signal"):
        n = len(grp)
        c = grp["is_correct"].sum()
        by_signal[sig] = {"correct": int(c), "total": n, "accuracy": round(c / n * 100, 1)}

    # Confusion matrix: rows=actual, cols=predicted
    labels = ["rise", "flat", "fall"]
    cm: dict[str, dict[str, int]] = {a: {p: 0 for p in labels} for a in labels}
    for _, row in evaluated.iterrows():
        actual_d = row.get("actual_direction")
        pred_d = row.get("direction")
        if actual_d in labels and pred_d in labels:
            cm[actual_d][pred_d] += 1

    return {
        "total_evaluated": total_eval,
        "total_pending": int(pending_count),
        "accuracy_pct": accuracy,
        "high_precision_pct": high_precision,
        "high_recall_pct": high_recall,
        "by_commodity": by_commodity,
        "by_signal": by_signal,
        "confusion_matrix": cm,
    }
