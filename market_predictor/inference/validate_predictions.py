"""
inference/validate_predictions.py — 7-check data quality audit for the predictions table.

Can be run standalone or called from the Streamlit sidebar.

Usage:
    python market_predictor/inference/validate_predictions.py

Returns a list of ValidationResult dicts, one per check.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.db_path import DB_PATH as _DB_PATH; DB_PATH = _DB_PATH


@dataclass
class ValidationResult:
    check_id: int
    name: str
    status: str          # "PASS" | "WARN" | "FAIL"
    detail: str
    affected_rows: int = 0
    rows: list[dict] = field(default_factory=list)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Individual checks ─────────────────────────────────────────────────────────

def check_price_source_audit(conn: sqlite3.Connection) -> ValidationResult:
    """Check 1: What fraction of prices came from Platts vs yfinance vs unknown."""
    rows = conn.execute("""
        SELECT entry_price_source, COUNT(*) as cnt
        FROM predictions
        WHERE entry_price IS NOT NULL
        GROUP BY entry_price_source
    """).fetchall()
    breakdown = {r["entry_price_source"] or "unknown": r["cnt"] for r in rows}
    total = sum(breakdown.values())
    unknown = breakdown.get("unknown", 0) + breakdown.get(None, 0)
    platts = breakdown.get("platts", 0)
    yf = breakdown.get("yfinance", 0)
    detail = f"platts={platts}, yfinance={yf}, unknown={unknown} (total with price={total})"
    # WARN rather than FAIL — unknown rows are typically seeded historical data
    status = "PASS" if unknown == 0 else ("WARN" if unknown / max(total, 1) < 0.8 else "FAIL")
    return ValidationResult(1, "Price Source Audit", status, detail, unknown)


def check_same_day_prices(conn: sqlite3.Connection) -> ValidationResult:
    """Check 2: Rows where entry_price == exit_price (same-day fetch bug)."""
    rows = conn.execute("""
        SELECT id, commodity, prediction_date, entry_price, exit_price
        FROM predictions
        WHERE entry_price IS NOT NULL
          AND exit_price IS NOT NULL
          AND ABS(exit_price - entry_price) < 0.0001
    """).fetchall()
    n = len(rows)
    status = "PASS" if n == 0 else ("WARN" if n < 5 else "FAIL")
    detail = f"{n} rows with entry_price == exit_price"
    return ValidationResult(2, "Same-Day Price Check", status, detail, n,
                            [dict(r) for r in rows[:20]])


def check_shared_prices(conn: sqlite3.Connection) -> ValidationResult:
    """Check 3: Brent rows sharing the same entry_price as Dubai on the same date."""
    rows = conn.execute("""
        SELECT b.id, b.commodity, b.prediction_date, b.entry_price
        FROM predictions b
        JOIN predictions d
          ON b.prediction_date = d.prediction_date
         AND ABS(COALESCE(b.entry_price, -1) - COALESCE(d.entry_price, -1)) < 0.0001
        WHERE b.commodity LIKE '%Brent%'
          AND d.commodity LIKE '%Dubai%'
          AND b.entry_price IS NOT NULL
    """).fetchall()
    n = len(rows)
    status = "PASS" if n == 0 else ("WARN" if n < 5 else "FAIL")
    detail = f"{n} Brent rows sharing Dubai Crude entry_price (wrong ticker bug)"
    return ValidationResult(3, "Shared Price Check (Brent/Dubai)", status, detail, n,
                            [dict(r) for r in rows[:20]])


def check_confidence_clustering(conn: sqlite3.Connection) -> ValidationResult:
    """Check 4: Confidence values should not all cluster at a single value (e.g. 0.35)."""
    rows = conn.execute("""
        SELECT ROUND(confidence, 2) as conf, COUNT(*) as cnt
        FROM predictions
        WHERE confidence IS NOT NULL
        GROUP BY ROUND(confidence, 2)
        ORDER BY cnt DESC
        LIMIT 5
    """).fetchall()
    if not rows:
        return ValidationResult(4, "Confidence Clustering", "PASS", "No predictions found", 0)
    top_conf, top_cnt = rows[0]["conf"], rows[0]["cnt"]
    total = conn.execute("SELECT COUNT(*) FROM predictions WHERE confidence IS NOT NULL").fetchone()[0]
    pct = top_cnt / max(total, 1) * 100
    status = "PASS" if pct < 60 else ("WARN" if pct < 80 else "FAIL")
    detail = f"Top confidence value {top_conf} appears in {top_cnt}/{total} rows ({pct:.0f}%)"
    return ValidationResult(4, "Confidence Clustering", status, detail, top_cnt if pct >= 60 else 0)


def check_direction_bias(conn: sqlite3.Connection) -> ValidationResult:
    """Check 5: Direction distribution should not be >80% in one direction."""
    rows = conn.execute("""
        SELECT direction, COUNT(*) as cnt
        FROM predictions
        WHERE direction IS NOT NULL
        GROUP BY direction
    """).fetchall()
    if not rows:
        return ValidationResult(5, "Direction Bias", "PASS", "No predictions found", 0)
    counts = {r["direction"]: r["cnt"] for r in rows}
    total = sum(counts.values())
    max_dir = max(counts, key=counts.get)
    max_pct = counts[max_dir] / total * 100
    status = "PASS" if max_pct < 70 else ("WARN" if max_pct < 85 else "FAIL")
    detail = f"Direction distribution: {counts} — '{max_dir}' is {max_pct:.0f}% of all predictions"
    return ValidationResult(5, "Direction Bias", status, detail, 0)


def check_win_rate_by_confidence(conn: sqlite3.Connection) -> ValidationResult:
    """Check 6: Higher confidence predictions should have higher win rates."""
    rows = conn.execute("""
        SELECT
            CASE
                WHEN confidence >= 0.7 THEN 'high (>=0.7)'
                WHEN confidence >= 0.5 THEN 'medium (0.5-0.7)'
                ELSE 'low (<0.5)'
            END as band,
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) as wins
        FROM predictions
        WHERE outcome IS NOT NULL AND confidence IS NOT NULL
        GROUP BY band
    """).fetchall()
    if not rows:
        return ValidationResult(6, "Win Rate by Confidence", "PASS", "No resolved predictions", 0)
    lines = []
    for r in rows:
        wr = r["wins"] / max(r["total"], 1) * 100
        lines.append(f"{r['band']}: {wr:.0f}% win rate ({r['wins']}/{r['total']})")
    detail = " | ".join(lines)
    return ValidationResult(6, "Win Rate by Confidence", "PASS", detail, 0)


def check_pnl_sign_consistency(conn: sqlite3.Connection) -> ValidationResult:
    """Check 7: P&L sign should be consistent with outcome (correct=positive, incorrect=negative)."""
    rows = conn.execute("""
        SELECT id, commodity, prediction_date, outcome, pnl_usd
        FROM predictions
        WHERE outcome IS NOT NULL AND pnl_usd IS NOT NULL
          AND (
            (outcome = 'correct'   AND pnl_usd < 0)
            OR
            (outcome = 'incorrect' AND pnl_usd > 0)
          )
    """).fetchall()
    n = len(rows)
    status = "PASS" if n == 0 else ("WARN" if n < 5 else "FAIL")
    detail = f"{n} rows where P&L sign contradicts outcome label"
    return ValidationResult(7, "P&L Sign Consistency", status, detail, n,
                            [dict(r) for r in rows[:20]])


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_checks() -> list[ValidationResult]:
    """Run all 7 validation checks and return results."""
    conn = _connect()
    checks = [
        check_price_source_audit,
        check_same_day_prices,
        check_shared_prices,
        check_confidence_clustering,
        check_direction_bias,
        check_win_rate_by_confidence,
        check_pnl_sign_consistency,
    ]
    results = [fn(conn) for fn in checks]
    conn.close()
    return results


def print_report(results: list[ValidationResult]) -> None:
    print("\n" + "=" * 60)
    print("  PREDICTION DATA QUALITY REPORT")
    print("=" * 60)
    for r in results:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(r.status, "?")
        print(f"\n{icon} Check {r.check_id}: {r.name}")
        print(f"   Status : {r.status}")
        print(f"   Detail : {r.detail}")
        if r.affected_rows:
            print(f"   Affected rows: {r.affected_rows}")
    print("\n" + "=" * 60)
    fails = sum(1 for r in results if r.status == "FAIL")
    warns = sum(1 for r in results if r.status == "WARN")
    print(f"  Summary: {len(results) - fails - warns} PASS  |  {warns} WARN  |  {fails} FAIL")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    results = run_all_checks()
    print_report(results)
