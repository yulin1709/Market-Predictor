"""
data/backfill_signals.py — Generate historical predictions for past dates that have articles.

For each date with ≥5 articles that doesn't already have predictions,
runs the model and logs signals — then backfill_actuals.py fills the prices.

Usage:
    python market_predictor/data/backfill_signals.py
    python market_predictor/data/backfill_signals.py --days 60   # last 60 days only
    python market_predictor/data/backfill_signals.py --from 2026-01-01
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
MIN_ARTICLES = 5


def get_dates_needing_signals(conn: sqlite3.Connection, from_date: str, to_date: str) -> list[str]:
    """Return dates that have ≥MIN_ARTICLES articles but no predictions yet."""
    existing = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT prediction_date FROM predictions"
        ).fetchall()
    }
    rows = conn.execute("""
        SELECT date(published_at), COUNT(*) as n
        FROM articles
        WHERE date(published_at) BETWEEN ? AND ?
        GROUP BY date(published_at)
        HAVING n >= ?
        ORDER BY date(published_at) ASC
    """, (from_date, to_date, MIN_ARTICLES)).fetchall()

    return [r[0] for r in rows if r[0] not in existing]


def generate_signals_for_date(target_date: str, conn: sqlite3.Connection) -> int:
    """
    Load articles for target_date, run the model, log predictions.
    Returns number of predictions logged.
    """
    from inference.pipeline import predict, log_prediction, _load_models
    from features.extract_entities import get_event_type, get_impact_score
    from data.ticker_config import COMMODITY_CONFIG

    _load_models()

    COMMODITIES = [
        ("Dubai Crude (PCAAT00)", "PCAAT00"),
        ("Brent Dated (PCAAS00)", "PCAAS00"),
        ("WTI (PCACG00)",         "PCACG00"),
        ("LNG / Nat Gas (JKM)",   "JKM"),
    ]

    COMMODITY_KEYWORDS = {
        "Dubai Crude (PCAAT00)": ["dubai","oman","saudi","opec","middle east","gulf","crude","iran","iraq","uae","kuwait","qatar","barrel"],
        "Brent Dated (PCAAS00)": ["brent","north sea","europe","russia","urals","norway","uk","crude oil","barrel"],
        "WTI (PCACG00)":         ["wti","cushing","texas","shale","permian","us crude","american","gulf of mexico","refinery"],
        "LNG / Nat Gas (JKM)":   ["lng","natural gas","liquefied","tanker","cargo","freight","jkm","asia","pacific","shipping","vessel","hormuz"],
    }

    # Load articles for this date
    articles = conn.execute("""
        SELECT id, headline, body_text
        FROM articles
        WHERE date(published_at) = ?
        ORDER BY published_at ASC
    """, (target_date,)).fetchall()

    if len(articles) < MIN_ARTICLES:
        return 0

    # Score all articles
    scored = []
    for art_id, headline, body_text in articles:
        text = (str(headline or "") + " " + str(body_text or "")[:400]).strip()
        if not text:
            continue
        try:
            r = predict(text, skip_explanation=True)
            scored.append({"id": art_id, "headline": headline or "", "text": text, **r})
        except Exception:
            pass

    if not scored:
        return 0

    logged = 0
    for comm, ticker in COMMODITIES:
        kws = COMMODITY_KEYWORDS[comm]
        rel = [a for a in scored if sum(1 for k in kws if k in a.get("headline","").lower()) >= 1]
        pool = rel if rel else scored

        # Re-score with commodity model
        re_scored = []
        for a in pool:
            try:
                r = predict(a["text"], skip_explanation=True, commodity=comm)
                re_scored.append({**a, **r})
            except Exception:
                re_scored.append(a)

        if not re_scored:
            continue

        # Aggregate signal
        conf = sum(a.get("high_prob", 0) for a in re_scored) / len(re_scored)
        top = sorted(re_scored, key=lambda x: x.get("high_prob", 0), reverse=True)[0]

        label = "HIGH" if conf >= 0.30 else ("MEDIUM" if conf >= 0.15 else "LOW")

        # Direction from directional model votes
        from collections import Counter
        dir_votes = [a.get("direction_from_model") for a in re_scored if a.get("direction_from_model")]
        if dir_votes:
            direction = Counter(dir_votes).most_common(1)[0][0]
            _dmap = {"rise": "bullish", "fall": "bearish", "flat": "neutral"}
            direction = _dmap.get(direction, direction)
        else:
            direction = "neutral"

        # Event type and impact
        event_type = get_event_type(top.get("headline", ""))
        impact_score = get_impact_score(event_type)

        # Scale confidence
        label_factor = 1.4 if label == "HIGH" else (1.2 if label == "MEDIUM" else 1.0)
        dir_factor = 1.0 if direction != "neutral" else 0.6
        n_rel = len(rel)
        article_factor = min(n_rel / 15.0, 1.0)
        scaled_conf = min(max(conf * 4.5 * label_factor * dir_factor * (0.7 + article_factor * 0.3), 35), 95)

        sig = {
            "commodity": comm, "ticker": ticker,
            "signal": label, "direction": direction,
            "scaled_conf": scaled_conf,
            "high_prob": conf,
            "move": ">3%" if label == "HIGH" else ("1-3%" if label == "MEDIUM" else "<1%"),
            "top_headline": top.get("headline", ""),
            "event_type": event_type,
            "impact_score": impact_score,
            "entities": {"event_type": event_type, "impact_score": impact_score},
        }

        try:
            # Temporarily override today's date for log_prediction
            import inference.pipeline as _pip
            _orig_date = None
            # Patch: write directly to avoid date override issues
            _log_historical(conn, target_date, comm, ticker, sig, top.get("headline",""), len(articles))
            logged += 1
        except Exception as e:
            print(f"    ERROR logging {comm}: {e}")

    return logged


def _log_historical(conn: sqlite3.Connection, pred_date: str, commodity: str,
                    ticker: str, sig: dict, headline: str, article_count: int) -> None:
    """Write a historical prediction directly to DB."""
    from data.ticker_config import COMMODITY_CONFIG, STOP_LOSS_PCT
    from inference.trade_filter import HIGH_CONFIDENCE_THRESHOLD

    cfg = COMMODITY_CONFIG.get(commodity, {})
    ticker_yf = cfg.get("ticker_yf", "")
    price_unit = cfg.get("unit", "barrel")

    signal = sig["signal"]
    direction = sig["direction"]
    confidence = sig.get("high_prob", sig.get("scaled_conf", 35) / 100)
    if confidence > 1.0:
        confidence = confidence / 100.0
    expected_move = sig.get("move", "<1%")
    event_type = sig.get("event_type", "other")
    impact_score = sig.get("impact_score", 2)

    # Trade status
    trade_status = "TRADE" if (
        signal == "HIGH" and confidence >= HIGH_CONFIDENCE_THRESHOLD
    ) else "HOLD"

    # Ensure columns exist
    for col, col_type in [
        ("entry_price","REAL"), ("exit_price","REAL"), ("price_currency","TEXT"),
        ("price_unit","TEXT"), ("ticker_yf","TEXT"), ("data_quality","TEXT"),
        ("entry_price_source","TEXT"), ("exit_price_source","TEXT"),
        ("trade_status","TEXT"), ("trade_number","INTEGER"),
        ("event_type","TEXT"), ("cut_loss_price","REAL"), ("impact_score","INTEGER"),
        ("trade_direction","TEXT"), ("return_pct","REAL"), ("trade_outcome","TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
        except Exception:
            pass

    conn.execute("""
        INSERT OR IGNORE INTO predictions
            (prediction_date, commodity, ticker, ticker_yf, signal, direction,
             confidence, expected_move, headline, price_unit, data_quality,
             trade_status, event_type, impact_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'backfilled', ?, ?, ?)
    """, (
        pred_date, commodity, ticker, ticker_yf, signal, direction,
        confidence, expected_move, headline[:200], price_unit,
        trade_status, event_type, impact_score,
    ))
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, help="Look back N days (default 90)")
    parser.add_argument("--from", dest="from_date", type=str, default=None)
    args = parser.parse_args()

    to_date = date.today().isoformat()
    if args.from_date:
        from_date = args.from_date
    else:
        from_date = (date.today() - timedelta(days=args.days)).isoformat()

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    dates = get_dates_needing_signals(conn, from_date, to_date)
    print(f"[backfill_signals] {len(dates)} dates need signals ({from_date} → {to_date})")

    total = 0
    for d in dates:
        n = generate_signals_for_date(d, conn)
        if n > 0:
            print(f"  {d}: logged {n} signals")
            total += n

    conn.close()
    print(f"\n[backfill_signals] Done. {total} signals logged.")
    print("Now run: python market_predictor/data/backfill_actuals.py")


if __name__ == "__main__":
    main()
