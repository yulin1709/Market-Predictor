"""
data/align_and_label.py — Join articles to prices and assign impact labels.

COMMODITY-SPECIFIC LABELLING (item 3):
  Each article is matched to the price of the commodity it is ABOUT,
  not always Dubai Crude. An LNG article is labelled by NG price change;
  a Brent article by Brent price change; etc.

  Commodity detection uses keyword matching on headline + body_text.
  Falls back to Dubai (primary Petronas benchmark) if no match.

  Commodity → prices.symbol mapping:
    dubai/gulf/opec/middle east → "Dubai"
    brent/north sea/europe      → "Brent"
    wti/us crude/cushing        → "WTI"
    lng/natural gas/jkm         → "Oman"  (closest available proxy)
    default                     → "Dubai"

Label logic — ROLLING PERCENTILE (regime-aware):
  Uses the 33rd and 67th percentile of |price_change| over a rolling 90-day
  window per symbol. Keeps labels balanced (~33% each class) regardless of
  market regime.

  Outlier cap: |price_change| > 20% capped before labelling.
  Fallback to fixed thresholds if < 30 observations in window.

Usage:
    python data/align_and_label.py
    python data/align_and_label.py --fixed-thresholds
    python data/align_and_label.py --no-commodity-routing   # old behaviour (Dubai only)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")

HIGH_THRESHOLD_FIXED  = 3.0
MEDIUM_THRESHOLD_FIXED = 1.0
MIN_ROLLING_OBS = 30
ROLLING_WINDOW_DAYS = 90
OUTLIER_CAP_PCT = 20.0

# Commodity keyword → prices.symbol mapping
# Keywords are checked against lowercase(headline + " " + body_text[:200])
COMMODITY_ROUTING: list[tuple[list[str], str]] = [
    # LNG / Natural Gas — check first (specific terms)
    (["lng", "natural gas", "liquefied", "jkm", "henry hub", "ttf", "nbp",
      "regasification", "gas price", "spot gas", "lng terminal"], "Oman"),
    # WTI / US crude
    (["wti", "west texas", "cushing", "shale", "permian", "eagle ford",
      "bakken", "us crude", "gulf coast", "usgc", "houston", "midland"], "WTI"),
    # Brent / North Sea / Europe
    (["brent", "north sea", "dated brent", "forties", "oseberg", "ekofisk",
      "europe", "urals", "russia", "norway", "uk crude", "rotterdam",
      "mediterranean"], "Brent"),
    # Dubai / Gulf / OPEC — broadest, check last
    (["dubai", "oman", "saudi", "abu dhabi", "kuwait", "iraq", "iran",
      "middle east", "gulf", "opec", "arab", "murban", "basrah",
      "arabian", "sour crude"], "Dubai"),
]
DEFAULT_SYMBOL = "Dubai"


def _detect_symbol(headline: str, body_text: str) -> str:
    """Return the prices.symbol most relevant to this article."""
    text = (headline + " " + (body_text or "")[:200]).lower()
    for keywords, symbol in COMMODITY_ROUTING:
        if any(kw in text for kw in keywords):
            return symbol
    return DEFAULT_SYMBOL


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labelled_articles (
            id              TEXT PRIMARY KEY,
            headline        TEXT,
            body_text       TEXT,
            source          TEXT,
            published_at    TEXT,
            url             TEXT,
            aligned_date    TEXT,
            price_change    REAL,
            label           TEXT,
            price_symbol    TEXT
        )
    """)
    # Add price_symbol column if missing (migration)
    try:
        conn.execute("ALTER TABLE labelled_articles ADD COLUMN price_symbol TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _build_rolling_thresholds(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 33rd/67th percentile thresholds per date."""
    prices_df = prices_df.copy()
    prices_df["date_dt"] = pd.to_datetime(prices_df["date"])
    prices_df = prices_df.sort_values("date_dt").reset_index(drop=True)
    prices_df["abs_change"] = prices_df["pct_change_24h"].abs().clip(upper=OUTLIER_CAP_PCT)

    thresh_low_list, thresh_high_list = [], []
    for i, row in prices_df.iterrows():
        cutoff = row["date_dt"] - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
        window = prices_df[
            (prices_df["date_dt"] <= row["date_dt"]) &
            (prices_df["date_dt"] >= cutoff)
        ]["abs_change"].dropna()

        if len(window) >= MIN_ROLLING_OBS:
            t_low  = max(float(np.percentile(window, 33)), 0.3)
            t_high = max(float(np.percentile(window, 67)), t_low + 0.3)
        else:
            t_low, t_high = MEDIUM_THRESHOLD_FIXED, HIGH_THRESHOLD_FIXED

        thresh_low_list.append(t_low)
        thresh_high_list.append(t_high)

    prices_df["thresh_low"]  = thresh_low_list
    prices_df["thresh_high"] = thresh_high_list
    return prices_df


def label_from_pct_rolling(pct: float, thresh_low: float, thresh_high: float) -> str:
    abs_pct = min(abs(pct), OUTLIER_CAP_PCT)
    if abs_pct >= thresh_high:
        return "HIGH"
    if abs_pct >= thresh_low:
        return "MEDIUM"
    return "LOW"


def label_from_pct_fixed(pct: float) -> str:
    if abs(pct) >= HIGH_THRESHOLD_FIXED:
        return "HIGH"
    if abs(pct) >= MEDIUM_THRESHOLD_FIXED:
        return "MEDIUM"
    return "LOW"


def next_trading_date(dt: datetime, trading_dates: set, max_gap_days: int = 3) -> str | None:
    candidate = dt.date()
    for offset in range(max_gap_days + 1):
        s = candidate.isoformat()
        if s in trading_dates:
            return s
        candidate += timedelta(days=1)
    return None


def main(fixed_thresholds: bool = False, no_commodity_routing: bool = False) -> None:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    price_columns = table_columns(conn, "prices")
    if not {"date", "symbol", "pct_change_24h"}.issubset(price_columns):
        print("[label] ERROR: prices table missing columns. Run collect_prices.py first.")
        conn.close()
        return

    # Load ALL available price symbols
    available_symbols = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM prices WHERE pct_change_24h IS NOT NULL"
        ).fetchall()
    }
    print(f"[label] Available price symbols: {sorted(available_symbols)}")

    # Build per-symbol price lookups and rolling thresholds
    symbol_price_lookup: dict[str, dict[str, float]] = {}
    symbol_thresh_lookup: dict[str, dict[str, tuple[float, float]]] = {}
    symbol_trading_dates: dict[str, set[str]] = {}

    for symbol in available_symbols:
        df_sym = pd.read_sql(
            "SELECT date, pct_change_24h FROM prices WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,)
        )
        if df_sym.empty:
            continue
        if not fixed_thresholds:
            df_sym = _build_rolling_thresholds(df_sym)
            symbol_thresh_lookup[symbol] = {
                row["date"]: (row["thresh_low"], row["thresh_high"])
                for _, row in df_sym.iterrows()
            }
        symbol_price_lookup[symbol] = dict(zip(df_sym["date"], df_sym["pct_change_24h"]))
        symbol_trading_dates[symbol] = set(df_sym["date"].tolist())
        print(f"[label]   {symbol}: {len(df_sym)} price rows")

    if not symbol_price_lookup:
        print("[label] ERROR: No price data found. Run collect_prices.py first.")
        conn.close()
        return

    # All trading dates (union across symbols) for date alignment
    all_trading_dates = set().union(*symbol_trading_dates.values())
    max_gap_days = 3 if len(all_trading_dates) >= 5 else 0

    # Load articles — exclude routine price report names that add noise to training
    # These are price tables, not news events. Including them teaches the model that
    # "Crude Oil Marketwire" = HIGH/LOW based on coincidental price moves, not content.
    NOISE_REPORT_NAMES = {
        "Crude Oil Marketwire", "Arab Gulf Marketscan", "Gulf Arab Marketscan",
        "European Marketscan", "Latin American Wire", "Oilgram Price Report",
        "Bunkerwire", "Solventswire", "Asia-Pacific - Arab Gulf Marketscan",
        "North American Crude and Products Scan", "European Gas Daily",
        "European Power Daily",
    }
    noise_filter = " AND ".join(
        [f"COALESCE(report_name,'') != '{n}'" for n in NOISE_REPORT_NAMES]
    )
    articles_df = pd.read_sql(
        f"SELECT id, headline, body_text, source, published_at, url FROM articles WHERE {noise_filter}",
        conn
    )
    print(f"[label] Loaded {len(articles_df)} articles")

    labelled = []
    skipped = 0
    symbol_counts: dict[str, int] = {}

    for _, row in articles_df.iterrows():
        pub_str = str(row["published_at"] or "")
        if not pub_str or pub_str == "None":
            skipped += 1
            continue
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00").replace(" ", "T"))
        except ValueError:
            skipped += 1
            continue

        aligned = next_trading_date(pub_dt, all_trading_dates, max_gap_days=max_gap_days)
        if aligned is None:
            skipped += 1
            continue

        # Commodity-specific routing
        if no_commodity_routing:
            symbol = DEFAULT_SYMBOL
        else:
            symbol = _detect_symbol(
                str(row["headline"] or ""),
                str(row["body_text"] or "")
            )
            # Fall back to Dubai if this symbol has no price on this date
            if aligned not in symbol_price_lookup.get(symbol, {}):
                symbol = DEFAULT_SYMBOL

        pct = symbol_price_lookup.get(symbol, {}).get(aligned)
        if pct is None:
            skipped += 1
            continue

        if fixed_thresholds:
            label = label_from_pct_fixed(float(pct))
        else:
            thresh_low, thresh_high = symbol_thresh_lookup.get(symbol, {}).get(
                aligned, (MEDIUM_THRESHOLD_FIXED, HIGH_THRESHOLD_FIXED)
            )
            label = label_from_pct_rolling(float(pct), thresh_low, thresh_high)

        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        labelled.append({
            "id":           row["id"],
            "headline":     row["headline"],
            "body_text":    row["body_text"],
            "source":       row["source"],
            "published_at": row["published_at"],
            "url":          row["url"],
            "aligned_date": aligned,
            "price_change": round(float(pct), 4),
            "label":        label,
            "price_symbol": symbol,
        })

    conn.executemany(
        """INSERT OR REPLACE INTO labelled_articles
           (id, headline, body_text, source, published_at, url,
            aligned_date, price_change, label, price_symbol)
           VALUES (:id, :headline, :body_text, :source, :published_at, :url,
                   :aligned_date, :price_change, :label, :price_symbol)""",
        labelled,
    )
    conn.commit()

    print(f"\n[label] Labelled {len(labelled)} articles. Skipped {skipped}.")
    print(f"[label] Symbol routing: {symbol_counts}")

    if labelled:
        df = pd.DataFrame(labelled)
        dist = df["label"].value_counts()
        print("\n[label] Label distribution:")
        for lbl, cnt in dist.items():
            print(f"  {lbl}: {cnt} ({cnt/len(df)*100:.1f}%)")

        print("\n[label] Label distribution by price symbol:")
        print(df.groupby(["price_symbol", "label"]).size().unstack(fill_value=0).to_string())

    conn.close()
    print("\n[label] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-thresholds", action="store_true",
                        help="Use fixed 1%%/3%% thresholds instead of rolling percentiles")
    parser.add_argument("--no-commodity-routing", action="store_true",
                        help="Use Dubai price for all articles (old behaviour)")
    args = parser.parse_args()
    main(fixed_thresholds=args.fixed_thresholds,
         no_commodity_routing=args.no_commodity_routing)
