"""
features/extract_entities.py — Extract structured signals from article headlines.

NO external API calls. NO LLM. Runs completely offline.

Uses:
  - Keyword rules for event_type and region
  - TextBlob for sentiment polarity
  - Keyword rules for supply_impact
  - IMPACT_SCORES from ticker_config for fundamental impact scoring

Reads from:  labelled_articles (SQLite)
Writes to:   entities (SQLite)

Usage:
    python features/extract_entities.py
"""
import os
import sqlite3
import sys
from datetime import datetime, timezone

from textblob import TextBlob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")


# ── Keyword lists ────────────────────────────────────────────────────────────

EVENT_RULES = [
    ("opec_decision",        ["opec", "opec+", "quota", "production cut", "output cut", "opec meeting"]),
    ("sanctions",            ["sanction", "embargo", "ban on", "restrict", "blocked"]),
    ("force_majeure",        ["force majeure", "act of god", "unforeseeable"]),
    ("pipeline_outage",      ["pipeline outage", "pipeline shutdown", "pipeline disruption", "pipeline explosion"]),
    ("supply_disruption",    ["outage", "shutdown", "disruption", "explosion", "fire at", "refinery outage"]),
    ("geopolitical",         ["war", "conflict", "tension", "attack", "military", "geopolit", "invasion", "strike"]),
    ("demand_shock",         ["demand surge", "demand spike", "demand collapse", "demand shock"]),
    ("lng_outage",           ["lng outage", "lng plant", "lng terminal", "lng disruption", "lng shutdown"]),
    ("inventory_data",       ["eia", "api inventory", "crude draw", "crude build", "inventory data", "stock data"]),
    ("production_change",    ["production change", "output change", "rig count", "shale output"]),
    ("refinery_maintenance", ["maintenance", "repair", "scheduled", "turnaround", "refinery"]),
    ("shipping_disruption",  ["tanker", "ship", "vessel", "cargo", "freight", "strait", "hormuz", "suez"]),
    ("demand_change",        ["demand", "consumption", "import", "export", "forecast"]),
]

REGION_RULES = [
    ("europe",       ["europe", "european", "germany", "uk", "france", "netherlands", "italy", "norway"]),
    ("middle_east",  ["middle east", "saudi", "iran", "iraq", "uae", "kuwait", "qatar", "oman", "gulf"]),
    ("russia",       ["russia", "russian", "moscow"]),
    ("asia",         ["china", "japan", "korea", "asia", "india", "singapore", "malaysia"]),
    ("us",           ["us", "usa", "american", "texas", "houston", "gulf of mexico"]),
]

SUPPLY_IMPACT_WORDS = [
    "outage", "shutdown", "disruption", "sanction", "embargo", "cut",
    "maintenance", "repair", "explosion", "attack", "pipeline", "refinery", "tanker",
]

# Impact scores for fundamental traders (0-10)
IMPACT_SCORES: dict[str, int] = {
    "opec_decision":        10,
    "sanctions":             9,
    "force_majeure":         9,
    "pipeline_outage":       8,
    "supply_disruption":     8,
    "geopolitical":          7,
    "demand_shock":          7,
    "lng_outage":            7,
    "inventory_data":        6,
    "production_change":     6,
    "refinery_maintenance":  5,
    "shipping_disruption":   5,
    "demand_change":         4,
    "analyst_comment":       3,
    "routine_report":        2,
    "price_movement":        1,
    "general_market":        1,
    "other":                 2,
}


# ── Feature extraction ───────────────────────────────────────────────────────

def get_event_type(headline: str) -> str:
    text = headline.lower()
    for event_type, keywords in EVENT_RULES:
        if any(kw in text for kw in keywords):
            return event_type
    return "other"


def get_impact_score(event_type: str) -> int:
    """Return the fundamental impact score (0-10) for an event type."""
    return IMPACT_SCORES.get(event_type, 2)


def get_region(headline: str) -> str:
    text = headline.lower()
    for region, keywords in REGION_RULES:
        if any(kw in text for kw in keywords):
            return region
    return "global"


def get_sentiment(headline: str) -> float:
    return round(TextBlob(headline).sentiment.polarity, 4)


def get_supply_impact(headline: str) -> bool:
    text = headline.lower()
    return any(kw in text for kw in SUPPLY_IMPACT_WORDS)


def extract_entities(headline: str) -> dict:
    """Extract all entities from a headline. Returns a dict."""
    event_type = get_event_type(headline)
    return {
        "event_type":    event_type,
        "region":        get_region(headline),
        "sentiment":     get_sentiment(headline),
        "supply_impact": get_supply_impact(headline),
        "impact_score":  get_impact_score(event_type),
    }


# ── Database ─────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            article_id    TEXT PRIMARY KEY,
            event_type    TEXT,
            region        TEXT,
            sentiment     REAL,
            supply_impact INTEGER,
            impact_score  INTEGER DEFAULT 2,
            extracted_at  TEXT
        )
    """)
    # Add impact_score column if missing
    try:
        conn.execute("ALTER TABLE entities ADD COLUMN impact_score INTEGER DEFAULT 2")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def get_unprocessed(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute("""
        SELECT la.id, la.headline
        FROM labelled_articles la
        LEFT JOIN entities e ON la.id = e.article_id
        WHERE e.article_id IS NULL
    """).fetchall()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    rows = get_unprocessed(conn)
    total = len(rows)
    print(f"[entities] {total} articles to process (no API calls — offline only).")

    if total == 0:
        print("[entities] All articles already processed.")
        conn.close()
        return

    now = datetime.now(timezone.utc).isoformat()
    processed = 0

    for article_id, headline in rows:
        headline = headline or ""
        event_type = get_event_type(headline)
        impact_score = get_impact_score(event_type)

        conn.execute(
            """INSERT OR REPLACE INTO entities
               (article_id, event_type, region, sentiment, supply_impact, impact_score, extracted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                article_id,
                event_type,
                get_region(headline),
                get_sentiment(headline),
                int(get_supply_impact(headline)),
                impact_score,
                now,
            ),
        )
        processed += 1
        if processed % 100 == 0:
            conn.commit()
            print(f"  Progress: {processed}/{total}")

    conn.commit()
    print(f"\n[entities] Done. {processed} articles extracted (0 API calls).")
    conn.close()


if __name__ == "__main__":
    main()
