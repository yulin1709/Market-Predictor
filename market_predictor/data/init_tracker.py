"""
data/init_tracker.py — Create and migrate the predictions table in articles.db.

Run once after pulling new code:
    python market_predictor/data/init_tracker.py

Safe to run multiple times — uses ALTER TABLE with error handling so existing
data is never lost.
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")

NEW_COLUMNS = [
    ("entry_price",        "REAL"),
    ("exit_price",         "REAL"),
    ("price_currency",     "TEXT DEFAULT 'USD'"),
    ("price_unit",         "TEXT"),
    ("ticker_yf",          "TEXT"),
    ("entry_price_source", "TEXT"),
    ("exit_price_source",  "TEXT"),
    ("data_quality",       "TEXT"),
    ("trade_status",       "TEXT DEFAULT 'HOLD'"),
    ("trade_number",       "INTEGER"),
    ("event_type",         "TEXT"),
    ("cut_loss_price",     "REAL"),
    ("impact_score",       "INTEGER"),
]


def init_tracker(conn: sqlite3.Connection) -> None:
    """
    Create the predictions table if it doesn't exist, and add any missing
    price columns to an existing table via ALTER TABLE migration.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction_date     DATE NOT NULL,
            commodity           TEXT NOT NULL,
            ticker              TEXT NOT NULL,
            ticker_yf           TEXT,
            signal              TEXT NOT NULL,
            direction           TEXT NOT NULL,
            confidence          REAL NOT NULL,
            expected_move       TEXT NOT NULL,
            headline            TEXT,
            entry_price         REAL,
            entry_price_source  TEXT,
            exit_price          REAL,
            exit_price_source   TEXT,
            price_currency      TEXT DEFAULT 'USD',
            price_unit          TEXT,
            actual_move         REAL,
            outcome             TEXT,
            pnl_usd             REAL,
            data_quality        TEXT,
            UNIQUE(prediction_date, commodity)
        )
    """)
    conn.commit()

    # Migrate existing tables that lack the new price columns
    for col_name, col_type in NEW_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            conn.commit()
            print(f"[init_tracker] Added column: {col_name}")
        except sqlite3.OperationalError:
            pass  # Column already exists — safe to ignore

    print("[init_tracker] predictions table ready.")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    init_tracker(conn)
    conn.close()
