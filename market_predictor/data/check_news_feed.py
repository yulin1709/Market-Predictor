"""
data/check_news_feed.py — Daily news feed health check.

Checks whether today's article fetch was successful.
Exit code 1 if unhealthy (stops downstream pipeline scripts).
Exit code 0 if healthy.

Usage in daily pipeline (run_pipeline.bat):
    python collect_news.py
    if %errorlevel% neq 0 goto :end
    python check_news_feed.py
    if %errorlevel% neq 0 goto :end
    python align_and_label.py
    ...
    :end
    echo Pipeline stopped due to insufficient news data.

NOTE: Windows Task Scheduler runs each script as a separate task and does NOT
check exit codes between tasks. Use a wrapper .bat file (run_pipeline.bat) to
chain scripts with exit code checking, then schedule only the .bat file.
"""
import os
import sqlite3
import sys
from datetime import date

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
MIN_ARTICLES = 5     # below this → pipeline should not run
WARN_ARTICLES = 20   # below this → warn but allow pipeline to continue


def check() -> None:
    today = date.today().strftime("%Y-%m-%d")

    if not os.path.exists(DB_PATH):
        print(f"[news_check] CRITICAL: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    count = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE date(published_at) = ?", (today,)
    ).fetchone()[0]

    latest = conn.execute(
        "SELECT MAX(published_at) FROM articles"
    ).fetchone()[0]

    conn.close()

    print(f"[news_check] Today ({today}): {count} articles fetched")
    print(f"[news_check] Most recent article: {latest}")

    if count == 0:
        print(
            f"[news_check] CRITICAL: 0 articles today — S&P Platts feed may be down "
            f"or token expired.\n"
            f"[news_check] Run: python market_predictor/data/collect_news.py\n"
            f"[news_check] Check: market_predictor/data/auth.py — token may need refresh"
        )
        sys.exit(1)

    if count < MIN_ARTICLES:
        print(
            f"[news_check] ERROR: Only {count} articles — below minimum {MIN_ARTICLES} "
            f"needed for reliable signals. Pipeline will not run."
        )
        sys.exit(1)

    if count < WARN_ARTICLES:
        print(
            f"[news_check] WARNING: Only {count} articles today — signals will have "
            f"limited confidence. Proceeding with caution."
        )
        sys.exit(0)

    print(f"[news_check] OK: {count} articles — sufficient for signal generation.")
    sys.exit(0)


if __name__ == "__main__":
    check()
