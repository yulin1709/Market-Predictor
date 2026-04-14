"""
data/backfill_actuals.py — Fill entry_price, exit_price, actual_move, outcome, pnl_usd
for predictions older than 24h that are still pending.

Run daily (e.g. 09:00 MYT via Task Scheduler):
    python market_predictor/data/backfill_actuals.py
"""
import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "articles.db")
DEFAULT_LOT_SIZE = 1000  # barrels or MMBtu per lot
SAME_PRICE_TOLERANCE = 0.0001
CUT_LOSS_THRESHOLD_PCT = 2.0  # 2% adverse move = cut-loss


def _ensure_data_quality_column(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN data_quality TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass


def _ensure_price_source_columns(conn: sqlite3.Connection) -> None:
    for col in ("entry_price_source", "exit_price_source", "cut_loss_price",
                "trade_status", "trade_number", "event_type", "impact_score"):
        col_type = "REAL" if col == "cut_loss_price" else ("INTEGER" if col in ("trade_number", "impact_score") else "TEXT")
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass


def _normalise_direction(direction: str) -> str:
    """Normalise direction to bullish/bearish/neutral regardless of legacy values."""
    d = (direction or "").lower().strip()
    if d in ("rise", "bullish"):
        return "bullish"
    if d in ("fall", "bearish"):
        return "bearish"
    return "neutral"


def check_cut_loss_triggered(
    entry_price: float,
    cut_loss_price: float,
    direction: str,
    exit_price: float | None = None,
) -> bool:
    """
    Returns True if price hit the thesis invalidation level during the trade window.
    Uses exit_price as proxy for intraday low/high when intraday data unavailable.
    """
    direction = _normalise_direction(direction)
    if cut_loss_price is None or entry_price is None:
        return False
    if direction == "bullish" and exit_price is not None:
        return exit_price <= cut_loss_price
    if direction == "bearish" and exit_price is not None:
        return exit_price >= cut_loss_price
    return False


def classify_outcome(signal: str, direction: str, actual_move: float,
                     entry_price: float | None = None,
                     cut_loss_price: float | None = None,
                     exit_price: float | None = None) -> str:
    """Classify outcome using direction-only logic with cut-loss detection."""
    direction = _normalise_direction(direction)

    # Check cut-loss first
    if (entry_price and cut_loss_price and exit_price and
            check_cut_loss_triggered(entry_price, cut_loss_price, direction, exit_price)):
        return "cut-loss"

    correct_dir = (
        (direction == "bullish" and actual_move > 0)
        or (direction == "bearish" and actual_move < 0)
        or (direction == "neutral" and abs(actual_move) < 1.0)
    )
    return "correct" if correct_dir else "incorrect"


def backfill() -> None:
    """Main backfill routine. Processes all pending predictions older than 24h."""
    from data.fetch_price import get_closing_price, get_next_trading_close, DEFAULT_LOT_SIZE
    from data.ticker_config import STOP_LOSS_PCT

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_data_quality_column(conn)
    _ensure_price_source_columns(conn)

    pending = conn.execute("""
        SELECT id, commodity, signal, direction, prediction_date,
               entry_price, exit_price, cut_loss_price
        FROM predictions
        WHERE prediction_date <= date('now', '-1 day')
          AND (outcome IS NULL OR entry_price IS NULL OR exit_price IS NULL)
    """).fetchall()

    print(f"[backfill] {len(pending)} predictions to process.")

    for row_id, commodity, signal, direction, pred_date, entry_price, exit_price, cut_loss_price in pending:
        direction_norm = _normalise_direction(direction)

        # Step 1: fill entry_price if still missing
        if entry_price is None:
            result = get_closing_price(commodity, pred_date)
            if result:
                entry_price = result["price"]
                entry_source = result["source"]
                # Compute cut_loss_price from entry
                sl_pct = STOP_LOSS_PCT.get(commodity, 1.5) / 100.0
                if direction_norm == "bullish":
                    cut_loss_price = round(entry_price * (1 - sl_pct), 4)
                elif direction_norm == "bearish":
                    cut_loss_price = round(entry_price * (1 + sl_pct), 4)
                conn.execute(
                    "UPDATE predictions SET entry_price=?, entry_price_source=?, cut_loss_price=?, data_quality='ok' WHERE id=?",
                    (entry_price, entry_source, cut_loss_price, row_id),
                )
                print(f"  [{row_id}] entry=${entry_price} cut_loss=${cut_loss_price} ({commodity})")
            else:
                conn.execute("UPDATE predictions SET data_quality='price_fetch_failed' WHERE id=?", (row_id,))

        # Step 2: fetch exit_price
        if exit_price is None:
            result = get_next_trading_close(commodity, pred_date)
            if result:
                exit_price = result["price"]
                exit_date = result["date"]
                exit_source = result["source"]
                conn.execute(
                    "UPDATE predictions SET exit_price=?, exit_price_source=? WHERE id=?",
                    (exit_price, exit_source, row_id),
                )
                print(f"  [{row_id}] exit=${exit_price} ({exit_date})")
            else:
                conn.execute("UPDATE predictions SET data_quality='price_fetch_failed' WHERE id=?", (row_id,))

        # Step 3: guard same-day price error
        if entry_price and exit_price:
            if abs(exit_price - entry_price) < SAME_PRICE_TOLERANCE:
                conn.execute("""
                    UPDATE predictions SET exit_price=NULL, data_quality='same_day_price_error' WHERE id=?
                """, (row_id,))
                conn.commit()
                continue

        # Step 4: compute actual_move, outcome, pnl_usd
        if entry_price and exit_price and entry_price > 0:
            actual_move = round((exit_price - entry_price) / entry_price * 100, 2)
            outcome = classify_outcome(signal, direction_norm, actual_move,
                                       entry_price, cut_loss_price, exit_price)
            direction_mult = 1 if direction_norm == "bullish" else (-1 if direction_norm == "bearish" else 0)
            pnl_usd = round((exit_price - entry_price) * DEFAULT_LOT_SIZE * direction_mult, 2)

            conn.execute("""
                UPDATE predictions
                SET actual_move=COALESCE(actual_move,?), outcome=COALESCE(outcome,?),
                    pnl_usd=?, data_quality=COALESCE(data_quality,'ok')
                WHERE id=?
            """, (actual_move, outcome, pnl_usd, row_id))
            print(f"  [{row_id}] {commodity}: ${entry_price}->${exit_price} = {actual_move:+.2f}% ({outcome}) P&L:${pnl_usd:+,.0f}")

    conn.commit()
    conn.close()
    print("[backfill] Done.")


if __name__ == "__main__":
    backfill()
