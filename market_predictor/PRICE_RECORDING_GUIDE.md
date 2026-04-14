# Price Recording Feature — Detailed Technical Guide
**AI Market Impact Predictor · Petronas Trading Digital**

---

## What This Feature Does

Every time the dashboard loads and scores today's news, it now also:
1. **Records the current market price** for each commodity (entry price)
2. **Records the price 24 hours later** (exit price) via a daily backfill job
3. **Computes actual P&L** from the real price difference
4. **Shows a price chart** in the Prediction Tracker tab — solid line for entry, dashed for exit
5. **Shows a day-by-day table** with entry price, exit price, move %, and outcome

This proves model credibility to traders: you can see exactly what price the signal fired at, what happened 24h later, and whether the prediction was correct.

---

## File-by-File Explanation

### 1. `market_predictor/data/ticker_config.py`

**What it is:** A single Python dictionary that maps every commodity display name to its yfinance ticker symbol.

**Why it exists:** Previously, ticker symbols were hardcoded in multiple places (`backfill_actuals.py`, `collect_prices.py`). If a ticker changes, you'd have to find and update every file. Now there's one place to change it.

**How it works:**
```python
COMMODITY_CONFIG = {
    "Dubai Crude (PCAAT00)": {"ticker_yf": "BZ=F", "unit": "barrel", "currency": "USD"},
    "Brent Dated (PCAAS00)": {"ticker_yf": "BZ=F", "unit": "barrel", "currency": "USD"},
    "WTI (PCACG00)":         {"ticker_yf": "CL=F", "unit": "barrel", "currency": "USD"},
    "LNG / Nat Gas (JKM)":   {"ticker_yf": "NG=F", "unit": "MMBtu",  "currency": "USD"},
    # Short aliases for legacy data
    "Dubai Crude": {"ticker_yf": "BZ=F", ...},
    "Brent":       {"ticker_yf": "BZ=F", ...},
    "WTI":         {"ticker_yf": "CL=F", ...},
    "LNG":         {"ticker_yf": "NG=F", ...},
}
```

**Ticker notes:**
- `BZ=F` = Brent Crude futures (used as proxy for Dubai Crude — closest available on yfinance)
- `CL=F` = WTI Crude Oil futures
- `NG=F` = Henry Hub Natural Gas futures (proxy for JKM LNG)

**How to use it:**
```python
from data.ticker_config import COMMODITY_CONFIG, get_ticker

cfg = COMMODITY_CONFIG.get("Dubai Crude (PCAAT00)")
ticker = cfg["ticker_yf"]   # "BZ=F"
unit   = cfg["unit"]        # "barrel"

# Or use the helper:
ticker = get_ticker("WTI (PCACG00)")  # "CL=F"
ticker = get_ticker("Unknown")         # None
```

---

### 2. `market_predictor/data/fetch_price.py`

**What it is:** The only module in the project that calls yfinance for price data.

**Why it exists:** Centralising yfinance calls makes the code testable (mock one module instead of many), prevents duplicate downloads, and ensures consistent weekend/holiday handling everywhere.

**Two functions:**

#### `get_closing_price(commodity, date) -> float | None`
Returns the closing price for a commodity on a given date.

```
Input:  commodity = "Dubai Crude (PCAAT00)", date = "2026-04-03"
Output: 84.52  (USD per barrel)
```

**Walk-forward logic:** If the market was closed on the requested date (weekend or holiday), it tries the next day, then the day after, up to 5 days forward. This handles:
- Predictions made on Saturday → uses Monday's price
- Predictions made on a public holiday → uses next trading day's price

```python
for offset in range(5):  # try date, date+1, date+2, date+3, date+4
    candidate = start_dt + timedelta(days=offset)
    df = yf.download(ticker, start=candidate, end=candidate+1day)
    if not df.empty:
        return float(df["Close"].iloc[0])
return None  # no data found in 5 days
```

#### `get_next_trading_close(commodity, from_date) -> tuple[float | None, str | None]`
Returns the closing price of the **next** trading day after `from_date`. Used by the backfill job to get the exit price.

```
Input:  commodity = "Brent Dated (PCAAS00)", from_date = "2026-04-03"
Output: (85.10, "2026-04-04")  — price on April 4th
```

Walk-forward starts from `from_date + 1 day` and looks up to 5 days ahead.

**Error handling:** Any yfinance network error, empty response, or missing data returns `None` silently — the prediction row is still saved, just without a price.

---

### 3. `market_predictor/data/init_tracker.py`

**What it is:** Creates and migrates the `predictions` table in `articles.db`.

**Why it was modified:** The original table didn't have price columns. We need to add them without destroying existing prediction data.

**How the migration works:**
```python
NEW_COLUMNS = [
    ("entry_price",    "REAL"),
    ("exit_price",     "REAL"),
    ("price_currency", "TEXT DEFAULT 'USD'"),
    ("price_unit",     "TEXT"),
    ("ticker_yf",      "TEXT"),
]

for col_name, col_type in NEW_COLUMNS:
    try:
        conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
    except sqlite3.OperationalError:
        pass  # column already exists — safe to ignore
```

**Fresh install:** If the table doesn't exist yet, `CREATE TABLE IF NOT EXISTS` creates it with all columns in one shot.

**Run this once** after pulling the new code:
```bash
python market_predictor/data/init_tracker.py
```

---

### 4. `market_predictor/data/backfill_actuals.py`

**What it is:** A daily job that fills in exit prices and computes outcomes for predictions older than 24 hours.

**When to run it:** Every morning (e.g. 9am MYT) via Windows Task Scheduler or cron. It looks for predictions from yesterday that still have `exit_price = NULL`.

**What it does step by step:**

```
1. SELECT all predictions WHERE exit_price IS NULL AND prediction_date <= yesterday
2. For each prediction:
   a. If entry_price is NULL → fetch it via get_closing_price(commodity, prediction_date)
   b. Fetch exit_price via get_next_trading_close(commodity, prediction_date)
   c. If exit_price is None → skip (try again tomorrow)
   d. Compute: actual_move = (exit_price - entry_price) / entry_price * 100
   e. Compute: pnl_usd = actual_move / 100 * entry_price * 1000
   f. Classify outcome: 'correct' or 'incorrect'
   g. UPDATE the predictions row
```

**Outcome classification:**
```python
def classify_outcome(signal, direction, actual_move):
    correct_direction = (
        direction == "rise" and actual_move > 0  OR
        direction == "fall" and actual_move < 0  OR
        direction == "flat" and abs(actual_move) < 1
    )
    correct_magnitude = (
        signal == "HIGH"   and abs(actual_move) > 3    OR
        signal == "MEDIUM" and 1 <= abs(actual_move) <= 3  OR
        signal == "LOW"    and abs(actual_move) < 1
    )
    return "correct" if (correct_direction AND correct_magnitude) else "incorrect"
```

**P&L calculation:**
- `DEFAULT_LOT_SIZE = 1000` barrels (or MMBtu for gas)
- If signal was "rise" and price went up 2%: `pnl_usd = +2/100 * 84.50 * 1000 = +$1,690`
- If signal was "rise" and price went down 1%: `pnl_usd = -1/100 * 84.50 * 1000 = -$845`

**Run manually:**
```bash
python market_predictor/data/backfill_actuals.py
```

---

### 5. `market_predictor/inference/pipeline.py` — `log_prediction()`

**What changed:** `log_prediction()` now fetches the current market price and stores it alongside the prediction.

**How it works:**
```python
def log_prediction(commodity, ticker, result, headline=None):
    # 1. Look up yfinance ticker from config
    cfg = COMMODITY_CONFIG.get(commodity, {})
    ticker_yf = cfg.get("ticker_yf")
    
    # 2. Fetch today's price (may be None if market not yet closed)
    entry_price = get_closing_price(commodity, today) if ticker_yf else None
    
    # 3. Insert into predictions table
    conn.execute("""
        INSERT OR IGNORE INTO predictions
            (prediction_date, commodity, ticker, ticker_yf, signal, direction,
             confidence, expected_move, headline, entry_price, price_unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (..., entry_price, cfg.get("unit")))
```

**Important:** `INSERT OR IGNORE` means if a prediction for this commodity already exists today, it's silently skipped. This prevents duplicate rows when Streamlit reruns.

**When is this called?** From `streamlit_app.py` once per page load, gated by `st.session_state`:
```python
if "logged_today" not in st.session_state:
    for commodity, sig in commodities.items():
        log_prediction(commodity, sig["ticker"], sig, top_headline)
    st.session_state.logged_today = True
```

---

### 6. `market_predictor/app/streamlit_app.py` — Prediction Tracker Tab

**Three new functions added to the Prediction Tracker tab:**

#### `load_price_history(commodity_filter, date_from, date_to)`
Queries the predictions table for rows that have `entry_price` populated, filtered by commodity and date range.

#### `render_price_chart(df)`
Draws a Plotly chart with:
- **Solid line** = entry price (what the price was when the signal fired)
- **Dashed line** = exit price (what the price was 24h later)
- **Signal markers** on the entry line:
  - `▲` triangle-up (green) = Bullish signal
  - `▼` triangle-down (red) = Bearish signal
  - `●` circle (grey) = Neutral signal
- One colour per commodity (Dubai=orange, Brent=blue, WTI=green, LNG=gold)
- Transparent background to match Streamlit dark theme

```
Example chart:
$85 ─────────────────────────────────────────
     ▲ Dubai entry (solid)
$84 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     Dubai exit (dashed)
$83 ─────────────────────────────────────────
    Apr 1    Apr 2    Apr 3    Apr 4
```

#### `render_price_table(df)`
Shows a clean table:

| Date | Commodity | Signal | Entry Price | Exit Price | Move | Outcome |
|------|-----------|--------|-------------|------------|------|---------|
| 2026-04-03 | Dubai Crude | HIGH rise 78% | $84.52 | $87.23 | +3.21% | ✓ Correct |
| 2026-04-03 | Brent | MEDIUM rise 62% | $85.10 | $86.44 | +1.57% | ✓ Correct |
| 2026-04-02 | LNG | LOW flat 91% | $3.21 | Pending | — | ⏳ Pending |

Color coding:
- ✓ Correct → green background
- ✗ Incorrect → red background
- ⏳ Pending → yellow background

---

## Data Flow Diagram

```
Dashboard loads
      │
      ▼
fetch_news() ──► score articles ──► build_summary()
                                          │
                                          ▼
                                  log_prediction()
                                          │
                                          ▼
                              get_closing_price()  ◄── fetch_price.py
                                          │              │
                                          ▼              ▼
                              INSERT into predictions  yfinance
                              (entry_price stored)

─────────────────────────────────────────────────────────────────
Next morning: run backfill_actuals.py
─────────────────────────────────────────────────────────────────

backfill_actuals.py
      │
      ▼
SELECT predictions WHERE exit_price IS NULL AND date <= yesterday
      │
      ▼
get_next_trading_close() ◄── fetch_price.py ◄── yfinance
      │
      ▼
actual_move = (exit - entry) / entry * 100
pnl_usd = actual_move / 100 * entry * 1000
outcome = "correct" or "incorrect"
      │
      ▼
UPDATE predictions row

─────────────────────────────────────────────────────────────────
Prediction Tracker tab
─────────────────────────────────────────────────────────────────

load_price_history() ──► render_price_chart() ──► Plotly chart
                    └──► render_price_table() ──► Styled table
```

---

## Database Schema — predictions table

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    prediction_date  DATE NOT NULL,
    commodity        TEXT NOT NULL,
    ticker           TEXT NOT NULL,          -- Platts code e.g. PCAAT00
    ticker_yf        TEXT,                   -- yfinance symbol e.g. BZ=F  [NEW]
    signal           TEXT NOT NULL,          -- HIGH / MEDIUM / LOW
    direction        TEXT NOT NULL,          -- rise / fall / flat
    confidence       REAL NOT NULL,          -- 0.0 to 1.0
    expected_move    TEXT NOT NULL,          -- <1% / 1-3% / >3%
    headline         TEXT,                   -- top driving headline
    entry_price      REAL,                   -- price at prediction time  [NEW]
    exit_price       REAL,                   -- price 24h later           [NEW]
    price_currency   TEXT DEFAULT 'USD',     -- always USD                [NEW]
    price_unit       TEXT,                   -- barrel or MMBtu           [NEW]
    actual_move      REAL,                   -- % change (computed)
    outcome          TEXT,                   -- correct / incorrect / NULL
    pnl_usd          REAL,                   -- estimated P&L in USD
    UNIQUE(prediction_date, commodity)
);
```

---

## How to Run Everything

### First time setup (run once):
```bash
python market_predictor/data/init_tracker.py
```

### Daily workflow:
```bash
# Morning: fill exit prices for yesterday's predictions
python market_predictor/data/backfill_actuals.py

# Then run the dashboard
.venv\Scripts\streamlit.exe run market_predictor\app\streamlit_app.py
```

### Automate backfill (Windows Task Scheduler):
- Program: `C:\path\to\.venv\Scripts\python.exe`
- Arguments: `market_predictor\data\backfill_actuals.py`
- Trigger: Daily at 09:00 MYT

---

## Frequently Asked Questions

**Q: Why is entry_price sometimes NULL?**
A: If the dashboard runs before markets close (e.g. 8am MYT), yfinance may not have today's closing price yet. The backfill job will fill it in the next morning.

**Q: Why use BZ=F for Dubai Crude instead of a Dubai-specific ticker?**
A: Dubai Crude (PCAAT00) is a Platts-assessed price not directly available on yfinance. Brent (BZ=F) is the closest freely available proxy. For production use, replace with a proper Dubai crude data feed.

**Q: What is DEFAULT_LOT_SIZE = 1000?**
A: This is the assumed position size for P&L calculation — 1000 barrels (or 1000 MMBtu for gas). Adjust this in `fetch_price.py` to match Petronas standard lot sizes.

**Q: How do I know if a prediction was "correct"?**
A: Both direction AND magnitude must be right:
- HIGH + rise → price must go up more than 3%
- MEDIUM + rise → price must go up 1-3%
- LOW + flat → price must move less than 1%

**Q: Can I run backfill multiple times?**
A: Yes, it's idempotent. It only processes rows where `exit_price IS NULL`, so already-filled rows are skipped.
