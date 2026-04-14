# Implementation Plan: Price Recording

## Overview

Extend the Prediction Tracker to capture entry/exit prices via yfinance, compute PnL, and surface a price chart and table in the Streamlit dashboard. Implementation follows the file order: ticker_config → fetch_price → init_tracker migration → backfill refactor → pipeline update → streamlit additions.

## Tasks

- [x] 1. Create `ticker_config.py` — single source of truth for commodity-to-ticker mappings
  - Define `COMMODITY_CONFIG` dict with all eight commodity entries (full names + aliases)
  - Implement `get_ticker(commodity) -> str | None` returning `None` for unmapped commodities
  - _Requirements: 2.1, 2.2, 2.3_

  - [ ]* 1.1 Write unit tests for `ticker_config.py`
    - Verify all four required commodities map to correct yfinance tickers
    - Verify unmapped string returns `None`
    - _Requirements: 2.1, 2.3_

  - [ ]* 1.2 Write property test for unmapped commodity returns None
    - **Property 3: Unmapped commodity returns None**
    - **Validates: Requirements 2.3**

- [x] 2. Create `fetch_price.py` — centralised yfinance price fetching
  - Implement `get_closing_price(commodity, date) -> float | None` with walk-forward up to 5 days (offsets 0–4)
  - Implement `get_next_trading_close(commodity, from_date) -> tuple[float | None, str | None]` with walk-forward up to 5 days (offsets 1–5)
  - Define `DEFAULT_LOT_SIZE = 1000`
  - Catch all yfinance exceptions; log warning and return `None` / `(None, None)`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 8.1, 8.2, 8.3_

  - [ ]* 2.1 Write unit tests for `fetch_price.py`
    - Mock yfinance; test happy path, empty DataFrame response, exception path
    - _Requirements: 3.1, 3.2, 3.5_

  - [ ]* 2.2 Write property test for weekend walk-forward returns a weekday price
    - **Property 4: Weekend and holiday walk-forward returns a weekday price**
    - **Validates: Requirements 3.3, 8.1, 8.2**

  - [ ]* 2.3 Write property test for walk-forward never exceeds 5 calendar days
    - **Property 5: Walk-forward never exceeds 5 calendar days**
    - **Validates: Requirements 3.4, 8.3**

  - [ ]* 2.4 Write property test for round-trip price consistency
    - **Property 12: Round-trip price consistency**
    - **Validates: Requirements 8.4**

- [x] 3. Checkpoint — Files created, syntax verified, DB migration applied.

- [x] 4. Modify `init_tracker.py` — ALTER TABLE migration for new price columns
  - Add `CREATE TABLE IF NOT EXISTS` with all columns including the five new price columns
  - Add `NEW_COLUMNS` list: `entry_price REAL`, `exit_price REAL`, `price_currency TEXT DEFAULT 'USD'`, `price_unit TEXT`, `ticker_yf TEXT`
  - Loop over `NEW_COLUMNS` and attempt `ALTER TABLE ADD COLUMN` per column, catching `OperationalError` to skip already-existing columns
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 4.1 Write unit tests for `init_tracker.py`
    - Test fresh DB gets all columns including new price columns
    - Test existing DB without new columns gets migrated without data loss
    - _Requirements: 1.2, 1.3_

  - [ ]* 4.2 Write property test for schema migration preserves existing data
    - **Property 2: Schema migration preserves existing data**
    - **Validates: Requirements 1.2**

  - [ ]* 4.3 Write property test for duplicate insert is silently ignored
    - **Property 1: Duplicate insert is silently ignored**
    - **Validates: Requirements 1.4, 1.5**

- [x] 5. Modify `backfill_actuals.py` — refactor to use `fetch_price.py`
  - Remove inline `TICKER_MAP` dict and `get_actual_move` / `next_trading_day` functions
  - Import `get_closing_price`, `get_next_trading_close`, `DEFAULT_LOT_SIZE` from `data.fetch_price`
  - Update SELECT query to filter `exit_price IS NULL AND prediction_date <= date('now', '-1 day')`
  - For each row: if `entry_price IS NULL`, fill via `get_closing_price(commodity, prediction_date)` first
  - Fetch `exit_price, exit_date = get_next_trading_close(commodity, prediction_date)`; skip row if `None`
  - Compute `actual_move = round((exit_price - entry_price) / entry_price * 100, 2)` with guard for `entry_price == 0`
  - Compute `pnl_usd = actual_move / 100 * entry_price * DEFAULT_LOT_SIZE`
  - UPDATE predictions with `exit_price`, `actual_move`, `pnl_usd`, `outcome`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 5.1 Write unit tests for `backfill_actuals.py`
    - Mock `get_closing_price` and `get_next_trading_close`; verify `actual_move` and `pnl_usd` computed correctly
    - Verify NULL `entry_price` is backfilled before computing `actual_move`
    - Verify row is skipped when `get_next_trading_close` returns `None`
    - _Requirements: 5.3, 5.4, 5.5, 5.6_

  - [ ]* 5.2 Write property test for backfill selects only eligible predictions
    - **Property 7: Backfill selects only eligible predictions**
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 5.3 Write property test for actual_move formula correctness
    - **Property 8: actual_move formula correctness**
    - **Validates: Requirements 5.4**

  - [ ]* 5.4 Write property test for pnl_usd formula correctness
    - **Property 9: pnl_usd formula correctness**
    - **Validates: Requirements 5.5**

- [x] 6. Checkpoint — All mandatory tasks pass syntax check.

- [x] 7. Modify `pipeline.py` — extend `log_prediction()` to store `entry_price` and `ticker_yf`
  - Import `get_closing_price` from `data.fetch_price` and `COMMODITY_CONFIG` from `data.ticker_config`
  - At the start of `log_prediction`, look up `cfg = COMMODITY_CONFIG.get(commodity, {})`; extract `ticker_yf`
  - Call `get_closing_price(commodity, date.today().isoformat())` if `ticker_yf` is set, else `entry_price = None`
  - Add `entry_price` and `ticker_yf` to the `INSERT OR IGNORE` statement
  - Remove the inline `CREATE TABLE IF NOT EXISTS` from `log_prediction` (table is managed by `init_tracker`)
  - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 7.1 Write unit tests for updated `log_prediction`
    - Mock `get_closing_price`; verify DB row has correct `ticker_yf` and `entry_price`
    - Verify `entry_price = NULL` stored when `get_closing_price` returns `None`
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 7.2 Write property test for log_prediction stores entry_price and ticker_yf
    - **Property 6: log_prediction stores entry_price and ticker_yf**
    - **Validates: Requirements 4.1, 4.3**

- [x] 8. Modify `streamlit_app.py` — add price chart and price table to Prediction Tracker tab
  - Add `load_price_history(commodity_filter, date_from, date_to) -> pd.DataFrame` querying `entry_price`, `exit_price`, `signal`, `direction`, `prediction_date`, `commodity` from predictions
  - Add `render_price_chart(df)`: Plotly figure with transparent background, solid entry-price line and dashed exit-price line per commodity, signal markers on entry line (triangle-up bullish / triangle-down bearish) colored HIGH=#FF4B4B, MEDIUM=#FFA500, LOW=#21C354; show `st.info` when df is empty
  - Add `render_price_table(df)`: tabular view with columns `prediction_date`, `commodity`, `signal`, `direction`, `entry_price`, `exit_price`, `actual_move`, `outcome`; NULL prices → "Pending"; `outcome` color-coded via `applymap`
  - Call `render_price_chart` and `render_price_table` inside tab2, below the existing accuracy charts, using the same commodity and date filters
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3_

  - [ ]* 8.1 Write unit tests for `streamlit_app.py` chart and table functions
    - Verify Plotly figure has solid and dashed traces with correct marker symbols and colors
    - _Requirements: 6.1, 6.2_

  - [ ]* 8.2 Write property test for price chart respects active filters
    - **Property 10: Price chart respects active filters**
    - **Validates: Requirements 6.5**

  - [ ]* 8.3 Write property test for NULL prices display as "Pending"
    - **Property 11: NULL prices display as "Pending"**
    - **Validates: Requirements 7.2**

- [x] 9. Final checkpoint — All files pass `py_compile`. DB migration applied. Docs created.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- `fetch_price.py` is the sole yfinance gateway — no other module may call yfinance for price data
- Property tests use Hypothesis with `@settings(max_examples=100)` and must include the comment `# Feature: price-recording, Property N: <property_text>`
