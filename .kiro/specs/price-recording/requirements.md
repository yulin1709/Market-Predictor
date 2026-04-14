# Requirements Document

## Introduction

The Prediction Tracker currently logs signal/direction for commodity price predictions but does not record actual market prices. This feature adds price recording to the predictions table — capturing entry and exit prices via yfinance — and surfaces a price chart in the Prediction Tracker tab of the Streamlit dashboard. This enables PnL calculation, richer accuracy analysis, and visual price history per commodity.

## Glossary

- **Prediction_Tracker**: The SQLite-backed system that logs ML predictions and their outcomes.
- **Entry_Price**: The closing price of a commodity on the prediction date (the price at signal time).
- **Exit_Price**: The closing price of a commodity on the next valid trading day after the prediction date (the price used to evaluate the prediction).
- **Ticker_Config**: The single source of truth mapping commodity display names to yfinance ticker symbols.
- **Price_Fetcher**: The module responsible for all yfinance calls to retrieve commodity closing prices.
- **Backfill_Service**: The scheduled script that fills missing entry/exit prices and computes actual_move and pnl_usd for predictions older than 24 hours.
- **Price_Chart**: The Plotly chart in the Prediction Tracker tab showing entry and exit price lines with signal markers.
- **Trading_Day**: Any weekday on which the commodity futures market is open (weekends and holidays excluded).
- **actual_move**: The percentage price change computed as `(exit_price - entry_price) / entry_price * 100`.
- **pnl_usd**: Estimated profit/loss in USD based on a default lot size of 1000 units.
- **DEFAULT_LOT_SIZE**: The fixed position size (1000 units) used for pnl_usd calculation.

---

## Requirements

### Requirement 1: Schema Migration for Price Columns

**User Story:** As a data engineer, I want the predictions table to store entry and exit prices, so that I can compute accurate PnL and actual price moves from raw price data.

#### Acceptance Criteria

1. THE Prediction_Tracker SHALL add `entry_price REAL`, `exit_price REAL`, `price_currency TEXT DEFAULT 'USD'`, `price_unit TEXT`, and `ticker_yf TEXT` columns to the predictions table.
2. WHEN the predictions table already exists without the new columns, THE Prediction_Tracker SHALL add the missing columns via `ALTER TABLE` migration without data loss.
3. WHEN the predictions table does not yet exist, THE Prediction_Tracker SHALL create it with all columns including the new price columns in a single `CREATE TABLE` statement.
4. THE Prediction_Tracker SHALL enforce a `UNIQUE(prediction_date, commodity)` constraint on the predictions table to prevent duplicate rows.
5. WHEN a duplicate `(prediction_date, commodity)` insert is attempted, THE Prediction_Tracker SHALL silently ignore the duplicate using `INSERT OR IGNORE`.

---

### Requirement 2: Commodity-to-Ticker Mapping

**User Story:** As a developer, I want a single source of truth for commodity-to-yfinance-ticker mappings, so that all modules use consistent ticker symbols without duplication.

#### Acceptance Criteria

1. THE Ticker_Config SHALL define a mapping from commodity display names to yfinance ticker symbols covering at minimum: Dubai Crude → `BZ=F`, Brent → `BZ=F`, WTI → `CL=F`, LNG / Nat Gas → `NG=F`.
2. THE Ticker_Config SHALL be the only place in the codebase where commodity-to-ticker mappings are defined.
3. WHEN a commodity name is not found in the Ticker_Config mapping, THE Ticker_Config SHALL return `None` to signal an unmapped commodity.

---

### Requirement 3: Price Fetching Module

**User Story:** As a developer, I want a dedicated price fetching module, so that yfinance calls are isolated, testable, and reusable across the pipeline and backfill scripts.

#### Acceptance Criteria

1. THE Price_Fetcher SHALL expose a `get_closing_price(commodity, date) -> float | None` function that returns the closing price for a commodity on a given date.
2. THE Price_Fetcher SHALL expose a `get_next_trading_close(commodity, from_date) -> tuple[float | None, str | None]` function that returns the closing price and actual date of the next valid trading day after `from_date`.
3. WHEN the requested date falls on a weekend or market holiday, THE Price_Fetcher SHALL walk forward up to 5 calendar days to find the next available closing price.
4. WHEN no closing price is found within 5 days, THE Price_Fetcher SHALL return `(None, None)` without raising an exception.
5. IF a yfinance network or data error occurs, THEN THE Price_Fetcher SHALL catch the exception, log a warning, and return `None` or `(None, None)` as appropriate.
6. THE Price_Fetcher SHALL be the only module in the codebase that calls yfinance for price data.

---

### Requirement 4: Entry Price at Prediction Time

**User Story:** As a trader, I want every new prediction to record the current market price, so that I can track the exact price level at which the signal was generated.

#### Acceptance Criteria

1. WHEN `log_prediction()` is called, THE Prediction_Tracker SHALL call `get_closing_price(commodity, today)` and store the result as `entry_price` in the predictions row.
2. WHEN `get_closing_price` returns `None` for the current date (e.g., market not yet closed), THE Prediction_Tracker SHALL store `NULL` for `entry_price` and proceed with logging the prediction.
3. THE Prediction_Tracker SHALL store the yfinance ticker symbol used for the price lookup in the `ticker_yf` column.

---

### Requirement 5: Exit Price and Outcome Backfill

**User Story:** As a data analyst, I want predictions older than 24 hours to have exit prices and computed outcomes, so that I can evaluate prediction accuracy using real price data.

#### Acceptance Criteria

1. WHEN the Backfill_Service runs, THE Backfill_Service SHALL select all predictions where `exit_price IS NULL` and `prediction_date <= date('now', '-1 day')`.
2. FOR each selected prediction, THE Backfill_Service SHALL call `get_next_trading_close(commodity, prediction_date)` to fetch the exit price.
3. WHEN `entry_price IS NULL` for a selected prediction, THE Backfill_Service SHALL first attempt to fill `entry_price` via `get_closing_price(commodity, prediction_date)` before computing `actual_move`.
4. WHEN both `entry_price` and `exit_price` are available, THE Backfill_Service SHALL compute `actual_move = (exit_price - entry_price) / entry_price * 100` rounded to 2 decimal places.
5. THE Backfill_Service SHALL compute `pnl_usd = actual_move / 100 * entry_price * DEFAULT_LOT_SIZE` where `DEFAULT_LOT_SIZE = 1000`.
6. WHEN `get_next_trading_close` returns `None`, THE Backfill_Service SHALL skip the prediction and leave `exit_price` as `NULL` for the next run.
7. THE Backfill_Service SHALL NOT fetch `actual_move` independently from yfinance; THE Backfill_Service SHALL compute `actual_move` exclusively from `entry_price` and `exit_price`.

---

### Requirement 6: Price Chart in Prediction Tracker Tab

**User Story:** As a trader, I want to see a price chart in the Prediction Tracker tab, so that I can visually correlate prediction signals with actual price movements.

#### Acceptance Criteria

1. THE Price_Chart SHALL display entry prices as a solid line and exit prices as a dashed line, one series per commodity.
2. THE Price_Chart SHALL overlay signal markers on the entry price line, colored by signal strength (HIGH = red, MEDIUM = orange, LOW = green).
3. WHEN no price data is available for the selected filters, THE Price_Chart SHALL display an informational message instead of an empty chart.
4. THE Price_Chart SHALL be rendered using Plotly and embedded in the Prediction Tracker tab of the Streamlit dashboard.
5. THE Price_Chart SHALL respect the commodity and date filters already present in the Prediction Tracker tab.

---

### Requirement 7: Price Table in Prediction Tracker Tab

**User Story:** As a trader, I want a day-by-day table showing entry price, exit price, and outcome, so that I can review individual prediction performance at a glance.

#### Acceptance Criteria

1. THE Prediction_Tracker SHALL render a tabular view showing `prediction_date`, `commodity`, `signal`, `direction`, `entry_price`, `exit_price`, `actual_move`, and `outcome` columns.
2. WHEN `entry_price` or `exit_price` is `NULL`, THE Prediction_Tracker SHALL display "Pending" in the corresponding table cell.
3. THE Prediction_Tracker SHALL color-code the `outcome` column: correct = green, incorrect = red, pending = yellow.

---

### Requirement 8: Weekend and Holiday Gap Handling

**User Story:** As a developer, I want price fetching to gracefully handle non-trading days, so that predictions made on weekends or holidays still receive valid prices.

#### Acceptance Criteria

1. WHEN a prediction date is a Saturday or Sunday, THE Price_Fetcher SHALL return the closing price of the following Monday (or next Trading_Day if Monday is a holiday).
2. WHEN a prediction date is a public market holiday, THE Price_Fetcher SHALL return the closing price of the next available Trading_Day.
3. THE Price_Fetcher SHALL walk forward a maximum of 5 calendar days before returning `None`.
4. FOR ALL valid commodity and date inputs where market data exists, `get_closing_price` followed by `get_next_trading_close` SHALL return two distinct prices representing consecutive Trading_Days (round-trip consistency).
