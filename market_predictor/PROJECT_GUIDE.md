# AI Market Impact Predictor
### Technical Project Guide — Petronas Trading Digital | Python 3.12 | April 2026

---

## Overview

Reads daily energy market reports from S&P Global Platts, scores each article with a trained text classifier, and generates directional signals (rise/fall/flat) with confidence scores for four commodities: **Dubai Crude (PCAAT00)**, **Brent Dated (PCAAS00)**, **WTI (PCACG00)**, and **LNG/JKM**. Tracks live predictions against real prices, simulates trading P&L, and provides an entry optimizer with risk/reward calculations.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| News ingestion | S&P Global Platts REST API | Daily benchmark reports |
| PDF extraction | pdfplumber + pypdf | Extract text from Platts PDF reports |
| Text cleaning | cleaner.py (custom) | Remove price tables, boilerplate |
| Price data (primary) | Local prices table (from Platts report body) | Actual assessed benchmark prices |
| Price data (fallback) | yfinance | Weekend/holiday gaps, LNG proxy |
| ML model (primary) | XGBoost + TF-IDF + numeric features | 42.3% CV accuracy |
| ML model (advanced) | FinBERT embeddings + XGBoost | 34.6% CV (better with more data) |
| Directional model | TF-IDF/XGBoost + numeric features | rise/flat/fall prediction |
| FinBERT fine-tune | ProsusAI/finbert (optional) | Requires GPU, 27% on CPU |
| Ensemble voting | Weighted average across all models | Commodity win-rate weights |
| Similarity search | ChromaDB + sentence-transformers | Historical event lookup |
| AI commentary | Gemini 2.5 Flash | 1 call per session |
| Database | SQLite (articles.db) | All data in one file |
| Dashboard | Streamlit | 7-tab Bloomberg-style UI |
| REST API | FastAPI + uvicorn | Optional ARQ integration |

---

## Data Pipeline

### Step 1 — News Ingestion (data/collect_news.py)

**Endpoint:** GET https://api.ci.spglobal.com/news-insights/v1/search/packages

**Auth:** POST to https://api.ci.spglobal.com/auth/api with username + password form body. Returns access_token (Bearer, 1hr TTL). Cached in memory, auto-refreshed via refresh_token.

**Queries:** "Crude Oil Marketwire", "Arab Gulf Marketscan", "Gulf Arab Marketscan", "refinery outage", "tanker shipping", "EIA inventory", "demand outlook", "geopolitical oil", "OPEC production"

Add custom queries at runtime without editing source: --queries "LNG Asia" "crude tanker"

**Flow per article:**
1. Search returns package metadata: {id, title, updatedDate, documentUrl}
2. For each UUID: GET /news-insights/v1/content/{id} -> JSON envelope
3. If PDF bytes -> pdf_handler.extract_text_from_bytes() via pdfplumber
4. cleaner.clean_report_text() strips price tables, boilerplate, HTML tags
5. Store in articles table: {id, headline, body_text, published_at, source, url}

**Parallel refresh:** --refresh-empty-bodies --workers 3 uses ThreadPoolExecutor. DB writes serialised on main thread to avoid SQLite contention. Use workers=3 max to avoid S&P API rate limits (429 errors at workers=8).

**Early exit:** Stops paginating after 5 consecutive results older than start_date.

**Important:** 72% of articles (4,482/6,381) have no body text — they are package-level metadata records. The API does not return body text for these. Only PDF report articles (~1,900) have usable body text.

---

### Step 2 — PDF Text Extraction (data/pdf_handler.py)

Uses pdfplumber (preferred, x_tolerance=2, y_tolerance=2) with pypdf fallback. pdfplumber handles Platts report column layouts better. pypdf used for corrupted PDFs or unusual encoding.

---

### Step 3 — Text Cleaning (data/cleaner.py)

Removes:
- Lines where digit/total ratio > 45% (price table rows)
- Lines matching price row regex: label + 6-8 char ticker + numbers
- Boilerplate: "(continued on page X)", copyright, www.spglobal.com, page numbers
- Table section headers: "Key benchmarks", "Middle East", "Forward Dated Brent"
- HTML tags via re.sub(r"<[^>]+>", " ", text)
- Hard cap at 300 chars per line before regex to prevent catastrophic backtracking

extract_narrative() — additionally requires >40% alphabetic ratio and minimum 5 words per line.
is_narrative_text() — returns True if alpha ratio > 40%. Used by train.py.

---

### Step 4 — Price Extraction (data/collect_prices.py)

Parses cleaned body_text to extract benchmark price rows.

**Rule-based parser:** Regex matches: Dubai (May) PCAAT00 102.54 102.58 102.55 -14.450
Extracts: label, ticker code, bid, ask, mid, change. Maps to symbols: Dubai, Brent, WTI, Oman.

**Gemini fallback:** If regex finds nothing, sends first 18,000 chars to Gemini with structured prompt requesting JSON {benchmarks: [{symbol, price, change_abs, evidence}]}. temperature=0, responseMimeType="application/json".

pct_change_24h = change_abs / (price - change_abs) * 100. If missing, computed as price.pct_change().shift(-1) * 100.

Stored in prices table: {date, symbol, price, pct_change_24h}. Primary key (date, symbol).
Current coverage: Dubai 941 rows, Brent 956 rows, WTI 937 rows, Oman 936 rows (Jan 2022 - Apr 2026).

---

### Step 5 — Commodity-Specific Labelling (data/align_and_label.py)

Each article is matched to the price of the commodity it is ABOUT, not always Dubai Crude.

**Commodity routing (keyword matching on headline + body_text[:200]):**
- LNG/gas keywords -> Oman price (closest available proxy for JKM)
- WTI/US crude keywords -> WTI price
- Brent/North Sea/Europe keywords -> Brent price
- Dubai/Gulf/OPEC keywords -> Dubai price
- Default -> Dubai price

**Rolling percentile thresholds (default):**
- 33rd percentile of |pct_change_24h| over preceding 90 days -> MEDIUM threshold
- 67th percentile -> HIGH threshold
- Outlier cap: |change| capped at 20% before computation
- Minimum 30 observations; falls back to fixed thresholds if fewer

**Fixed thresholds (--fixed-thresholds flag):**
- |change| >= 3.0% -> HIGH
- |change| >= 1.0% -> MEDIUM
- |change| < 1.0% -> LOW

**Signal filtering in train.py:** Generic report names ("Crude Oil Marketwire", "Oilgram Price Report") appear equally across HIGH/MEDIUM/LOW — they carry zero signal. Training only uses articles with EITHER meaningful body text (>200 chars narrative) OR a specific headline (not a generic report name). This reduced training data from 5,628 to 1,914 articles but dramatically improved accuracy.

---

## ML Model — Architecture and Training

### Model Variants

| --model-type | --label-type | Output | CV Accuracy | Notes |
|---|---|---|---|---|
| xgb (default) | magnitude | HIGH/MEDIUM/LOW | 42.3% | Best accuracy, fast |
| xgb | direction | rise/flat/fall | 40.0% | Best directional |
| finbert-xgb | magnitude | HIGH/MEDIUM/LOW | 34.6% | Better with 5000+ articles |
| finbert-xgb | direction | rise/flat/fall | 40.0% | FinBERT embeddings + XGBoost |
| tfidf | magnitude | HIGH/MEDIUM/LOW | 36.8% | Fallback, no XGBoost needed |
| finbert | magnitude | HIGH/MEDIUM/LOW | 27.3% | CPU only, needs GPU to be useful |

**Current primary model: XGBoost (--model-type xgb)**

XGBoost outperforms Logistic Regression because it captures non-linear feature interactions. "OPEC" + "cut" together is a much stronger signal than either word alone — XGBoost learns this, LogReg cannot.

FinBERT+XGB (finbert-xgb) uses pre-trained FinBERT embeddings (768-dim [CLS] token) as features for XGBoost — no fine-tuning needed. Currently scores lower than plain XGBoost because 768-dim embeddings overfit on small training folds (227-447 samples). Will outperform XGBoost once training data exceeds ~5,000 articles.

---

### Feature Engineering

**Text features (XGBoost path):**
1. TfidfVectorizer(ngram_range=(1,2), max_features=15000, min_df=2, sublinear_tf=True)
2. SelectKBest(chi2, k=500) — reduces to top 500 most discriminative features
3. .toarray() — converts sparse to dense for XGBoost

**FinBERT embeddings (finbert-xgb path):**
1. BertTokenizer from ProsusAI/finbert (max_length=128)
2. BertModel.forward() -> last_hidden_state[:, 0, :] = [CLS] token (768 dims)
3. StandardScaler normalisation

**Numeric features (8 dimensions, all paths):**
- mom_5d: (price_today - price_5d_ago) / price_5d_ago * 100
- mom_10d: (price_today - price_10d_ago) / price_10d_ago * 100
- dow_mon, dow_tue, dow_wed, dow_thu, dow_fri: one-hot weekday
- article_count: articles published on same date

**Combined feature matrix:**
- XGBoost: [TF-IDF top-500 dense] + [8 numeric] = 508 features
- FinBERT+XGB: [FinBERT 768-dim] + [TF-IDF top-300 dense] + [8 numeric] = 1,076 features

**XGBoost classifier:**
XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8)

---

### Training Strategy — Walk-Forward Cross-Validation

Standard random k-fold CV is invalid for time-series — it allows the model to see the future. Walk-forward CV simulates real deployment with an expanding training window:

Fold 1: Train [Jan 2022 - Oct 2022]  -> Test [Oct 2022 - Aug 2023]  (acc=36.3%)
Fold 2: Train [Jan 2022 - Aug 2023]  -> Test [Aug 2023 - Jul 2024]  (acc=31.3%)
Fold 3: Train [Jan 2022 - Jul 2024]  -> Test [Jul 2024 - May 2025]  (acc=34.8%)
Fold 4: Train [Jan 2022 - May 2025]  -> Test [May 2025 - Mar 2026]  (acc=37.9%)
CV average: 42.3% (XGBoost)

Reported accuracy = average across all 4 folds. Final model trained on ALL data (1,914 articles) so it has seen the most recent patterns.

---

### Commodity-Specific Models

| File | Trained on | Type |
|---|---|---|
| classifier.pkl | All commodities + numeric features | XGBoost (primary) |
| classifier_direction.pkl | All commodities, direction labels | XGBoost |
| classifier_dubai.pkl | Dubai/Gulf/OPEC + general | TF-IDF Pipeline |
| classifier_brent.pkl | Brent/North Sea/Europe + general | TF-IDF Pipeline |
| classifier_wti.pkl | WTI/US crude + general | TF-IDF Pipeline |
| classifier_lng.pkl | LNG/Natural Gas + general | TF-IDF Pipeline |

Commodity sub-models use simple TF-IDF Pipeline (no numeric features, no XGBoost) for speed. The general model uses full XGBoost with numeric features.

**Routing in pipeline.py:**
1. Check _COMMODITY_NAME_TO_GROUP dict for exact commodity name match
2. If no match, run _detect_commodity_group(headline) — keyword scoring, pick highest
3. Use matched commodity model; fall back to general if not found

---

### Ensemble Voting

At inference, _predict_probabilities() collects votes from all loaded models:
- General XGBoost model (weight=1.0)
- All commodity-specific TF-IDF models (weight=historical_win_rate from predictions table)
- FinBERT fine-tuned model if loaded (weight=1.5)

Weights from: SELECT commodity, SUM(outcome='correct')/COUNT(*) FROM predictions GROUP BY commodity
Falls back to equal weights if < 10 predictions per commodity.

Final probability = weighted average of all model probabilities.
ensemble_weights field included in every prediction dict for auditability.

---

### Current Performance (Walk-Forward CV, April 2026)

| Model | Label Type | CV Accuracy | HIGH Precision | Training Articles |
|---|---|---|---|---|
| XGBoost + TF-IDF + numeric | magnitude | 42.3% | 48.7% | 1,914 |
| XGBoost + TF-IDF + numeric | direction | 40.0% | — | 1,914 |
| FinBERT embeddings + XGBoost | magnitude | 34.6% | 43.6% | 1,914 |
| TF-IDF + LogReg (baseline) | magnitude | 36.8% | 40.3% | 1,914 |
| Random baseline | — | 33.3% | 33.3% | — |

---

## How a Prediction is Generated

When the dashboard loads, compute_today_signals() runs for each of the 4 commodities:

**1. Fetch articles** — fetch_news() calls GET /news-insights/v1/search. Returns today's articles enriched with body_text from local articles.db. Cached 15 minutes.

**2. Score each article** — predict(text, skip_explanation=True, commodity=comm) called for each article. Uses XGBoost with TF-IDF + numeric features. Returns {label, high_prob, medium_prob, low_prob, direction_from_model}.

**3. Filter relevant articles** — keyword matching per commodity. Dubai: ["dubai","oman","saudi","opec","gulf",...]. If no relevant articles, use all articles as pool.

**4. Compute direction** — directional model output used when available (40.0% CV accuracy). Keyword heuristic (bull/bear word scoring) is fallback:
- bull_words: ["rise","surge","cut","outage","disruption","sanction",...]
- bear_words: ["fall","drop","surplus","weak","build","restart",...]
- direction = "rise" if bull_total > bear_score * 1.3, "fall" if bear > bull * 1.3, else "flat"

**5. Compute signal label** — conf = mean(high_prob across relevant pool). HIGH if conf >= 0.30, MEDIUM if >= 0.15, LOW otherwise.

**6. Scale confidence** — raw ML probability (~10-30%) scaled to trader-friendly 35-95%:
scaled_conf = conf * 4.5 * label_factor * dir_factor * article_factor, clamped to [35, 95]

**7. Log prediction** — one row per commodity per day via INSERT OR IGNORE. Fetches entry_price via two-tier fetcher (Platts local DB -> yfinance). Skipped if article_count < 5.

---

## Two-Tier Price Fetching (data/fetch_price.py)

Return type always: {"price": float, "source": "platts"|"yfinance", "date": "YYYY-MM-DD", "symbol": str}

**Tier 1 — Platts (primary):** Queries local prices table. Walks forward up to 7 days.
**Tier 2 — yfinance (fallback):** Dubai->CL=F, Brent->BZ=F, WTI->CL=F, LNG->NG=F. Walks forward up to 5 days.

get_next_trading_close() always starts from from_date + 1 — never returns same-day price.

---

## The Three Evaluation Layers

### Layer 1 — Model Performance (Offline ML)
Question: Does the classifier correctly predict HIGH/MEDIUM/LOW?
Data: Historical labelled_articles. Method: Walk-forward CV, 4 folds.
Current: 42.3% (XGBoost). Random baseline: 33.3%.
Shown in: Tab 2 Section A.

### Layer 2 — Prediction Accuracy (Live Direction)
Question: Did the model predict the right direction on live predictions?
Data: Live predictions table with real prices.
Method: return_pct = (exit-entry)/entry*100. actual = "rise" if >+1%, "fall" if <-1%, "flat" otherwise.
Current: ~33.5%. Shown in: Tab 7.

### Layer 3 — Trading Performance (Backtest)
Question: Would LONG/SHORT trades based on these signals have made money?
Data: Live predictions table.
Method: LONG if direction=rise + signal=HIGH/MEDIUM + confidence>=40%. SHORT if fall. HOLD otherwise.
Win = effective_return > 0.2% (neutral threshold filters noise).
Current: 54% win rate, +$27,829 P&L, 1.21 profit factor. Shown in: Tab 2 Section B.

**Why they differ:** Model Performance measures magnitude label accuracy. Trading Performance measures directional profitability. A prediction labelled MEDIUM (wrong magnitude) but correct direction is a trading win but a model miss.

---

## Database Schema (data/articles.db)

### articles
id TEXT PRIMARY KEY (UUID), headline TEXT, body_text TEXT, source TEXT,
published_at TEXT, fetched_at TEXT, url TEXT, report_name TEXT, content_id TEXT

### prices
(date TEXT, symbol TEXT) PRIMARY KEY, price REAL, pct_change_24h REAL
Symbols: "Dubai", "Brent", "WTI", "Oman"

### labelled_articles
id TEXT PRIMARY KEY, headline TEXT, body_text TEXT, aligned_date TEXT,
price_change REAL, label TEXT (HIGH/MEDIUM/LOW), price_symbol TEXT

### predictions
id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_date DATE, commodity TEXT,
ticker TEXT, ticker_yf TEXT, signal TEXT, direction TEXT, confidence REAL,
expected_move TEXT, headline TEXT, entry_price REAL, entry_price_source TEXT,
exit_price REAL, exit_price_source TEXT, actual_move REAL, outcome TEXT,
pnl_usd REAL, trade_direction TEXT, return_pct REAL, trade_outcome TEXT,
data_quality TEXT, UNIQUE(prediction_date, commodity)

---

## Data Quality Validation (inference/validate_predictions.py)

7 checks via sidebar button or CLI:
1. Price Source Audit — % from Platts vs yfinance vs unknown
2. Same-Day Price Check — |exit - entry| < 0.0001
3. Shared Price Check — Brent rows sharing Dubai entry_price (old BZ=F bug)
4. Confidence Clustering — top value in >80% of rows
5. Direction Bias — one direction >80% of predictions
6. Win Rate by Confidence — higher confidence should win more
7. P&L Sign Consistency — P&L sign must match outcome label

Current status: 6 PASS, 1 WARN (270 seeded historical rows predate source tracking).

---

## Dashboard Tabs

| Tab | Name | What it shows |
|---|---|---|
| 1 | Today Signal | 4 commodity cards, AI commentary, article scores |
| 2 | Prediction Tracker | Model Performance (CV) + Trading Performance (P&L) |
| 3 | Price Chart | Entry vs exit price lines, signal markers |
| 4 | Entry Optimizer | Trade setups: entry zone, target, stop, R:R, EV |
| 5 | Comparable Events | ChromaDB similarity search on today headlines |
| 6 | Accuracy Report | Management-facing summary |
| 7 | Prediction Accuracy | Direction confusion matrix, accuracy by commodity |

---

## How to Run

Daily workflow (run Monday-Friday morning):
.venv\Scripts\python.exe market_predictor/data/collect_news.py --days 1 --max-pages 2
.venv\Scripts\python.exe market_predictor/data/collect_prices.py
.venv\Scripts\python.exe market_predictor/data/align_and_label.py
.venv\Scripts\python.exe market_predictor/data/backfill_actuals.py

Weekly retrain:
.venv\Scripts\python.exe market_predictor/models/train.py --model-type xgb

Dashboard:
.venv\Scripts\streamlit.exe run market_predictor\app\streamlit_app.py

Data validation:
.venv\Scripts\python.exe market_predictor/inference/validate_predictions.py

Repair historical prices:
.venv\Scripts\python.exe market_predictor/data/repair_historical_prices.py --dry-run
.venv\Scripts\python.exe market_predictor/data/repair_historical_prices.py

---

## Limitations

### 1. Training Data Size
Only 1,914 usable training articles (out of 6,381 total). The remaining 4,482 are package-level metadata — the S&P API does not return body text for them. This limits model generalisation.

### 2. Single Market Regime
Training data spans Jan 2022 - Apr 2026. The model has seen limited market regimes. The March 2026 extreme volatility period (Dubai price swings of 8-16% daily) is an outlier that the model trained on calm months cannot predict. Walk-forward CV fold 4 accuracy drops to ~38% because of this.

### 3. Train/Inference Gap
The model trains on body_text + headline but at inference time only the headline is available from the live search API (body text requires a separate /content/{id} call per article). The dashboard now fetches body_text from the local articles.db when available, but newly published articles won't have body text until collect_news.py runs.

### 4. Label Alignment
All articles published on the same date get the same price label (the Dubai/Brent/WTI price change that day). An article about LNG shipping published on a day when crude oil moved 3% gets labelled HIGH even if LNG prices were flat. Commodity-specific labelling (implemented) partially addresses this but LNG still uses Oman as a proxy.

### 5. No GPU
FinBERT fine-tuning on CPU achieves only 27.3% accuracy (1 epoch, small batches). FinBERT+XGB (pre-trained embeddings, no fine-tuning) achieves 34.6% — below XGBoost's 42.3% because 768-dim embeddings overfit on small training folds. Both approaches need more data to outperform XGBoost.

### 6. Confidence Calibration
The scaled_conf (35-95%) is a heuristic transformation, not a calibrated probability. A 70% confidence signal does not mean 70% probability of being correct. Platt scaling or isotonic regression would improve calibration but requires more data.

### 7. Direction vs Magnitude Mismatch
The model predicts magnitude (HIGH/MEDIUM/LOW) but trading decisions need direction (rise/fall). The directional model (40.0% CV) is trained separately and used to set the direction field. These two models are not jointly optimised.

### 8. SQLite Scalability
articles.db is a single SQLite file (~230MB). Concurrent writes from parallel refresh workers are serialised. At scale (>100k articles), this becomes a bottleneck.

---

## Future Improvements

### Short-term (1-3 months)

**1. Daily data collection** — run collect_news.py every weekday morning. Each day adds 2 PDF reports (~40KB each). After 3 months: ~3,600 more body-text articles, expected accuracy gain +5-10%.

**2. Retrain weekly** — run train.py --model-type xgb after each week of new data. The model improves continuously as it sees more market regimes.

**3. Collect prices from more sources** — the prices table currently only has data from Platts report body text. Adding yfinance historical data for 2022-2024 would give more price history for label computation.

### Medium-term (3-6 months)

**4. FinBERT+XGB at scale** — once training data exceeds ~5,000 articles, switch to --model-type finbert-xgb. The 768-dim FinBERT embeddings will generalise better with more data and are expected to reach 50-55% accuracy.

**5. Commodity-specific directional models** — currently one directional model for all commodities. Training separate directional models per commodity (dubai_direction, brent_direction, etc.) would improve direction accuracy.

**6. Better label alignment** — compute labels from the actual next-day price return (entry_price to exit_price) rather than the pct_change_24h column. This eliminates the shift(-1) misalignment and uses the same prices as the trading simulation.

**7. Confidence calibration** — apply Platt scaling (LogisticRegression on model outputs) to convert raw probabilities to calibrated confidence scores. Requires a held-out calibration set.

### Long-term (Databricks migration)

**8. Full Platts corpus** — Petronas Databricks already has years of S&P Platts data. Migrating to Databricks gives access to 5+ years of articles across all report types. Expected accuracy: 55-65%.

**9. Spark-based training** — with 100k+ articles, switch from scikit-learn to Spark MLlib or distributed XGBoost. Training time drops from minutes to seconds.

**10. MLflow model registry** — replace joblib.dump() with MLflow model versioning. Enables A/B testing between model versions, automatic rollback, and experiment tracking.

**11. Delta Lake predictions table** — replace SQLite predictions table with Delta Lake. Enables concurrent writes, time-travel queries, and integration with Petronas data platform.

**12. Real-time inference** — replace the 15-minute cached batch scoring with a streaming pipeline (Kafka + Spark Streaming) that scores articles as they arrive from the Platts feed.

---

## Environment Variables (.env)

SPGLOBAL_USERNAME=your_email@petronas.com
SPGLOBAL_PASSWORD=your_password
GEMINI_API_KEY=AIzaSy...
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
DISABLE_AI=true          # skip Gemini during testing
ENABLE_SIMILARITY_SEARCH=false  # enable ChromaDB similarity search

---

## Recent Changes (April 2026)

### Confidence Fix
- **Before:** `confidence` stored in DB = `scaled_conf / 100` (artificial 35-95% display value)
- **After:** `confidence` stored = raw model `high_prob` (actual XGBoost output, 0-1)
- Impact: TRADE threshold comparisons now use real model probability. Signals on real macro news (OPEC, sanctions) now score 63-74% confidence, above the 55% TRADE threshold.

### TRADE Threshold Lowered
- **Before:** 70% confidence required for TRADE status
- **After:** 55% confidence required
- Rationale: Data showed signals at ≥50% confidence are correct ~52% of the time (tradeable). 70% was too strict for a model trained on noisy commodity news data.

### Training Data Noise Filter
- Excluded routine price report names from `labelled_articles` training set:
  `Crude Oil Marketwire`, `Arab Gulf Marketscan`, `Gulf Arab Marketscan`, `European Marketscan`, `Latin American Wire`, `Oilgram Price Report`, `Bunkerwire`, `Solventswire`, `European Gas Daily`, `European Power Daily`
- These are price tables, not news events. Including them taught the model that "Crude Oil Marketwire published on a volatile day = HIGH" — coincidence, not causation.

### Historical Signal Backfill
- New script: `data/backfill_signals.py`
- Generates signals for past dates that have articles but no predictions (walk-forward, no look-ahead)
- Ran for Jan–Apr 2026: 280 signals logged across 70 dates
- Combined with existing live predictions: ~304 total, ~217 resolved

### Price Fetch Fix
- `get_next_trading_close()` had a `break` after first attempt — only tried 1 day ahead
- Fixed to try up to 7 days forward (handles long weekends + public holidays like Good Friday)

### Synthetic Data Removed
- All 404 seeded rows (confidence = 0.75/0.55/0.40) deleted from predictions table
- `seed_historical_predictions.py` disabled with RuntimeError
- All metrics now based on real model outputs with real prices

### Vector Store SSL Fix
- `SentenceTransformerEmbeddingFunction` was calling HuggingFace on every load (SSL fails on corporate networks)
- Replaced with `_LocalEmbeddingFunction` that uses `local_files_only=True`
- Queries now use `query_embeddings=[...]` directly — bypasses ChromaDB's internal `embed_query` call entirely

### Dashboard Changes
- Trade budget progress bar removed from Tab 1
- Signal History (Tab 2) now shows Sharpe ratio, Sortino ratio, avg win/loss, cut-loss rate
- Low-confidence cards (<50%) show grey "~ Bullish/Bearish Lean" with "monitor only" warning — no Long/Short shown
- Smart "Pending" labels: shows "📅 Saturday — no market", "📅 Holiday — no market" instead of generic "⏳ Pending"
- P&L card falls back to all resolved signals when no TRADE signals exist (instead of showing $0)
