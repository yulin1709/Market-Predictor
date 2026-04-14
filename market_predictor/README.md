# AI Market Impact Predictor

Analyses S&P Global Platts energy news to generate **Bullish / Bearish / Neutral** price signals for Dubai Crude, Brent, WTI, and LNG — built for **fundamental traders and market analysts** at Petronas Trading Digital.

> This is NOT a technical trading system. No RSI, MACD, or chart patterns. Signals are driven by fundamental catalysts: OPEC decisions, supply disruptions, sanctions, geopolitical events.

---

## Data Integrity Policy

**No synthetic data.** All predictions use real model outputs with real entry/exit prices from yfinance.

**Confidence thresholds (display):**
| Raw model `high_prob` | Displayed confidence | Card style |
|---|---|---|
| < 40% | 35–49% | Grey "~ Bullish/Bearish Lean" — monitor only |
| 40–60% | 50–70% | Full colour Bullish/Bearish |
| 60–100% | 70–95% | Full colour + TRADE eligible |

**TRADE action threshold:** confidence ≥ 55% (raw `high_prob` ≥ ~0.46)

---

## What It Does

Every trading day, the system:

1. Fetches S&P Global Platts articles + enriches with recent high-quality DB articles (body text > 500 chars)
2. Falls back to GDELT public news (Reuters/AP/Bloomberg) when S&P only has price tables
3. Scores articles using commodity-specific XGBoost classifiers — uses **top-5 by confidence** to avoid dilution by empty package articles
4. Generates Bullish / Bearish / Neutral signals with real model confidence
5. Tracks signal accuracy and P&L against real prices
6. Writes AI market commentary via Gemini 2.5 Flash

---

## Dashboard Tabs

| Tab | What it shows |
|---|---|
| 📡 Today's Signal | Live signal cards. Grey when confidence <50% |
| 📊 Signal History | Real predictions vs actual price moves. Accuracy, P&L, Sharpe, Sortino |
| 📈 Price Chart | Entry vs exit price chart |
| 🎯 Entry Optimizer | Auto-selects best trade setup (highest EV) |
| 🔍 Comparable Events | ChromaDB semantic search — past similar events |
| 📋 Accuracy Report | ML model offline evaluation |
| 🎯 Prediction Accuracy | Confusion matrix, rolling accuracy |

---

## How It Works

### Signal Computation
1. **Fetch articles** — S&P API + DB quality articles (non-price-table, last 7 days, body > 500 chars)
2. **Score all articles** — XGBoost general model → `high_prob` per article
3. **Filter by commodity keywords** — relevant articles per commodity
4. **Top-5 confidence** — average `high_prob` of top-5 articles (prevents dilution by empty packages)
5. **Direction** — directional model votes → Bullish/Bearish/Neutral
6. **Scale confidence** — linear mapping: `high_prob` 0.35→47%, 0.55→65%, 0.73→78%

### Training
- XGBoost + TF-IDF + numeric features (momentum, day-of-week, article count)
- Walk-forward CV (4 folds, expanding window) — no data leakage
- Excludes routine price report names from training labels (noise filter)
- Commodity-specific models: Dubai, Brent, WTI, LNG

### Price Fetching
- Two-tier: Platts local DB → yfinance fallback
- 7-day lookahead for holidays/weekends
- `articles_deploy.db` (0.4MB) committed for Streamlit Cloud deployment

---

## Project Structure

```
market_predictor/
├── .env                          # API keys (never commit)
├── .env.example
├── requirements.txt
├── run_daily.bat                 # Windows Task Scheduler daily pipeline
│
├── data/
│   ├── auth.py                   # S&P Global token refresh
│   ├── collect_news.py           # Fetch S&P Platts articles
│   ├── collect_prices.py         # Download price history
│   ├── fetch_price.py            # Two-tier price fetch (7-day lookahead)
│   ├── fetch_platts_price.py     # Platts local DB price lookup
│   ├── align_and_label.py        # Label articles (excludes noise reports)
│   ├── backfill_actuals.py       # Fill entry/exit prices
│   ├── backfill_signals.py       # Generate signals for past dates
│   ├── db_path.py                # DB path helper (articles.db → articles_deploy.db fallback)
│   ├── ticker_config.py          # Commodity config, stop-loss %, event badges
│   ├── init_tracker.py           # DB schema migration
│   └── articles_deploy.db        # Minimal DB for Streamlit Cloud (308 predictions + prices)
│
├── features/
│   └── extract_entities.py       # Keyword features + impact scores
│
├── models/
│   ├── train.py
│   └── saved/                    # Pre-trained models (committed)
│
├── inference/
│   ├── pipeline.py               # predict() + log_prediction()
│   ├── trade_filter.py           # Monthly budget + quality gate
│   ├── tracking.py               # P&L simulation
│   ├── metrics.py                # Risk metrics
│   ├── evaluation.py             # Offline ML evaluation
│   ├── vector_store.py           # ChromaDB (SSL-safe, local model cache)
│   ├── explain.py                # Gemini commentary
│   ├── optimizer.py              # Entry timing optimizer
│   └── validate_predictions.py   # Data quality checks
│
├── utils/
│   └── gemini.py
│
├── api/
│   └── main.py                   # FastAPI REST endpoint
│
└── app/
    └── streamlit_app.py          # 7-tab Streamlit dashboard
```

---

## Setup

```powershell
# Activate venv
.venv\Scripts\activate

# First-time setup
python market_predictor/data/collect_prices.py
python market_predictor/data/collect_news.py
python market_predictor/data/align_and_label.py
python market_predictor/models/train.py
python market_predictor/inference/vector_store.py
streamlit run market_predictor/app/streamlit_app.py

# Daily workflow (or use run_daily.bat via Task Scheduler at 08:00 MYT)
python market_predictor/data/backfill_actuals.py
python market_predictor/data/collect_news.py --days 2 --max-pages 2

# Backfill historical signals
python market_predictor/data/backfill_signals.py --days 90
python market_predictor/data/backfill_actuals.py

# Weekly retrain
python market_predictor/data/align_and_label.py
python market_predictor/models/train.py
```

---

## Environment Variables

```env
SPGLOBAL_USERNAME=your_email@petronas.com
SPGLOBAL_PASSWORD=your_password
GEMINI_API_KEY=AIzaSy...
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
DISABLE_AI=false
```

For Streamlit Cloud: add these in Settings → Secrets.

---

## Streamlit Cloud Deployment

- Main file path: `market_predictor/app/streamlit_app.py`
- `articles_deploy.db` (0.4MB) is committed — contains 308 predictions + 3,770 price rows
- On local dev: uses `articles.db` (full DB, gitignored)
- On Cloud: auto-falls back to `articles_deploy.db`

---

## Current Performance (April 2026)

| Metric | Value |
|---|---|
| Total predictions | ~308 (Jan–Apr 2026) |
| Resolved | ~217 |
| Direction accuracy (all) | ~56% |
| Accuracy (conf ≥55%) | ~58% |
| P&L (conf ≥55% trades) | +$90,773 |
| April 7 loss | -$51,746 (Trump tariff shock) |

---

## Limitations

- Model accuracy ~36–42% on held-out test data
- S&P Platts API returns price report packages, not news articles — GDELT fallback used for real news
- Price data uses yfinance proxies (CL=F, BZ=F, NG=F), not exact Platts assessments
- No intraday data — entry/exit are daily closes
- Low-confidence days (routine price reports) show grey "monitor only" cards — correct behaviour

## Future Improvements

- Connect to Databricks with live Platts price feed
- FinBERT fine-tuning on Petronas corpus
- Multi-day hold period (currently fixed 24h)
- Accumulate 6+ months of live signals for statistically meaningful metrics
