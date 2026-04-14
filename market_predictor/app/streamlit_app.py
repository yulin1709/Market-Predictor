"""AI Market Impact Predictor — Petronas Trading Digital (6-tab dashboard)."""
import os
import sys
import sqlite3
import subprocess
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Streamlit Cloud secrets → os.environ (so existing os.getenv() calls work) ─
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass  # running locally without secrets.toml — .env already loaded above

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Market Impact Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
# Use deploy DB on Streamlit Cloud (articles.db is too large to commit)
_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
_full_db   = os.path.join(_data_dir, "articles.db")
_deploy_db = os.path.join(_data_dir, "articles_deploy.db")
DB_PATH = _full_db if os.path.exists(_full_db) else _deploy_db
SEARCH_URL = "https://api.ci.spglobal.com/news-insights/v1/search"
QUERIES = ["crude oil", "OPEC", "LNG", "natural gas", "oil price", "energy market"]

COMMODITY_COLORS = {
    "Dubai Crude (PCAAT00)": "#E8593C",
    "Brent Dated (PCAAS00)": "#378ADD",
    "WTI (PCACG00)": "#639922",
    "LNG / Nat Gas (JKM)": "#BA7517",
}
SIGNAL_COLORS = {"Bullish": "#00C853", "Neutral": "#FFB300", "Bearish": "#F44336"}
COMMODITY_KEYWORDS = {
    "Dubai Crude (PCAAT00)": ["dubai","oman","saudi","opec","middle east","gulf","crude","iran","iraq","uae","kuwait","qatar","barrel"],
    "Brent Dated (PCAAS00)": ["brent","north sea","europe","russia","urals","norway","uk","crude oil","barrel"],
    "WTI (PCACG00)": ["wti","cushing","texas","shale","permian","us crude","american","gulf of mexico","refinery"],
    "LNG / Nat Gas (JKM)": ["lng","natural gas","liquefied","tanker","cargo","freight","jkm","asia","pacific","shipping","vessel","hormuz"],
}
PLOTLY_DEFAULTS = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=12),
    margin=dict(l=0, r=0, t=24, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
COMMODITIES = [
    ("Dubai Crude (PCAAT00)", "PCAAT00"),
    ("Brent Dated (PCAAS00)", "PCAAS00"),
    ("WTI (PCACG00)", "PCACG00"),
    ("LNG / Nat Gas (JKM)", "JKM"),
]
COMM_NAMES = [c[0] for c in COMMODITIES]
BULL_WORDS = ["rise","rises","surge","gain","rally","higher","up","increase","tight","cut","outage",
              "disruption","shortage","sanction","attack","conflict","opec cut"]
BEAR_WORDS = ["fall","falls","drop","decline","lower","down","decrease","weak","surplus","oversupply",
              "build","restart","ease","glut","slow","demand falls"]

# ── DB helpers ────────────────────────────────────────────────────────────────
def query_db(sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def ensure_predictions_table():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        from data.init_tracker import init_tracker
        init_tracker(conn)
        conn.close()
    except Exception:
        pass  # DB may not exist on first deploy — will be created on first write


ensure_predictions_table()

# ── GDELT fallback news fetch ─────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gdelt_news() -> list[dict]:
    """
    Fetch energy headlines from GDELT 2.0 Doc API — free, no key required.
    Used as fallback when S&P articles are unavailable or all low-impact.
    Returns articles in the same format as fetch_news().
    """
    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    queries = [
        "OPEC oil production",
        "crude oil price",
        "LNG natural gas",
        "oil supply disruption",
        "energy sanctions",
    ]
    seen: set = set()
    articles: list = []
    today = date.today().isoformat()

    for q in queries:
        try:
            r = requests.get(
                GDELT_URL,
                params={
                    "query": q,
                    "mode": "artlist",
                    "maxrecords": 10,
                    "timespan": "1d",
                    "sort": "DateDesc",
                    "format": "json",
                },
                timeout=15,
            )
            r.raise_for_status()
            for item in r.json().get("articles", []):
                url = item.get("url", "")
                hl = (item.get("title") or "").strip()
                if not hl or url in seen:
                    continue
                seen.add(url)
                pub = str(item.get("seendate", today))[:8]
                pub_iso = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}" if len(pub) >= 8 else today
                articles.append({
                    "id": f"gdelt-{abs(hash(url))}",
                    "headline": hl,
                    "body_text": "",
                    "published": pub_iso,
                    "url": url,
                    "date": pub_iso,
                    "source": "GDELT",
                })
        except Exception:
            continue

    return articles


# ── S&P News fetch ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3000)
def _get_token() -> str:
    u = os.getenv("SPGLOBAL_USERNAME", "")
    p = os.getenv("SPGLOBAL_PASSWORD", "")
    if not u or not p:
        return ""
    try:
        r = requests.post(
            "https://api.ci.spglobal.com/auth/api",
            headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={"username": u, "password": p},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("access_token", "")
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news() -> list[dict]:
    """Fetch energy news from S&P Global Platts API, enriched with body_text from local DB.
    Falls back to most recent available articles when today has none (weekends/holidays)."""
    token = _get_token()
    today = date.today().isoformat()
    seen: list = []
    articles: list = []

    if token:
        for query in QUERIES:
            try:
                r = requests.get(
                    SEARCH_URL,
                    headers={"Authorization": f"Bearer {token}", "accept": "application/json"},
                    params={"q": query, "pageSize": 15, "page": 1},
                    timeout=20,
                )
                r.raise_for_status()
                for item in r.json().get("results", []):
                    aid = item.get("id", "")
                    hl = (item.get("headline") or item.get("title") or "").strip()
                    if not hl or aid in seen:
                        continue
                    seen.append(aid)
                    articles.append({
                        "id": aid,
                        "headline": hl,
                        "published": str(item.get("updatedDate", ""))[:19],
                        "url": item.get("documentUrl", ""),
                        "date": str(item.get("updatedDate", ""))[:10],
                        "body_text": "",
                    })
            except Exception:
                continue

    # Enrich with body_text from local DB
    if articles:
        try:
            import sqlite3 as _sq
            conn = _sq.connect(DB_PATH)
            ids = [a["id"] for a in articles if a["id"]]
            if ids:
                placeholders = ",".join("?" * len(ids))
                rows = conn.execute(
                    f"SELECT id, body_text FROM articles WHERE id IN ({placeholders})", ids
                ).fetchall()
                body_map = {r[0]: r[1] or "" for r in rows}
                for a in articles:
                    a["body_text"] = body_map.get(a["id"], "")
            conn.close()
        except Exception:
            pass

    # Filter to today's articles
    todays = [a for a in articles if a["date"] == today]

    # If no articles from API for today, load from local DB (handles holidays/weekends)
    if len(todays) < 5:
        try:
            import sqlite3 as _sq
            from datetime import timedelta
            conn = _sq.connect(DB_PATH)
            # First try: find recent date with articles that have real body text (>500 chars)
            for days_back in range(0, 14):
                check_date = (date.today() - timedelta(days=days_back)).isoformat()
                db_rows = conn.execute("""
                    SELECT id, headline, body_text, published_at, url
                    FROM articles
                    WHERE date(published_at) = ?
                      AND LENGTH(COALESCE(body_text,'')) > 500
                    ORDER BY LENGTH(body_text) DESC
                    LIMIT 30
                """, (check_date,)).fetchall()
                if len(db_rows) >= 2:
                    db_articles = [{
                        "id": r[0], "headline": r[1] or "",
                        "body_text": r[2] or "",
                        "published": str(r[3] or "")[:19],
                        "url": r[4] or "",
                        "date": check_date,
                    } for r in db_rows if r[1]]
                    conn.close()
                    return db_articles
            # Fallback: any date with ≥5 articles
            for days_back in range(0, 8):
                check_date = (date.today() - timedelta(days=days_back)).isoformat()
                db_rows = conn.execute("""
                    SELECT id, headline, body_text, published_at, url
                    FROM articles
                    WHERE date(published_at) = ?
                    ORDER BY published_at DESC
                    LIMIT 50
                """, (check_date,)).fetchall()
                if len(db_rows) >= 5:
                    db_articles = [{
                        "id": r[0], "headline": r[1] or "",
                        "body_text": r[2] or "",
                        "published": str(r[3] or "")[:19],
                        "url": r[4] or "",
                        "date": check_date,
                    } for r in db_rows if r[1]]
                    conn.close()
                    return db_articles
            conn.close()
        except Exception:
            pass

    return todays if todays else articles


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_with_fallback() -> tuple[list[dict], str]:
    """
    Returns (articles, source) where source is 'S&P Global Platts', 'S&P (DB cache)', or 'GDELT (fallback)'.
    Always enriches with the best recent DB articles (body text > 500 chars) for better scoring.
    Falls back to GDELT when S&P only has price tables.
    """
    sp_articles = fetch_news()
    today = date.today().isoformat()
    today_sp = [a for a in sp_articles if a.get("date") == today]

    _noise_names = {
        "crude oil marketwire", "arab gulf marketscan", "gulf arab marketscan",
        "european marketscan", "latin american wire", "oilgram price report",
        "bunkerwire", "energy trader", "gas daily", "megawatt daily",
        "us marketscan", "rec daily", "asia-pacific - arab gulf marketscan",
    }

    def _is_real_news(a: dict) -> bool:
        hl = str(a.get("headline") or "").strip().lower()
        body = str(a.get("body_text") or "")
        if len(body) > 500:
            return True
        if len(hl) > 40 and not any(noise in hl for noise in _noise_names):
            return True
        return False

    real_sp = [a for a in today_sp if _is_real_news(a)]

    # Always pull the best recent DB articles (last 7 days, body text > 500 chars)
    # These contain real market analysis even on holiday/weekend days
    db_quality_articles = []
    try:
        import sqlite3 as _sq
        conn = _sq.connect(DB_PATH, timeout=10)
        db_rows = conn.execute("""
            SELECT id, headline, body_text, published_at, url
            FROM articles
            WHERE LENGTH(COALESCE(body_text,'')) > 500
              AND date(published_at) >= date('now', '-7 days')
              AND LOWER(COALESCE(headline,'')) NOT LIKE '%marketwire%'
              AND LOWER(COALESCE(headline,'')) NOT LIKE '%marketscan%'
              AND LOWER(COALESCE(headline,'')) NOT LIKE '%oilgram%'
              AND LOWER(COALESCE(headline,'')) NOT LIKE '%bunkerwire%'
            ORDER BY published_at DESC
            LIMIT 10
        """).fetchall()
        conn.close()
        db_quality_articles = [{
            "id": r[0], "headline": r[1] or "",
            "body_text": r[2] or "",
            "published": str(r[3] or "")[:19],
            "url": r[4] or "",
            "date": str(r[3] or "")[:10],
        } for r in db_rows if r[1]]
    except Exception:
        pass

    # Merge: SP articles + DB quality articles (deduplicated)
    sp_ids = {a["id"] for a in sp_articles}
    extra = [a for a in db_quality_articles if a["id"] not in sp_ids]
    merged_base = sp_articles + extra

    if len(real_sp) >= 3 or len(db_quality_articles) >= 2:
        source = "S&P Global Platts" if len(real_sp) >= 3 else "S&P + DB cache"
        return merged_base, source

    # S&P only has price tables and no recent DB articles — try GDELT
    gdelt = fetch_gdelt_news()
    today_gdelt = [a for a in gdelt if a.get("date") == today]

    if len(today_gdelt) >= 3:
        merged = today_gdelt + [a for a in merged_base if a.get("id") not in {g["id"] for g in today_gdelt}]
        return merged, "GDELT (fallback)"

    return merged_base, "S&P (DB cache)"


# ── ML model loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model...")
def _load_predict_fn():
    try:
        from inference.pipeline import predict, _load_models
        _load_models()
        return predict
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


# ── Signal computation ────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def compute_today_signals(articles_json: str) -> list[dict]:
    """Score articles and compute per-commodity signals. Returns list of commodity signal dicts."""
    articles = json.loads(articles_json)
    predict_fn = _load_predict_fn()

    # Support both old format (list of strings) and new format (list of dicts with text+headline)
    def _get_text(a) -> str:
        if isinstance(a, dict):
            return a.get("text") or a.get("headline") or ""
        return str(a)

    def _get_headline(a) -> str:
        if isinstance(a, dict):
            return a.get("headline") or a.get("text") or ""
        return str(a)

    # Pre-score all articles with the general model
    scored: list[dict] = []
    if predict_fn and articles:
        for a in articles:
            text = _get_text(a)
            headline = _get_headline(a)
            try:
                r = predict_fn(text, skip_explanation=True)
                scored.append({"headline": headline, "text": text, **r})
            except Exception:
                scored.append({
                    "headline": headline, "text": text,
                    "label": "LOW", "score": 0,
                    "high_prob": 0, "medium_prob": 0, "low_prob": 1,
                    "entities": {}, "similar_events": [],
                })

    out: list[dict] = []
    for comm, ticker in COMMODITIES:
        kws = COMMODITY_KEYWORDS[comm]
        rel_articles = [a for a in scored
                        if sum(1 for k in kws if k in a.get("headline", "").lower()) >= 1]
        rel_texts = [a["text"] for a in rel_articles]

        # Re-score relevant articles with the commodity-specific model
        rel: list[dict] = []
        if predict_fn and rel_texts:
            for a in rel_articles:
                try:
                    r = predict_fn(a["text"], skip_explanation=True, commodity=comm)
                    rel.append({**a, **r})
                except Exception:
                    rel.append(a)
        else:
            rel = rel_articles

        pool = rel if rel else scored

        if not pool:
            out.append({
                "commodity": comm, "ticker": ticker,
                "signal": "LOW", "direction": "flat",
                "strength": "Weak", "bias": "Hold",
                "scaled_conf": 35.0, "move": "<1%",
                "n_relevant": 0, "top_headline": "",
            })
            continue

        # Use top-5 articles by high_prob for confidence — avoids dilution by empty/noise articles
        pool_sorted = sorted(pool, key=lambda a: a.get("high_prob", 0), reverse=True)
        top_pool = pool_sorted[:5] if len(pool_sorted) >= 5 else pool_sorted
        conf = sum(a.get("high_prob", 0) for a in top_pool) / len(top_pool) if top_pool else 0
        bull = sum(sum(1 for w in BULL_WORDS if w in a.get("headline", "").lower()) for a in pool)
        bear = sum(sum(1 for w in BEAR_WORDS if w in a.get("headline", "").lower()) for a in pool)
        sup = sum(1 for a in pool if a.get("entities", {}).get("supply_impact", False))
        bull += sup * 2

        # Keyword heuristic → bullish/bearish/neutral
        direction_keyword = "bullish" if bull > bear * 1.3 else ("bearish" if bear > bull * 1.3 else "neutral")

        # Prefer directional model output (already mapped to bullish/bearish/neutral)
        direction_model_votes = [a.get("direction_from_model") for a in pool
                                 if a.get("direction_from_model")]
        if direction_model_votes:
            from collections import Counter
            direction = Counter(direction_model_votes).most_common(1)[0][0]
            # Normalise legacy values
            _dmap = {"rise": "bullish", "fall": "bearish", "flat": "neutral"}
            direction = _dmap.get(direction, direction)
        else:
            direction = direction_keyword

        label = "HIGH" if conf >= 0.30 else ("MEDIUM" if conf >= 0.15 else "LOW")
        move = ">3%" if label == "HIGH" else ("1-3%" if label == "MEDIUM" else "<1%")

        n_rel = len(rel)
        article_factor = min(n_rel / 15.0, 1.0)
        label_factor = 1.4 if label == "HIGH" else (1.2 if label == "MEDIUM" else 1.0)
        dir_factor = 1.0 if direction != "neutral" else 0.6
        scaled_conf = min(max(conf * 4.5 * label_factor * dir_factor * (0.7 + article_factor * 0.3), 35), 95)

        # Only show directional bias if confidence is meaningful (≥50%)
        # Below 50% = model is uncertain, don't suggest Long/Short
        if direction == "bullish":
            strength = "Strong" if scaled_conf >= 70 else ("Moderate" if scaled_conf >= 50 else "Weak")
            if scaled_conf >= 50:
                bias = "Bullish — Consider Long"
            else:
                bias = "Bullish lean — Low confidence, monitor only"
        elif direction == "bearish":
            strength = "Strong" if scaled_conf >= 70 else ("Moderate" if scaled_conf >= 50 else "Weak")
            if scaled_conf >= 50:
                bias = "Bearish — Consider Short"
            else:
                bias = "Bearish lean — Low confidence, monitor only"
        else:
            strength = "Weak"
            bias = "Neutral — Hold"

        # Compute top event type and impact score from pool
        from features.extract_entities import get_event_type, get_impact_score
        top_event_type = "other"
        top_impact_score = 2
        if pool:
            top_art = sorted(pool, key=lambda x: x.get("high_prob", 0), reverse=True)[0]
            top_event_type = get_event_type(top_art.get("headline", ""))
            top_impact_score = get_impact_score(top_event_type)

        top_sorted = sorted(
            [a for a in pool if sum(1 for k in kws if k in a.get("headline", "").lower()) >= 1],
            key=lambda x: x.get("high_prob", 0), reverse=True,
        )
        top_headline = top_sorted[0]["headline"] if top_sorted else (pool[0]["headline"] if pool else "")

        out.append({
            "commodity": comm, "ticker": ticker,
            "signal": label, "direction": direction,
            "strength": strength, "bias": bias,
            "scaled_conf": round(scaled_conf, 1),
            "move": move, "n_relevant": n_rel,
            "top_headline": top_headline,
            "event_type": top_event_type,
            "impact_score": top_impact_score,
        })
    return out


# ── Auto-collect: fetch today's articles if DB is stale ──────────────────────
def _auto_collect_if_stale() -> bool:
    """
    Returns True if collection was triggered.
    Runs collect_news.py in the background when today has < 5 articles.
    Only runs once per session and only on weekdays.
    """
    if st.session_state.get("_auto_collected"):
        return False
    st.session_state["_auto_collected"] = True  # mark immediately to avoid double-run

    today = date.today()
    # Skip weekends — no Platts reports published
    if today.weekday() >= 5:
        return False

    try:
        import sqlite3 as _sq
        _conn = _sq.connect(DB_PATH, timeout=10)
        _conn.execute("PRAGMA journal_mode=WAL")
        _today_count = _conn.execute(
            "SELECT COUNT(*) FROM articles WHERE date(published_at)=?",
            (today.isoformat(),)
        ).fetchone()[0]
        _conn.close()
    except Exception:
        _today_count = 0

    if _today_count >= 5:
        return False  # already have today's articles

    # Trigger background collection — delayed 3s to avoid racing with page load DB writes
    _collect_script = os.path.join(os.path.dirname(__file__), "..", "data", "collect_news.py")
    _python = sys.executable
    try:
        # Use a small wrapper so we can delay start without blocking
        subprocess.Popen(
            [_python, "-c",
             f"import time; time.sleep(3); import subprocess; subprocess.run([r'{_python}', r'{_collect_script}', '--days', '2', '--max-pages', '2'])"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


_was_stale = _auto_collect_if_stale()

# ── Session state bootstrap ───────────────────────────────────────────────────
if "articles" not in st.session_state:
    _arts, _news_source = fetch_news_with_fallback()
    st.session_state.articles = _arts
    st.session_state.news_source = _news_source

# Article count check — must happen before prediction
if "article_count" not in st.session_state:
    from inference.pipeline import get_article_count_today, MIN_ARTICLES_TO_PREDICT
    _ac = get_article_count_today()  # returns today's OR yesterday's count
    st.session_state.article_count = _ac
    st.session_state.no_news_today = _ac < MIN_ARTICLES_TO_PREDICT

if "today_signals" not in st.session_state:
    st.session_state.today_signals = compute_today_signals(
        json.dumps([
            {
                "headline": a["headline"],
                "text": (a["headline"] + " " + (a.get("body_text") or "")[:400]).strip()
            }
            for a in st.session_state.articles
        ])
    )

if "gemini_commentary" not in st.session_state:
    if (not st.session_state.get("no_news_today", False)
            and os.getenv("DISABLE_AI", "").lower() not in ("true", "1", "yes")):
        try:
            from utils.gemini import generate_text
            top_hl = [s["top_headline"] for s in st.session_state.today_signals if s.get("top_headline")][:4]
            dom = max(st.session_state.today_signals, key=lambda x: x["scaled_conf"])
            prompt = (
                f"Energy analyst. Signals: {dom['signal']} {dom['direction']} for {dom['commodity']}. "
                f"Headlines: {'; '.join(top_hl)}. "
                "3 sentences: what drives this, which commodity most affected, what to watch. Professional."
            )
            st.session_state.gemini_commentary = generate_text(prompt, max_tokens=150)
        except Exception:
            st.session_state.gemini_commentary = ""
    else:
        st.session_state.gemini_commentary = ""

if "logged_today" not in st.session_state:
    from inference.pipeline import log_prediction, MIN_ARTICLES_TO_PREDICT
    _ac = st.session_state.get("article_count", 0)
    if _ac >= MIN_ARTICLES_TO_PREDICT:
        import time as _time
        for sig in st.session_state.today_signals:
            for _attempt in range(3):  # retry up to 3x on lock
                try:
                    log_prediction(sig["commodity"], sig["ticker"], sig,
                                   sig.get("top_headline"), article_count=_ac)
                    break
                except Exception as e:
                    if "locked" in str(e).lower() and _attempt < 2:
                        _time.sleep(1.5)
                    else:
                        print(f"[app] log_prediction failed: {e}")
                        break
    else:
        print(f"[app] Skipping log_prediction — only {_ac} articles today")
    st.session_state.logged_today = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Market Impact Predictor")
    st.caption("Petronas Trading Digital")
    st.divider()

    with st.expander("About this dashboard", expanded=False):
        st.markdown("""
**AI Market Impact Predictor**
Built for: **Fundamental traders & market analysts**

This tool analyses S&P Global Platts news articles to identify market-moving events and generate directional price signals for:
- Dubai Crude (PCAAT00)
- Brent Dated (PCAAS00)
- WTI (PCACG00)
- LNG / JKM

**What it does:**
- Identifies HIGH-impact fundamental catalysts (OPEC, sanctions, outages)
- Limits recommendations to ~8 trades/month (quality over quantity)
- Provides Bullish/Bearish signals with thesis invalidation levels
- Tracks signal accuracy and risk-adjusted returns

**What it does NOT do:**
- Technical analysis (no chart patterns, RSI, MACD)
- Intraday trading signals
- Position sizing for leveraged products

*Powered by XGBoost + S&P Global Platts + Gemini*
        """)

    st.divider()
    try:
        import joblib
        mi = joblib.load(Path(__file__).parent.parent / "models" / "saved" / "model_info.pkl")
        st.caption(f"**Model:** {mi.get('model_type', 'unknown')}")
        st.caption(f"**Trained:** {mi.get('trained_at', '—')}")
        if mi.get("accuracy"):
            st.caption(f"**Val accuracy:** {mi['accuracy']*100:.1f}%")
    except Exception:
        st.caption("Model info unavailable")
    st.divider()
    show_raw_confidence = st.toggle("Show raw confidence scores", value=False)
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        for key in ["articles", "today_signals", "gemini_commentary", "logged_today", "news_source"]:
            st.session_state.pop(key, None)
        st.cache_data.clear()
        st.rerun()
    if st.button("▶ Run Backfill (fill exit prices)", use_container_width=True):
        result = subprocess.run(
            [sys.executable, "market_predictor/data/backfill_actuals.py"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            st.success("Backfill complete")
        else:
            st.error(result.stderr[:200])
    st.divider()
    if st.button("🔍 Run Data Validation", use_container_width=True):
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from inference.validate_predictions import run_all_checks
            _vresults = run_all_checks()
            for _vr in _vresults:
                _icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(_vr.status, "?")
                _msg = f"{_icon} **Check {_vr.check_id}: {_vr.name}**  \n{_vr.detail}"
                if _vr.status == "PASS":
                    st.success(_msg)
                elif _vr.status == "WARN":
                    st.warning(_msg)
                else:
                    st.error(_msg)
        except Exception as _ve:
            st.error(f"Validation error: {_ve}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ AI Market Impact Predictor")
st.caption(f"Petronas Trading Digital  |  {date.today().strftime('%d %B %Y')}")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📡 Today's Signal",
    "📊 Prediction Tracker",
    "📈 Price Chart",
    "🎯 Entry Optimizer",
    "🔍 Comparable Events",
    "📋 Accuracy Report",
    "🎯 Prediction Accuracy",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Today's Signal
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    signals = st.session_state.today_signals
    articles = st.session_state.articles
    _article_count = st.session_state.get("article_count", len(articles))
    _no_news = st.session_state.get("no_news_today", False)
    from inference.pipeline import MIN_ARTICLES_TO_PREDICT as _MIN_ART
    from inference.trade_filter import get_trades_this_month, MAX_TRADES_PER_MONTH
    from data.ticker_config import EVENT_TYPE_BADGES, STOP_LOSS_PCT as _STOP_PCT

    # ── Header row: title ────────────────────────────────────────────────────
    st.subheader(f"Today's Market Signal — {date.today().strftime('%d %B %Y')}")

    # ── No-news warning state ─────────────────────────────────────────────────
    if _was_stale:
        st.info(
            f"📡 Fetching today's S&P Global Platts articles in the background... "
            f"Click **🔄 Refresh Data** in the sidebar in ~30 seconds to load today's signal.",
            icon="⏳",
        )

    if _no_news or _article_count < _MIN_ART:
        st.warning(
            f"**No market signal available for {date.today().strftime('%d %B %Y')}**\n\n"
            f"Only **{_article_count} articles** were fetched from S&P Global Platts today "
            f"(minimum {_MIN_ART} required to generate a reliable signal).\n\n"
            f"**What to do:** Run `python market_predictor/data/collect_news.py` manually."
        )
        _last = query_db("""
            SELECT prediction_date FROM predictions
            WHERE outcome IS NOT NULL OR entry_price IS NOT NULL
            ORDER BY prediction_date DESC LIMIT 1
        """)
        if not _last.empty:
            st.info(f"Last available signal: **{_last.iloc[0]['prediction_date']}** — use the Prediction Tracker tab.")
    else:
        _supply_count = sum(1 for s in signals if s.get("impact_score", 0) >= 7)
        _news_source = st.session_state.get("news_source", "S&P Global Platts")
        _source_badge = (
            "⚠️ GDELT fallback — S&P unavailable today" if "GDELT" in _news_source
            else ("📦 DB cache — showing most recent available" if "cache" in _news_source
                  else "S&P Global Platts")
        )
        _impact_note = f" · {_supply_count} HIGH-impact catalysts" if _supply_count > 0 else " · No HIGH-impact catalysts today — signals based on routine reports"
        st.caption(f"Based on {_article_count} articles · {_source_badge}{_impact_note}")
        if "GDELT" in _news_source:
            st.info(
                "📡 S&P Global Platts has no new articles today — signals are based on **GDELT public news** "
                "(Reuters, AP, Bloomberg wires). Confidence scores may be lower than usual.",
                icon="ℹ️",
            )

        def _article_badge(n: int) -> str:
            if n >= 15:
                return f"<span style='color:#639922;font-size:11px'>● {n} articles</span>"
            elif n >= 5:
                return f"<span style='color:#BA7517;font-size:11px'>⚠ {n} articles</span>"
            else:
                return f"<span style='color:#E24B4A;font-size:11px'>✗ {n} articles — unreliable</span>"

        def _event_badge(event_type: str) -> str:
            if event_type in EVENT_TYPE_BADGES:
                label, color = EVENT_TYPE_BADGES[event_type]
                return (f'<span style="background:{color}22;color:{color};font-size:0.65em;'
                        f'padding:2px 8px;border-radius:10px;border:1px solid {color}44;'
                        f'font-weight:600;letter-spacing:0.5px">{label}</span>')
            return ""

        cols = st.columns(4)
        for col, sig in zip(cols, signals):
            comm = sig["commodity"]
            ticker = sig["ticker"]
            direction = sig["direction"]
            label = sig["signal"]
            sc = sig["scaled_conf"]
            move = sig["move"]
            strength = sig["strength"]
            bias = sig["bias"]
            n = sig["n_relevant"]
            event_type = sig.get("event_type", "other")
            impact_score = sig.get("impact_score", 2)

            # Normalise direction to bullish/bearish/neutral
            _dmap = {"rise": "bullish", "fall": "bearish", "flat": "neutral"}
            direction = _dmap.get(direction, direction)

            # Low confidence (<50%) — mute the card, don't suggest Long/Short
            _low_conf = sc < 50

            if direction == "bullish":
                if _low_conf:
                    sig_icon = "~"; sig_txt = "Bullish Lean"; sig_clr = "#666"
                    sig_bg = "#0d0d0d"; border = "#333"
                else:
                    sig_icon = "▲"; sig_txt = "Bullish"; sig_clr = "#00C853"
                    sig_bg = "#001f0a"; border = "#00C853"
                move_lbl = f"+{move}"
            elif direction == "bearish":
                if _low_conf:
                    sig_icon = "~"; sig_txt = "Bearish Lean"; sig_clr = "#666"
                    sig_bg = "#0d0d0d"; border = "#333"
                else:
                    sig_icon = "▼"; sig_txt = "Bearish"; sig_clr = "#F44336"
                    sig_bg = "#1a0000"; border = "#F44336"
                move_lbl = f"-{move}"
            else:
                sig_icon = "→"; sig_txt = "Neutral"; sig_clr = "#FFB300"
                sig_bg = "#1a1400"; border = "#FFB300"
                move_lbl = "<1%"

            opacity_style = ""

            conf_lbl = "High" if sc >= 70 else ("Medium" if sc >= 50 else "Low — monitor only")
            conf_disp = f"{sc:.0f}% ({conf_lbl})"
            if show_raw_confidence:
                raw = sig.get("confidence", sc)
                conf_disp = f"{sc:.0f}% (raw: {raw:.1f}%)"

            badge = _article_badge(n)
            ev_badge = _event_badge(event_type)

            # Cut loss level — only show for actionable (≥50% confidence) signals
            cut_loss_html = ""
            if label == "HIGH" and direction in ("bullish", "bearish") and not _low_conf:
                sl_pct = _STOP_PCT.get(comm, 1.5)
                cut_loss_html = (
                    f'<div style="margin-top:8px;padding:5px 8px;background:#1a0a0a;'
                    f'border-radius:6px;font-size:0.75em;color:#aaa;text-align:left">'
                    f'<span style="color:#F44336">✂ Thesis invalidation:</span> '
                    f'price moves >{sl_pct}% against position</div>'
                )

            # Low-confidence warning banner inside card
            low_conf_warning = ""
            if _low_conf:
                low_conf_warning = (
                    '<div style="margin-top:8px;padding:5px 8px;background:#1a1a0a;'
                    'border-radius:6px;font-size:0.72em;color:#888;text-align:left">'
                    '⚠ Confidence &lt;50% — do not trade. Monitor only.</div>'
                )

            with col:
                st.markdown(f"""
<div style="background:{sig_bg};border:2px solid {border};border-radius:14px;
padding:18px 14px;text-align:center;min-height:300px;{opacity_style}">
<div style="color:#aaa;font-size:0.78em;font-weight:600">{comm}</div>
<div style="color:#666;font-size:0.7em;margin-bottom:4px">{ticker}</div>
<div style="margin:4px 0">{ev_badge}</div>
<div style="font-size:2.2em;font-weight:bold;color:{sig_clr};margin:6px 0;line-height:1.1">
{sig_icon} {sig_txt}</div>
<div style="color:{sig_clr};font-size:0.82em;margin-bottom:8px">({strength})</div>
<div style="color:#ccc;font-size:0.8em;margin:2px 0">Expected Move: <b>{move_lbl}</b></div>
<div style="color:#ccc;font-size:0.8em;margin:2px 0">Confidence: <b>{conf_disp}</b></div>
<div style="color:#888;font-size:0.72em;margin:2px 0">Impact: {impact_score}/10</div>
<div style="margin-top:8px;padding:5px 0;border-top:1px solid #333">
<b style="color:{sig_clr};font-size:0.85em">{bias}</b></div>
{cut_loss_html}
{low_conf_warning}
<div style="margin-top:6px">{badge}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Supply/Demand Context Panel (Change 6a) ───────────────────────────
        with st.expander("Supply & Demand Context", expanded=False):
            _sc1, _sc2, _sc3 = st.columns(3)
            _supply_kws = ["outage","shutdown","disruption","cut","sanction","opec","pipeline","refinery","tanker","attack","shortage","force majeure"]
            _demand_kws = ["demand","import","consumption","buying","tender","china","india","asia","growth","forecast"]
            _geo_kws = ["iran","russia","ukraine","war","conflict","sanction","geopolit","hormuz","strait","military","tension"]

            supply_items = [a["headline"][:90] for a in articles
                           if any(k in a.get("headline","").lower() for k in _supply_kws)][:5]
            demand_items = [a["headline"][:90] for a in articles
                           if any(k in a.get("headline","").lower() for k in _demand_kws)][:5]
            geo_items = [a["headline"][:90] for a in articles
                        if any(k in a.get("headline","").lower() for k in _geo_kws)][:5]

            with _sc1:
                st.markdown("**Supply signals**")
                if supply_items:
                    for item in supply_items:
                        st.markdown(f"• {item}")
                else:
                    st.caption("No supply disruption signals today")
            with _sc2:
                st.markdown("**Demand signals**")
                if demand_items:
                    for item in demand_items:
                        st.markdown(f"• {item}")
                else:
                    st.caption("No demand signals today")
            with _sc3:
                st.markdown("**Geopolitical risk**")
                if geo_items:
                    for item in geo_items:
                        st.markdown(f"• {item}")
                else:
                    st.caption("No geopolitical signals today")

        # AI Commentary
        commentary = st.session_state.gemini_commentary
        if commentary:
            st.markdown(f"""
<div style="background:#0d1117;border:1px solid #1f6feb;border-radius:10px;
padding:18px 22px;margin:12px 0">
<div style="color:#58a6ff;font-size:0.85em;font-weight:bold;margin-bottom:8px">
🤖 AI Market Commentary (Gemini)</div>
<div style="color:#e6edf3;font-size:1em;line-height:1.6">{commentary}</div>
</div>""", unsafe_allow_html=True)

        # ── Key articles with News Quality Score (Change 6c) ──────────────────
        if articles:
            st.markdown("**Key articles driving today's signal:**")
            from features.extract_entities import get_event_type as _get_et, get_impact_score as _get_is
            predict_fn = _load_predict_fn()
            scored_cache = []
            if predict_fn:
                for art in articles[:20]:
                    try:
                        r = predict_fn(art["headline"], skip_explanation=True)
                        _et = _get_et(art["headline"])
                        _is = _get_is(_et)
                        scored_cache.append({**art, **r, "event_type": _et, "impact_score": _is})
                    except Exception:
                        scored_cache.append({**art, "label": "LOW", "score": 0, "high_prob": 0,
                                             "event_type": "other", "impact_score": 2})
        else:
            scored_cache = [{**a, "label": "LOW", "score": 0, "high_prob": 0,
                            "event_type": "other", "impact_score": 2} for a in articles]

        top5 = sorted(scored_cache, key=lambda x: (x.get("high_prob", 0) + x.get("impact_score", 0) * 0.05), reverse=True)[:8]
        sig_label_colors = {"HIGH": "#F44336", "MEDIUM": "#FFB300", "LOW": "#00C853"}
        for art in top5:
            lbl = art.get("label", "LOW")
            sc2 = art.get("score", 0) * 100
            clr = sig_label_colors.get(lbl, "#888")
            _et = art.get("event_type", "other")
            _is = art.get("impact_score", 2)
            _et_label, _et_color = EVENT_TYPE_BADGES.get(_et, ("", "#666"))
            _et_html = (f'<span style="background:{_et_color}22;color:{_et_color};font-size:0.7em;'
                       f'padding:1px 6px;border-radius:8px;margin-left:6px">{_et_label}</span>'
                       if _et_label else "")
            _dim = ""  # never dim articles — show all with impact score badge
            _is_color = "#00C853" if _is >= 7 else ("#FFB300" if _is >= 5 else "#555")
            st.markdown(f"""
<div style="border-left:3px solid {clr};padding:8px 12px;margin:3px 0;
background:#111;border-radius:4px;color:#eee;{_dim}">
<span style="color:{clr};font-weight:bold">{lbl} {sc2:.0f}%</span>
<span style="color:{_is_color};font-size:0.8em;margin-left:8px">Impact: {_is}/10</span>
{_et_html}
&nbsp; {art["headline"][:110]}</div>""", unsafe_allow_html=True)

    st.divider()
    # Manual prediction
    st.subheader("Manual Prediction")
    c1, c2 = st.columns([4, 1])
    with c1:
        manual_hl = st.text_input("Headline", placeholder="Paste any headline...", label_visibility="collapsed")
    with c2:
        go_btn = st.button("Predict", type="primary", use_container_width=True)
    if go_btn and manual_hl.strip():
        predict_fn = _load_predict_fn()
        if predict_fn:
            with st.spinner("Analysing..."):
                try:
                    res = predict_fn(manual_hl.strip())
                    lbl = res["label"]
                    clr = sig_label_colors.get(lbl, "#888")
                    st.markdown(f"""
<div style="background:#111;border-left:6px solid {clr};padding:18px 22px;border-radius:10px;margin:12px 0">
<div style="font-size:1.6em;font-weight:bold;color:{clr}">{lbl} | {res['score']*100:.0f}%</div>
<div style="color:#ccc;margin-top:6px">Sentiment: {res['entities'].get('sentiment', 0):+.2f}</div>
</div>""", unsafe_allow_html=True)
                    a1, a2, a3 = st.columns(3)
                    a1.metric("HIGH", f"{res['high_prob']*100:.0f}%")
                    a2.metric("MEDIUM", f"{res['medium_prob']*100:.0f}%")
                    a3.metric("LOW", f"{res['low_prob']*100:.0f}%")
                    if res.get("explanation"):
                        st.info(res["explanation"])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.warning("Model not loaded.")
    elif go_btn:
        st.warning("Enter a headline first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Signal History
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Signal History — Predicted vs Actual")
    st.caption(
        "Every signal logged by the model, with the actual price move that followed. "
        "Use this to evaluate where the model was right and wrong."
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    _hf1, _hf2, _hf3, _hf4 = st.columns([1.5, 1, 1, 1])
    with _hf1:
        _h_comm = st.selectbox("Commodity", ["All"] + COMM_NAMES, key="h_comm")
    with _hf2:
        _h_from = st.date_input("From", value=date.today() - timedelta(days=60), key="h_from")
    with _hf3:
        _h_to = st.date_input("To", value=date.today(), key="h_to")
    with _hf4:
        _h_outcome = st.selectbox(
            "Outcome", ["All", "Correct", "Incorrect", "Cut-Loss", "Pending"],
            key="h_outcome"
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    _hist_df = query_db("""
        SELECT
            prediction_date                AS "Date",
            commodity                      AS "Commodity",
            signal                         AS "Signal",
            direction                      AS "Predicted Direction",
            ROUND(confidence * 100, 1)     AS "Confidence %",
            expected_move                  AS "Expected Move",
            ROUND(entry_price, 3)          AS "Entry $",
            ROUND(exit_price, 3)           AS "Exit $",
            ROUND(actual_move, 2)          AS "Actual Move %",
            COALESCE(outcome, '')          AS "Outcome",
            ROUND(pnl_usd, 0)              AS "P&L $",
            COALESCE(trade_status, 'HOLD') AS "Trade Status",
            COALESCE(data_quality, '')     AS "data_quality",
            COALESCE(headline, '—')        AS "Driving Headline"
        FROM predictions
        ORDER BY prediction_date DESC
    """)

    if not _hist_df.empty:
        # Normalise confidence — stored as 0-1 or 0-100
        if "Confidence %" in _hist_df.columns:
            _hist_df["Confidence %"] = _hist_df["Confidence %"].apply(
                lambda x: round(float(x), 1) if pd.notna(x) else None
            )

        # Apply filters
        if _h_comm != "All":
            _hist_df = _hist_df[_hist_df["Commodity"] == _h_comm]
        if _h_from:
            _hist_df = _hist_df[_hist_df["Date"] >= str(_h_from)]
        if _h_to:
            _hist_df = _hist_df[_hist_df["Date"] <= str(_h_to)]
        if _h_outcome != "All":
            _om = {
                "Correct": "correct", "Incorrect": "incorrect",
                "Cut-Loss": "cut-loss", "Pending": None,
            }
            _ov = _om[_h_outcome]
            if _ov is None:
                _hist_df = _hist_df[_hist_df["Outcome"].isin(["⏳ Pending", None, ""])]
            else:
                _hist_df = _hist_df[_hist_df["Outcome"].str.lower() == _ov]

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    if not _hist_df.empty:
        _resolved = _hist_df[_hist_df["Outcome"].isin(["correct", "incorrect", "cut-loss"])]
        _n_res = len(_resolved)
        _n_correct = (_resolved["Outcome"] == "correct").sum()
        _n_incorrect = (_resolved["Outcome"].isin(["incorrect", "cut-loss"])).sum()
        _acc = round(_n_correct / _n_res * 100, 1) if _n_res > 0 else 0.0
        _acc_warn = _n_res < 30

        # P&L from all resolved signals with real prices
        # Also compute TRADE-only P&L separately if any exist
        _trade_rows = _hist_df[_hist_df["Trade Status"] == "TRADE"] \
            if "Trade Status" in _hist_df.columns else pd.DataFrame()
        _n_trade = len(_trade_rows)

        # All resolved P&L (real prices, any confidence)
        _resolved_with_pnl = _resolved[_resolved["P&L $"].notna()]
        _total_pnl_all = _resolved_with_pnl["P&L $"].sum() if not _resolved_with_pnl.empty else 0.0

        # TRADE-only P&L (conf ≥70%)
        _trade_resolved = _trade_rows[_trade_rows["Outcome"].isin(["correct", "incorrect", "cut-loss"])] \
            if not _trade_rows.empty else pd.DataFrame()
        _total_pnl = _trade_rows["P&L $"].sum() if not _trade_rows.empty else _total_pnl_all
        _pnl_label = "P&L $ (conf ≥70%)" if _n_trade > 0 else "P&L $ (all resolved)"
        _pnl_sub = f"conf ≥70% · {_n_trade} trades" if _n_trade > 0 else f"all {len(_resolved_with_pnl)} resolved signals"

        # Accuracy for TRADE-only rows (apples-to-apples with P&L)
        _trade_acc = round(_trade_resolved["Outcome"].eq("correct").mean() * 100, 1) \
            if len(_trade_resolved) > 0 else None

        # Accuracy split by confidence band
        _high_conf = _resolved[_resolved["Confidence %"] >= 50] if "Confidence %" in _resolved.columns else pd.DataFrame()
        _low_conf  = _resolved[_resolved["Confidence %"] < 50]  if "Confidence %" in _resolved.columns else pd.DataFrame()
        _acc_high  = round(_high_conf["Outcome"].eq("correct").mean() * 100, 1) if len(_high_conf) > 0 else None
        _acc_low   = round(_low_conf["Outcome"].eq("correct").mean() * 100, 1)  if len(_low_conf) > 0 else None

        # ── Sharpe & Sortino from resolved P&L series ─────────────────────────
        import numpy as np
        _sharpe = _sortino = None
        _pnl_series = _resolved_with_pnl["P&L $"].dropna() if not _resolved_with_pnl.empty else pd.Series(dtype=float)
        if len(_pnl_series) >= 5:
            # Normalise to % return per trade (entry price ~$100 avg, 1000 lot)
            _avg_entry = _resolved["Entry $"].dropna().apply(
                lambda x: float(str(x).replace("$","").replace(",","")) if pd.notna(x) else None
            ).dropna().mean() if "Entry $" in _resolved.columns else 100.0
            _avg_entry = _avg_entry or 100.0
            _ret_series = _pnl_series / (_avg_entry * 1000) * 100  # % return per trade
            _rf_per_trade = 0.05 / 252  # daily risk-free rate
            _excess = _ret_series - _rf_per_trade
            if _excess.std() > 0:
                # Annualise assuming ~2 trades/week = 104 trades/year
                _ann_factor = (104 ** 0.5)
                _sharpe = round(float(_excess.mean() / _excess.std() * _ann_factor), 2)
            _downside = _excess[_excess < 0]
            if len(_downside) >= 2 and _downside.std() > 0:
                _sortino = round(float(_excess.mean() / _downside.std() * _ann_factor), 2)

        _low_sample_ratios = len(_pnl_series) < 30

        def _skpi(label, val, sub, color, warn=False):
            w = " ⚠️" if warn else ""
            return (
                f'<div style="background:#0d0d0d;border:1px solid #1e1e2e;border-radius:8px;'
                f'padding:12px 16px;text-align:center">'
                f'<div style="color:#555;font-size:0.68em;text-transform:uppercase;letter-spacing:1px">{label}{w}</div>'
                f'<div style="color:{color};font-size:1.6em;font-weight:bold;font-family:monospace">{val}</div>'
                f'<div style="color:#444;font-size:0.68em">{sub}</div>'
                f'</div>'
            )

        # Row 1: core metrics
        _sk1, _sk2, _sk3, _sk4 = st.columns(4)
        with _sk1:
            st.markdown(_skpi("All Signals", str(len(_hist_df)),
                f"{_n_res} resolved · {_n_trade} acted on", "#e0e0e0"), unsafe_allow_html=True)
        with _sk2:
            _acc_c = "#00C853" if _acc >= 55 else ("#FFB300" if _acc >= 45 else "#F44336")
            st.markdown(_skpi("Direction Accuracy (all)", f"{_acc}%",
                f"all {_n_res} signals incl. low-confidence",
                _acc_c, warn=_acc_warn), unsafe_allow_html=True)
        with _sk3:
            _pnl_c = "#00C853" if _total_pnl >= 0 else "#F44336"
            st.markdown(_skpi(_pnl_label, f"${_total_pnl:+,.0f}",
                _pnl_sub, _pnl_c), unsafe_allow_html=True)
        with _sk4:
            if _trade_acc is not None:
                _ta_c = "#00C853" if _trade_acc >= 55 else ("#FFB300" if _trade_acc >= 45 else "#F44336")
                st.markdown(_skpi("Accuracy (conf ≥55%)", f"{_trade_acc}%",
                    f"same {_n_trade} trades as P&L above", _ta_c), unsafe_allow_html=True)
            else:
                _avg_move = _resolved["Actual Move %"].mean() if _n_res > 0 else 0.0
                st.markdown(_skpi("Avg Actual Move", f"{_avg_move:+.2f}%",
                    "all resolved signals", "#e0e0e0"), unsafe_allow_html=True)

        st.markdown("")

        # Row 2: risk-adjusted metrics
        _sk5, _sk6, _sk7, _sk8 = st.columns(4)
        with _sk5:
            _sh_val = f"{_sharpe:.2f}" if _sharpe is not None else "—"
            _sh_c = "#00C853" if (_sharpe or 0) > 1 else ("#FFB300" if (_sharpe or 0) > 0 else "#F44336")
            st.markdown(_skpi("Sharpe Ratio", _sh_val,
                "⚠️ n<30" if _low_sample_ratios else "annualised, rf=5%",
                _sh_c, warn=_low_sample_ratios), unsafe_allow_html=True)
        with _sk6:
            _so_val = f"{_sortino:.2f}" if _sortino is not None else "—"
            _so_c = "#00C853" if (_sortino or 0) > 1 else ("#FFB300" if (_sortino or 0) > 0 else "#F44336")
            st.markdown(_skpi("Sortino Ratio", _so_val,
                "⚠️ n<30" if _low_sample_ratios else "downside risk only",
                _so_c, warn=_low_sample_ratios), unsafe_allow_html=True)
        with _sk7:
            _wins_r = _resolved[_resolved["Outcome"] == "correct"]
            _loss_r = _resolved[_resolved["Outcome"].isin(["incorrect", "cut-loss"])]
            _avg_win_r  = _wins_r["Actual Move %"].mean() if not _wins_r.empty else 0.0
            _avg_loss_r = _loss_r["Actual Move %"].mean() if not _loss_r.empty else 0.0
            st.markdown(_skpi("Avg Win / Avg Loss",
                f"{_avg_win_r:+.2f}%",
                f"loss avg: {_avg_loss_r:+.2f}%", "#e0e0e0"), unsafe_allow_html=True)
        with _sk8:
            _cut_n = (_resolved["Outcome"] == "cut-loss").sum()
            _cut_pct = round(_cut_n / max(_n_res, 1) * 100, 0)
            st.markdown(_skpi("Cut-Loss Rate", f"{_cut_pct:.0f}%",
                f"{_cut_n} of {_n_res} resolved signals",
                "#FFB300" if _cut_pct > 20 else "#e0e0e0"), unsafe_allow_html=True)

        # ── Explanation banner ────────────────────────────────────────────────
        _pnl_explain = (
            f'P&L counts the <b>{_n_trade} signals with confidence ≥70%</b> — the ones the system '
            f'actually recommended acting on. '
            f'{"Accuracy for those " + str(_n_trade) + " traded signals: <b style=color:#00C853>" + str(_trade_acc) + "%</b>." if _trade_acc and _n_trade >= 3 else "Not enough TRADE signals yet for meaningful accuracy."}'
        ) if _n_trade > 0 else (
            f'No signals have reached the ≥70% confidence threshold yet — '
            f'P&L shows all <b>{len(_resolved_with_pnl)} resolved signals</b> regardless of confidence. '
            f'Total: <b style="color:{"#00C853" if _total_pnl_all >= 0 else "#F44336"}">${_total_pnl_all:+,.0f}</b>. '
            f'The April 7 loss was the Trump tariff shock — oil dropped 13–16% in one day.'
        )
        st.markdown(
            f'<div style="background:#0d0d0d;border-left:3px solid #378ADD;'
            f'border-radius:0 6px 6px 0;padding:10px 16px;margin:10px 0;font-size:0.82em;color:#aaa">'
            f'<b style="color:#e0e0e0">Why do these numbers look different?</b><br>'
            f'Direction Accuracy ({_acc}%) counts <b>all {_n_res} signals</b> including low-confidence ones '
            f'(which are only ~32% accurate and show as grey "monitor only" cards). '
            f'{_pnl_explain}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if _acc_warn and _n_res > 0:
            st.caption(f"⚠️ Only {_n_res} resolved signals — not statistically reliable until n ≥ 30.")

        # ── Confidence band breakdown ─────────────────────────────────────────
        if _acc_high is not None and _acc_low is not None:
            _insight_color = "#00C853" if _acc_high >= 50 else "#FFB300"
            st.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;'
                f'padding:10px 16px;margin:6px 0;font-size:0.82em">'
                f'<span style="color:#555;text-transform:uppercase;font-size:0.72em;letter-spacing:1px">Accuracy by confidence band</span><br>'
                f'<span style="color:{_insight_color}">Conf ≥50% (cards show direction): <b>{_acc_high}%</b> correct ({len(_high_conf)} signals)</span>'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                f'<span style="color:#F44336">Conf &lt;50% (monitor only): <b>{_acc_low}%</b> correct ({len(_low_conf)} signals)</span>'
                f'<br><span style="color:#555;font-size:0.9em">Note: TRADE action requires conf ≥70% — a separate, stricter threshold.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

    # ── Main history table ────────────────────────────────────────────────────
    if _hist_df.empty:
        st.info("No signal history found for the selected filters.")
    else:
        disp_h = _hist_df.drop(columns=["data_quality"], errors="ignore").copy()

        # Format Predicted Direction
        if "Predicted Direction" in disp_h.columns:
            def _fmt_dir(v):
                v = str(v or "").lower()
                if v in ("bullish", "rise"):  return "▲ Bullish"
                if v in ("bearish", "fall"):  return "▼ Bearish"
                return "→ Neutral"
            disp_h["Predicted Direction"] = disp_h["Predicted Direction"].apply(_fmt_dir)

        # Format Actual Move % with direction arrow
        if "Actual Move %" in disp_h.columns:
            def _fmt_move(v):
                if pd.isna(v) or v == "": return "—"
                v = float(v)
                arrow = "▲" if v > 0 else ("▼" if v < 0 else "→")
                return f"{arrow} {v:+.2f}%"
            disp_h["Actual Move %"] = disp_h["Actual Move %"].apply(_fmt_move)

        # Format Outcome
        if "Outcome" in disp_h.columns:
            _today_str = date.today().isoformat()
            def _fmt_oc(row):
                v = str(row.get("Outcome") or "").lower().strip()
                d = str(row.get("Date") or "")
                dq = str(row.get("data_quality") or "")
                if v == "correct":   return "✓ Correct"
                if v == "cut-loss":  return "✗ Cut-Loss"
                if v == "incorrect": return "✗ Incorrect"
                # Smart pending labels
                if d == _today_str:  return "⏳ Today — awaiting close"
                try:
                    from datetime import datetime as _dt
                    dow = _dt.strptime(d, "%Y-%m-%d").weekday()
                    if dow == 5: return "📅 Saturday — no market"
                    if dow == 6: return "📅 Sunday — no market"
                except Exception:
                    pass
                if "holiday" in dq: return "📅 Holiday — no market"
                if "price_fetch_failed" in dq: return "⚠ Price unavailable"
                return "⏳ Pending"
            disp_h["Outcome"] = disp_h.apply(_fmt_oc, axis=1)

        # Format P&L
        if "P&L $" in disp_h.columns:
            disp_h["P&L $"] = disp_h["P&L $"].apply(
                lambda x: f"${float(x):+,.0f}" if pd.notna(x) and x != "" else "—"
            )

        # Format prices
        for _pc in ("Entry $", "Exit $"):
            if _pc in disp_h.columns:
                disp_h[_pc] = disp_h[_pc].apply(
                    lambda x: f"${float(x):.3f}" if pd.notna(x) and x != "" else "—"
                )

        # Format confidence
        if "Confidence %" in disp_h.columns:
            disp_h["Confidence %"] = disp_h["Confidence %"].apply(
                lambda x: f"{float(x):.1f}%" if pd.notna(x) else "—"
            )

        # Truncate headline
        if "Driving Headline" in disp_h.columns:
            disp_h["Driving Headline"] = disp_h["Driving Headline"].apply(
                lambda x: str(x)[:90] + "…" if len(str(x)) > 90 else str(x)
            )

        def _style_hist(row):
            styles = [""] * len(row)
            cols_l = list(row.index)
            def _si(n): return cols_l.index(n) if n in cols_l else None

            oi = _si("Outcome")
            if oi is not None:
                v = str(row.iloc[oi])
                if "Correct" in v:   styles[oi] = "color:#00C853"
                elif "Incorrect" in v or "Cut" in v: styles[oi] = "color:#F44336"
                else:                styles[oi] = "color:#FFB300"

            mi = _si("Actual Move %")
            if mi is not None:
                v = str(row.iloc[mi])
                if "▲" in v:  styles[mi] = "color:#00C853"
                elif "▼" in v: styles[mi] = "color:#F44336"

            di = _si("Predicted Direction")
            if di is not None:
                v = str(row.iloc[di])
                if "Bullish" in v: styles[di] = "color:#00C853"
                elif "Bearish" in v: styles[di] = "color:#F44336"
                else:               styles[di] = "color:#FFB300"

            pi = _si("P&L $")
            if pi is not None:
                v = str(row.iloc[pi])
                if "+" in v:   styles[pi] = "color:#00C853"
                elif "-" in v: styles[pi] = "color:#F44336"

            return styles

        st.dataframe(
            disp_h.style.apply(_style_hist, axis=1),
            use_container_width=True,
            height=480,
            hide_index=True,
        )
        st.caption(
            f"Showing {len(disp_h)} signals. "
            "Outcome = whether predicted direction matched actual price move. "
            "P&L = (exit − entry) × 1,000 barrels/MMBtu, direction-adjusted."
        )

    # ── Accuracy over time chart ───────────────────────────────────────────────
    _chart_df = query_db("""
        SELECT prediction_date, commodity, outcome, actual_move, direction
        FROM predictions
        WHERE outcome IS NOT NULL AND outcome != 'pending'
        ORDER BY prediction_date ASC
    """)
    if not _chart_df.empty:
        st.divider()
        st.markdown("#### Direction Accuracy Over Time (rolling 10 signals)")
        _chart_df["correct_flag"] = _chart_df["outcome"].apply(
            lambda x: 1 if str(x).lower() == "correct" else 0
        )
        _chart_df = _chart_df.sort_values("prediction_date")
        _chart_df["rolling_acc"] = _chart_df["correct_flag"].rolling(10, min_periods=3).mean() * 100

        fig_acc = go.Figure(go.Scatter(
            x=_chart_df["prediction_date"],
            y=_chart_df["rolling_acc"],
            mode="lines",
            line=dict(color="#378ADD", width=2),
            fill="tozeroy",
            fillcolor="rgba(55,138,221,0.08)",
            hovertemplate="%{x}<br>Rolling accuracy: %{y:.1f}%<extra></extra>",
        ))
        fig_acc.add_hline(y=50, line_dash="dash", line_color="#555",
                          annotation_text="50% baseline", annotation_position="right")
        fig_acc.update_layout(
            yaxis_title="Accuracy %", yaxis_range=[0, 100],
            **PLOTLY_DEFAULTS,
        )
        st.plotly_chart(fig_acc, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Price Chart
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Price Chart — Entry vs Exit")
    st.caption("Solid line = entry price · Dashed = exit price · Markers = signal type")

    c1, c2 = st.columns([2, 1])
    with c1:
        comm3 = st.selectbox("Commodity", ["All"] + COMM_NAMES, key="t3_comm")
    with c2:
        days3 = st.slider("Days to show", 7, 90, 30, key="t3_days")

    since3 = (date.today() - timedelta(days=days3)).isoformat()
    sql3 = "SELECT * FROM predictions WHERE prediction_date>=?"
    par3: list = [since3]
    if comm3 != "All":
        sql3 += " AND commodity=?"; par3.append(comm3)
    sql3 += " ORDER BY prediction_date ASC"
    df3 = query_db(sql3, tuple(par3))

    if df3.empty:
        st.info("No price data for the selected period. Run the backfill to populate exit prices.")
    else:
        MARKER_SYMBOL = {
            "rise": "triangle-up", "increase": "triangle-up",
            "fall": "triangle-down", "decrease": "triangle-down",
            "flat": "circle", "stable": "circle",
        }
        SIG_MARKER_COLOR = {"HIGH": "#F44336", "MEDIUM": "#FFB300", "LOW": "#00C853"}

        # Chart 1 — entry/exit lines with signal markers
        fig3 = go.Figure()
        comms_to_plot = COMM_NAMES if comm3 == "All" else [comm3]
        for comm in comms_to_plot:
            sub = df3[df3["commodity"] == comm]
            if sub.empty:
                continue
            color = COMMODITY_COLORS.get(comm, "#888")
            if "entry_price" in sub.columns and sub["entry_price"].notna().any():
                fig3.add_trace(go.Scatter(
                    x=sub["prediction_date"], y=sub["entry_price"],
                    mode="lines", name=f"{comm[:12]} Entry",
                    line=dict(color=color, width=2),
                    hovertemplate="%{x}<br>Entry: $%{y:.3f}<extra>" + comm + "</extra>",
                ))
                # Signal markers
                for sl, mc in SIG_MARKER_COLOR.items():
                    s = sub[(sub["signal"] == sl) & sub["entry_price"].notna()] if "signal" in sub.columns else pd.DataFrame()
                    if s.empty:
                        continue
                    syms = [MARKER_SYMBOL.get(str(d), "circle") for d in s.get("direction", ["circle"] * len(s))]
                    fig3.add_trace(go.Scatter(
                        x=s["prediction_date"], y=s["entry_price"],
                        mode="markers", name=f"{sl}",
                        showlegend=False,
                        marker=dict(symbol=syms, size=9, color=mc, line=dict(color="#fff", width=1)),
                        hovertemplate=f"%{{x}}<br>{sl}<br>$%{{y:.3f}}<extra>{comm}</extra>",
                    ))
            if "exit_price" in sub.columns and sub["exit_price"].notna().any():
                fig3.add_trace(go.Scatter(
                    x=sub["prediction_date"], y=sub["exit_price"],
                    mode="lines", name=f"{comm[:12]} Exit",
                    line=dict(color=color, width=1.5, dash="dash"),
                    hovertemplate="%{x}<br>Exit: $%{y:.3f}<extra>" + comm + "</extra>",
                ))

        fig3.update_layout(title="Entry vs Exit Price by Commodity", height=420,
                           hovermode="x unified", **PLOTLY_DEFAULTS)
        fig3.update_yaxes(title="Price (USD)")
        st.plotly_chart(fig3, use_container_width=True)

        # Chart 2 — predicted move vs actual move scatter
        if "actual_move" in df3.columns and df3["actual_move"].notna().any():
            df3_scatter = df3[df3["actual_move"].notna()].copy()
            # Derive predicted_move numeric from expected_move string
            def _parse_move(m):
                if not m or not isinstance(m, str):
                    return 0.0
                if ">3" in m:   return 3.5
                if "1-3" in m:  return 2.0
                return 0.5
            df3_scatter["predicted_move"] = df3_scatter["expected_move"].apply(_parse_move)
            if not df3_scatter.empty:
                fig_sc = go.Figure()
                for comm in comms_to_plot:
                    sub_sc = df3_scatter[df3_scatter["commodity"] == comm]
                    if sub_sc.empty:
                        continue
                    fig_sc.add_trace(go.Scatter(
                        x=sub_sc["predicted_move"], y=sub_sc["actual_move"],
                        mode="markers", name=comm,
                        marker=dict(color=COMMODITY_COLORS.get(comm, "#888"), size=8, opacity=0.8),
                        hovertemplate="Predicted: %{x:.1f}%<br>Actual: %{y:.2f}%<extra>" + comm + "</extra>",
                    ))
                # y=x reference line
                all_vals = list(df3_scatter["predicted_move"]) + list(df3_scatter["actual_move"])
                mn, mx = min(all_vals, default=-5), max(all_vals, default=5)
                fig_sc.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    name="Perfect prediction", line=dict(color="#555", dash="dot", width=1),
                    showlegend=True,
                ))
                fig_sc.update_layout(title="Predicted Move vs Actual Move",
                                     xaxis_title="Predicted Move (%)", yaxis_title="Actual Move (%)",
                                     height=380, **PLOTLY_DEFAULTS)
                st.plotly_chart(fig_sc, use_container_width=True)

        # Day-by-day table
        st.markdown("**Day-by-day breakdown:**")
        tbl_cols = ["prediction_date", "commodity", "signal", "entry_price",
                    "exit_price", "expected_move", "actual_move", "pnl_usd", "outcome"]
        tbl_cols = [c for c in tbl_cols if c in df3.columns]
        tbl = df3[tbl_cols].copy().sort_values("prediction_date", ascending=False)
        if "entry_price" in tbl.columns:
            tbl["entry_price"] = tbl["entry_price"].apply(lambda x: f"${x:.3f}" if pd.notna(x) else "—")
        if "exit_price" in tbl.columns:
            tbl["exit_price"] = tbl["exit_price"].apply(lambda x: f"${x:.3f}" if pd.notna(x) else "—")
        if "actual_move" in tbl.columns:
            tbl["actual_move"] = tbl["actual_move"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
        if "pnl_usd" in tbl.columns:
            tbl["pnl_usd"] = tbl["pnl_usd"].apply(lambda x: f"${x:+,.0f}" if pd.notna(x) else "—")
        if "outcome" in tbl.columns:
            tbl["outcome"] = tbl["outcome"].apply(
                lambda x: "✓" if x == "correct" else ("✗" if x == "incorrect" else "⏳") if pd.notna(x) else "⏳")
        st.dataframe(tbl, use_container_width=True, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Entry Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    from inference.optimizer import (
        build_all_setups, build_setup, summary_stats,
        load_historical_setups, TARGET_MOVE, STOP_LOSS_PCT, MIN_CONFIDENCE,
    )

    st.markdown("## 🎯 Entry Optimizer")
    st.caption(
        "Auto-generates the best trade setup from today's signals. "
        "Entry zone · Target · Thesis invalidation · R:R · Expected Value. "
        "Adjust lot size or stop-loss below if needed."
    )

    # ── Controls — collapsed by default so auto result is front and centre ────
    with st.expander("⚙️ Adjust parameters (optional)", expanded=False):
        oc1, oc2, oc3, oc4 = st.columns([1, 1, 1, 1])
        with oc1:
            opt_lot_size = st.number_input("Lot Size (bbl/MMBtu)", value=1000, min_value=1,
                                           step=100, key="opt4_lot")
        with oc2:
            opt_stop_pct = st.slider("Stop-Loss %", 0.5, 3.0, 1.0, step=0.1, key="opt4_stop")
        with oc3:
            opt_sort = st.selectbox("Sort by", ["EV (best first)", "Confidence", "R:R Ratio"],
                                    key="opt4_sort")
        with oc4:
            opt_best_only = st.toggle("Best setups only (EV > 0)", value=False, key="opt4_best")

    # ── Build setups ──────────────────────────────────────────────────────────
    _opt_conn = sqlite3.connect(DB_PATH, timeout=30)
    _opt_conn.execute("PRAGMA journal_mode=WAL")
    _opt_setups = build_all_setups(
        st.session_state.today_signals,
        _opt_conn,
        lot_size=opt_lot_size,
        stop_pct_override=opt_stop_pct,
    )
    _opt_conn.close()

    if opt_best_only:
        _opt_setups = [s for s in _opt_setups if s.is_best]

    # Sort
    if opt_sort == "EV (best first)":
        _opt_setups.sort(key=lambda s: s.expected_value, reverse=True)
    elif opt_sort == "Confidence":
        _opt_setups.sort(key=lambda s: s.confidence_pct, reverse=True)
    else:
        _opt_setups.sort(key=lambda s: s.risk_reward, reverse=True)

    stats = summary_stats(_opt_setups)

    # ── AUTO RECOMMENDATION — best setup highlighted at top ───────────────────
    _best_setups = [s for s in _opt_setups if s.is_best]
    _top = _best_setups[0] if _best_setups else (_opt_setups[0] if _opt_setups else None)

    if _top:
        _dir_color = "#00C853" if _top.trade_direction == "Bullish" else "#F44336"
        _dir_icon  = "▲" if _top.trade_direction == "Bullish" else "▼"
        _ev_color  = "#00C853" if _top.expected_value > 0 else "#FFB300"
        _rc_colors = {"strong": "#00C853", "moderate": "#378ADD", "cautious": "#FFB300", "skip": "#888"}
        _rc = _rc_colors.get(_top.rating, "#888")

        st.markdown(
            f'<div style="background:#0d1a0d;border:2px solid {_dir_color};border-radius:12px;'
            f'padding:20px 24px;margin-bottom:16px">'
            f'<div style="color:#666;font-size:0.72em;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px">'
            f'★ Auto-Selected Best Setup</div>'
            f'<div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap">'
            f'<span style="color:{_dir_color};font-size:2em;font-weight:bold">'
            f'{_dir_icon} {_top.trade_direction} — {_top.commodity}</span>'
            f'<span style="background:{_rc}22;color:{_rc};font-size:0.8em;padding:3px 12px;'
            f'border-radius:10px;border:1px solid {_rc}44;text-transform:uppercase">{_top.rating}</span>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px">'
            f'<div><div style="color:#555;font-size:0.68em;text-transform:uppercase">Entry</div>'
            f'<div style="color:#e0e0e0;font-size:1.1em;font-weight:bold;font-family:monospace">${_top.entry_price:,.3f}</div>'
            f'<div style="color:#444;font-size:0.7em">${_top.entry_low:,.3f} – ${_top.entry_high:,.3f}</div></div>'
            f'<div><div style="color:#555;font-size:0.68em;text-transform:uppercase">Target (TP)</div>'
            f'<div style="color:#00C853;font-size:1.1em;font-weight:bold;font-family:monospace">${_top.target_price:,.3f}</div>'
            f'<div style="color:#444;font-size:0.7em">+{_top.target_move_pct:.2f}%</div></div>'
            f'<div><div style="color:#555;font-size:0.68em;text-transform:uppercase">Thesis Invalidation</div>'
            f'<div style="color:#F44336;font-size:1.1em;font-weight:bold;font-family:monospace">${_top.stop_loss:,.3f}</div>'
            f'<div style="color:#444;font-size:0.7em">-{_top.stop_loss_pct:.1f}%</div></div>'
            f'<div><div style="color:#555;font-size:0.68em;text-transform:uppercase">R:R Ratio</div>'
            f'<div style="color:#378ADD;font-size:1.1em;font-weight:bold;font-family:monospace">{_top.risk_reward:.2f}:1</div>'
            f'<div style="color:#444;font-size:0.7em">${_top.risk_usd:,.0f} risk / ${_top.reward_usd:,.0f} reward</div></div>'
            f'<div><div style="color:#555;font-size:0.68em;text-transform:uppercase">Expected Value</div>'
            f'<div style="color:{_ev_color};font-size:1.1em;font-weight:bold;font-family:monospace">${_top.expected_value:+,.0f}</div>'
            f'<div style="color:#444;font-size:0.7em">{_top.win_rate*100:.0f}% hist. win rate</div></div>'
            f'</div>'
            f'<div style="margin-top:12px;color:#888;font-size:0.82em">💡 {_top.rationale}</div>'
            f'{"<div style=margin-top:6px;color:#666;font-size:0.78em>📰 " + _top.headline[:110] + "</div>" if _top.headline else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if len(_best_setups) > 1:
            st.caption(f"↓ {len(_best_setups) - 1} more qualifying setups below")
    else:
        st.info(
            "No qualifying setups today — all signals are below confidence threshold or direction is neutral. "
            "This is expected on low-news days. Check back after S&P articles are refreshed."
        )

    st.markdown("")

    # ── KPI summary row ───────────────────────────────────────────────────────
    if _opt_setups:
        sk1, sk2, sk3, sk4 = st.columns(4)
        def _okpi(label, val, sub, color):
            return (
                f'<div style="background:#0a0a0a;border:1px solid #1a1a2e;border-radius:6px;'
                f'padding:10px 14px;text-align:center">'
                f'<div style="color:#555;font-size:0.68em;text-transform:uppercase;letter-spacing:1px">{label}</div>'
                f'<div style="color:{color};font-size:1.5em;font-weight:bold;font-family:monospace">{val}</div>'
                f'<div style="color:#444;font-size:0.68em">{sub}</div>'
                f'</div>'
            )
        with sk1:
            st.markdown(_okpi("Setups Generated", str(stats["total"]),
                f"{stats['best_count']} best (EV>0, R:R≥1.5)", "#e0e0e0"), unsafe_allow_html=True)
        with sk2:
            _rrc = "#00C853" if stats["avg_rr"] >= 1.5 else ("#FFB300" if stats["avg_rr"] >= 1.0 else "#F44336")
            st.markdown(_okpi("Avg R:R", f"{stats['avg_rr']:.2f}:1",
                "reward / risk", _rrc), unsafe_allow_html=True)
        with sk3:
            _evc = "#00C853" if stats["avg_ev"] > 0 else "#F44336"
            st.markdown(_okpi("Avg Expected Value", f"${stats['avg_ev']:+,.0f}",
                "per lot", _evc), unsafe_allow_html=True)
        with sk4:
            _hcc = "#00C853" if stats["high_conf_pct"] >= 50 else "#FFB300"
            st.markdown(_okpi("High Confidence", f"{stats['high_conf_pct']:.0f}%",
                "setups ≥70%", _hcc), unsafe_allow_html=True)
        st.markdown("")

    # ── All setups (remaining, collapsed) ────────────────────────────────────
    _remaining = [s for s in _opt_setups if s is not _top]
    if _remaining:
        st.divider()
        st.markdown("#### All Setups")
        for setup in _remaining:
            _di = "▲ Bullish" if setup.trade_direction == "Bullish" else "▼ Bearish"
            _dc = "#00C853" if setup.trade_direction == "Bullish" else "#F44336"
            _rc2 = {"strong":"#00C853","moderate":"#378ADD","cautious":"#FFB300","skip":"#888"}.get(setup.rating,"#888")
            with st.expander(
                f"{setup.commodity}  ·  {_di}  ·  R:R {setup.risk_reward:.2f}:1  ·  EV ${setup.expected_value:+,.0f}",
                expanded=False,
            ):
                lv1, lv2, lv3, lv4, lv5 = st.columns(5)
                lv1.metric("Entry", f"${setup.entry_price:,.3f}")
                lv2.metric("Zone", f"${setup.entry_low:,.3f}–${setup.entry_high:,.3f}")
                lv3.metric("Target", f"${setup.target_price:,.3f}",
                           delta=f"{'+' if setup.trade_direction=='Bullish' else '-'}{setup.target_move_pct:.2f}%")
                lv4.metric("Invalidation", f"${setup.stop_loss:,.3f}",
                           delta=f"-{setup.stop_loss_pct:.1f}%", delta_color="inverse")
                lv5.metric("R:R", f"{setup.risk_reward:.2f}:1")
                sv1, sv2, sv3, sv4 = st.columns(4)
                sv1.metric("Confidence", f"{setup.confidence_pct:.0f}%")
                sv2.metric("Win Rate", f"{setup.win_rate*100:.0f}%")
                sv3.metric("EV", f"${setup.expected_value:+,.0f}")
                sv4.metric("Risk/Reward $", f"${setup.risk_usd:,.0f} / ${setup.reward_usd:,.0f}")
                st.caption(f"💡 {setup.rationale}")

    st.divider()

    # ── Historical setups table ───────────────────────────────────────────────
    st.markdown("### Historical Trade Setups (last 30 days)")
    st.caption("Reconstructed entry/target/stop levels from past predictions with real prices.")

    try:
        _hist_conn = sqlite3.connect(DB_PATH, timeout=30)
        _hist_conn.execute("PRAGMA journal_mode=WAL")
        df_hist = load_historical_setups(_hist_conn, days=30)
        _hist_conn.close()
    except Exception:
        df_hist = pd.DataFrame()

    if df_hist.empty:
        st.info("No historical setups with price data yet.")
    else:
        hf1, hf2, hf3 = st.columns(3)
        with hf1:
            h_comm = st.selectbox("Commodity", ["All"] + COMM_NAMES, key="hist_comm")
        with hf2:
            h_outcome = st.selectbox("Outcome", ["All", "correct", "incorrect", "pending"],
                                     key="hist_outcome")
        with hf3:
            h_signal = st.selectbox("Signal", ["All", "HIGH", "MEDIUM", "LOW"], key="hist_sig")

        df_h = df_hist.copy()
        if h_comm != "All": df_h = df_h[df_h["commodity"] == h_comm]
        if h_outcome != "All": df_h = df_h[df_h["outcome"] == h_outcome]
        if h_signal != "All": df_h = df_h[df_h["signal"] == h_signal]

        display_cols = {
            "prediction_date": "Date", "commodity": "Commodity", "signal": "Signal",
            "direction": "Direction", "entry_price": "Entry $", "target_price": "Target $",
            "stop_loss": "Stop $", "risk_reward": "R:R", "confidence_pct": "Conf %",
            "outcome": "Outcome", "pnl_usd": "P&L $",
        }
        df_display = df_h[[c for c in display_cols if c in df_h.columns]].rename(columns=display_cols)

        def _color_row(row):
            oc = row.get("Outcome", "")
            if oc == "correct":   return ["background-color:#0a1f0a"] * len(row)
            if oc == "incorrect": return ["background-color:#1f0a0a"] * len(row)
            return [""] * len(row)

        st.dataframe(df_display.style.apply(_color_row, axis=1),
                     use_container_width=True, hide_index=True)



    if opt_best_only:
        _opt_setups = [s for s in _opt_setups if s.is_best]

    # Sort
    if opt_sort == "EV (best first)":
        _opt_setups.sort(key=lambda s: s.expected_value, reverse=True)
    elif opt_sort == "Confidence":
        _opt_setups.sort(key=lambda s: s.confidence_pct, reverse=True)
    else:
        _opt_setups.sort(key=lambda s: s.risk_reward, reverse=True)

    stats = summary_stats(_opt_setups)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1:
        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1a1a2e;border-radius:6px;'
            f'padding:12px 16px;text-align:center">'
            f'<div style="color:#666;font-size:0.72em;text-transform:uppercase;letter-spacing:1px">Trade Setups</div>'
            f'<div style="color:#e0e0e0;font-size:1.8em;font-weight:bold;font-family:monospace">{stats["total"]}</div>'
            f'<div style="color:#444;font-size:0.7em">{stats["best_count"]} best (EV&gt;0, R:R≥1.5)</div>'
            f'</div>', unsafe_allow_html=True)
    with sk2:
        rr_color = "#00C853" if stats["avg_rr"] >= 1.5 else ("#FFB300" if stats["avg_rr"] >= 1.0 else "#F44336")
        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1a1a2e;border-radius:6px;'
            f'padding:12px 16px;text-align:center">'
            f'<div style="color:#666;font-size:0.72em;text-transform:uppercase;letter-spacing:1px">Avg R:R Ratio</div>'
            f'<div style="color:{rr_color};font-size:1.8em;font-weight:bold;font-family:monospace">{stats["avg_rr"]:.2f}:1</div>'
            f'<div style="color:#444;font-size:0.7em">reward / risk per lot</div>'
            f'</div>', unsafe_allow_html=True)
    with sk3:
        ev_color = "#00C853" if stats["avg_ev"] > 0 else "#F44336"
        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1a1a2e;border-radius:6px;'
            f'padding:12px 16px;text-align:center">'
            f'<div style="color:#666;font-size:0.72em;text-transform:uppercase;letter-spacing:1px">Avg Expected Value</div>'
            f'<div style="color:{ev_color};font-size:1.8em;font-weight:bold;font-family:monospace">${stats["avg_ev"]:+,.0f}</div>'
            f'<div style="color:#444;font-size:0.7em">per lot</div>'
            f'</div>', unsafe_allow_html=True)
    with sk4:
        hc_color = "#00C853" if stats["high_conf_pct"] >= 50 else "#FFB300"
        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1a1a2e;border-radius:6px;'
            f'padding:12px 16px;text-align:center">'
            f'<div style="color:#666;font-size:0.72em;text-transform:uppercase;letter-spacing:1px">High Confidence</div>'
            f'<div style="color:{hc_color};font-size:1.8em;font-weight:bold;font-family:monospace">{stats["high_conf_pct"]:.0f}%</div>'
            f'<div style="color:#444;font-size:0.7em">setups ≥70% confidence</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Setup cards ───────────────────────────────────────────────────────────
    if not _opt_setups:
        st.info("No trade setups generated. All signals are flat or below confidence threshold. "
                "Refresh data or lower the stop-loss to generate setups.")
    else:
        for setup in _opt_setups:
            _dir_icon = "▲ Bullish" if setup.trade_direction == "Bullish" else "▼ Bearish"
            _dir_color = "#00C853" if setup.trade_direction == "Bullish" else "#F44336"
            _ev_color = "#00C853" if setup.expected_value > 0 else "#F44336"
            _rating_colors = {
                "strong": "#00C853", "moderate": "#378ADD",
                "cautious": "#FFB300", "skip": "#888",
            }
            _rc = _rating_colors.get(setup.rating, "#888")
            _best_badge = (
                '<span style="background:#1a3a1a;color:#00C853;font-size:0.7em;'
                'padding:2px 8px;border-radius:10px;margin-left:8px">★ BEST SETUP</span>'
                if setup.is_best else ""
            )

            with st.expander(
                f"{setup.commodity}  ·  {_dir_icon}  ·  {setup.signal}  ·  "
                f"R:R {setup.risk_reward:.2f}:1  ·  EV ${setup.expected_value:+,.0f}",
                expanded=True,
            ):
                # Header row
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin-bottom:12px">'
                    f'<span style="color:{_dir_color};font-size:1.4em;font-weight:bold">'
                    f'{_dir_icon}</span>'
                    f'<span style="color:#aaa;font-size:0.9em;margin-left:12px">'
                    f'{setup.commodity}</span>'
                    f'<span style="background:#1a1a2e;color:{_rc};font-size:0.75em;'
                    f'padding:2px 10px;border-radius:10px;margin-left:10px;text-transform:uppercase">'
                    f'{setup.rating}</span>'
                    f'{_best_badge}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Price levels grid
                lv1, lv2, lv3, lv4, lv5 = st.columns(5)
                lv1.metric("Entry Price", f"${setup.entry_price:,.3f}")
                lv2.metric(
                    "Entry Zone",
                    f"${setup.entry_low:,.3f} – ${setup.entry_high:,.3f}",
                )
                lv3.metric(
                    "Target (TP)",
                    f"${setup.target_price:,.3f}",
                    delta=f"+{setup.target_move_pct:.2f}%" if setup.trade_direction == "Bullish"
                          else f"-{setup.target_move_pct:.2f}%",
                )
                lv4.metric(
                    "Thesis Invalidation",
                    f"${setup.stop_loss:,.3f}",
                    delta=f"-{setup.stop_loss_pct:.1f}%",
                    delta_color="inverse",
                )
                lv5.metric("R:R Ratio", f"{setup.risk_reward:.2f}:1")

                st.markdown("")

                # Stats row
                sv1, sv2, sv3, sv4 = st.columns(4)
                sv1.metric("Confidence", f"{setup.confidence_pct:.0f}%")
                sv2.metric("Win Rate (hist.)", f"{setup.win_rate*100:.0f}%")
                sv3.metric(
                    "Expected Value",
                    f"${setup.expected_value:+,.0f}",
                    delta="positive EV" if setup.expected_value > 0 else "negative EV",
                    delta_color="normal" if setup.expected_value > 0 else "inverse",
                )
                sv4.metric("Risk / Reward $", f"${setup.risk_usd:,.0f} / ${setup.reward_usd:,.0f}")

                # Rationale
                st.caption(f"💡 {setup.rationale}")
                if setup.headline:
                    st.caption(f"📰 Driving headline: {setup.headline[:120]}")

    st.divider()

    # ── Historical setups table ───────────────────────────────────────────────
    st.markdown("### Historical Trade Setups (last 30 days)")
    st.caption("Reconstructed entry/target/stop levels from past predictions with real prices.")

    try:
        _hist_conn = sqlite3.connect(DB_PATH, timeout=30)
        _hist_conn.execute("PRAGMA journal_mode=WAL")
        df_hist = load_historical_setups(_hist_conn, days=30)
        _hist_conn.close()
    except Exception:
        df_hist = pd.DataFrame()

    if df_hist.empty:
        st.info("No historical setups with price data yet. Run backfill to populate.")
    else:
        # Filter controls
        hf1, hf2, hf3 = st.columns(3)
        with hf1:
            h_comm = st.selectbox("Commodity", ["All"] + COMM_NAMES, key="hist_comm2")
        with hf2:
            h_outcome = st.selectbox("Outcome", ["All", "correct", "incorrect", "pending"],
                                     key="hist_outcome2")
        with hf3:
            h_signal = st.selectbox("Signal", ["All", "HIGH", "MEDIUM", "LOW"], key="hist_sig2")

        df_h = df_hist.copy()
        if h_comm != "All":
            df_h = df_h[df_h["commodity"] == h_comm]
        if h_outcome != "All":
            df_h = df_h[df_h["outcome"] == h_outcome]
        if h_signal != "All":
            df_h = df_h[df_h["signal"] == h_signal]

        # Build display table
        display_cols = {
            "prediction_date": "Date",
            "commodity": "Commodity",
            "signal": "Signal",
            "direction": "Direction",
            "entry_price": "Entry $",
            "target_price": "Target $",
            "stop_loss": "Stop $",
            "risk_reward": "R:R",
            "confidence_pct": "Conf %",
            "outcome": "Outcome",
            "pnl_usd": "P&L $",
        }
        df_display = df_h[[c for c in display_cols if c in df_h.columns]].rename(columns=display_cols)

        def _color_row(row):
            outcome = row.get("Outcome", "")
            if outcome == "correct":
                return ["background-color:#0a1f0a"] * len(row)
            if outcome == "incorrect":
                return ["background-color:#1f0a0a"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_display.style.apply(_color_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # ── Manual setup builder ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Manual Setup Builder")
    st.caption("Override any parameter to build a custom trade setup.")

    mb1, mb2 = st.columns(2)
    with mb1:
        m_comm = st.selectbox("Commodity", COMM_NAMES, key="man_comm")
        m_price = st.number_input("Entry Price (USD)", value=75.0, min_value=0.01,
                                  step=0.01, format="%.3f", key="man_price")
        m_signal = st.selectbox("Signal", ["HIGH", "MEDIUM", "LOW"], key="man_sig")
        m_dir = st.selectbox("Direction", ["rise", "fall"], key="man_dir")
    with mb2:
        m_conf = st.slider("Confidence (%)", 35, 95, 65, key="man_conf")
        m_stop = st.slider("Stop-Loss %", 0.5, 3.0, 1.0, step=0.1, key="man_stop")
        m_lot = st.number_input("Lot Size", value=1000, min_value=1, step=100, key="man_lot")

    if st.button("Build Setup", type="primary", key="man_build"):
        _man_sig_dict = {
            "commodity": m_comm,
            "ticker": dict(COMMODITIES).get(m_comm, ""),
            "signal": m_signal,
            "direction": m_dir,
            "scaled_conf": float(m_conf),
            "top_headline": "",
        }
        _man_conn = sqlite3.connect(DB_PATH)
        _man_setup = build_setup(_man_sig_dict, m_price, _man_conn,
                                 lot_size=m_lot, stop_pct_override=m_stop)
        _man_conn.close()

        if _man_setup is None:
            st.warning("No setup generated — direction is flat or confidence below threshold.")
        else:
            _d_icon = "▲ Bullish" if _man_setup.trade_direction == "Bullish" else "▼ Bearish"
            _d_col = "#00C853" if _man_setup.trade_direction == "Bullish" else "#F44336"
            st.markdown(f"""
<div style="background:#0a0a0a;border:1px solid #1a1a2e;border-radius:10px;padding:20px 24px;margin:10px 0">
<div style="color:{_d_col};font-size:1.3em;font-weight:bold;margin-bottom:12px">{_d_icon} — {m_comm}</div>
<table style="width:100%;color:#ccc;font-size:0.9em;border-collapse:collapse">
<tr><td style="padding:4px 12px 4px 0;color:#666">Entry Zone</td>
    <td><b>${_man_setup.entry_low:,.3f} – ${_man_setup.entry_high:,.3f}</b></td>
    <td style="padding:4px 12px 4px 24px;color:#666">Target (TP)</td>
    <td><b style="color:#00C853">${_man_setup.target_price:,.3f} (+{_man_setup.target_move_pct:.2f}%)</b></td></tr>
<tr><td style="padding:4px 12px 4px 0;color:#666">Stop Loss</td>
    <td><b style="color:#F44336">${_man_setup.stop_loss:,.3f} (-{_man_setup.stop_loss_pct:.1f}%)</b></td>
    <td style="padding:4px 12px 4px 24px;color:#666">R:R Ratio</td>
    <td><b>{_man_setup.risk_reward:.2f}:1</b></td></tr>
<tr><td style="padding:4px 12px 4px 0;color:#666">Confidence</td>
    <td><b>{_man_setup.confidence_pct:.0f}%</b></td>
    <td style="padding:4px 12px 4px 24px;color:#666">Win Rate (hist.)</td>
    <td><b>{_man_setup.win_rate*100:.0f}%</b></td></tr>
<tr><td style="padding:4px 12px 4px 0;color:#666">Risk / Reward $</td>
    <td><b>${_man_setup.risk_usd:,.0f} / ${_man_setup.reward_usd:,.0f}</b></td>
    <td style="padding:4px 12px 4px 24px;color:#666">Expected Value</td>
    <td><b style="color:{'#00C853' if _man_setup.expected_value > 0 else '#F44336'}">${_man_setup.expected_value:+,.0f}</b></td></tr>
</table>
<div style="margin-top:12px;color:#888;font-size:0.82em">💡 {_man_setup.rationale}</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Comparable Events
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Comparable Historical Events")
    st.caption("ChromaDB semantic search — find historical articles similar to today's top headlines.")

    try:
        from inference.vector_store import find_similar
        vector_store_ok = True
    except Exception as e:
        vector_store_ok = False
        st.warning(f"Vector store unavailable: {e}")

    top3_headlines = [
        s["top_headline"] for s in st.session_state.today_signals if s.get("top_headline")
    ][:3]

    if not top3_headlines:
        st.info("No headlines available. Refresh data to fetch today's articles.")
    elif not vector_store_ok:
        st.info("Index articles first: `python inference/vector_store.py`")
    else:
        for i, hl in enumerate(top3_headlines):
            with st.expander(f"📰 {hl[:100]}{'...' if len(hl) > 100 else ''}", expanded=(i == 0)):
                st.markdown(f"**Current headline:** {hl}")
                st.markdown("---")
                try:
                    similar = find_similar(hl, n=3)
                    if not similar:
                        st.info("No similar historical articles found. Run `python inference/vector_store.py` to index articles.")
                    else:
                        for j, s in enumerate(similar):
                            sim_pct = round(s["similarity"] * 100, 1)
                            lbl = s.get("label", "—")
                            move = s.get("price_change", 0)
                            lbl_color = {"HIGH": "#F44336", "MEDIUM": "#FFB300", "LOW": "#00C853"}.get(lbl, "#888")
                            move_color = "#00C853" if move > 0 else ("#F44336" if move < 0 else "#888")
                            st.markdown(f"""
<div style="background:#111;border-left:3px solid #333;border-radius:6px;
padding:10px 14px;margin:6px 0">
<div style="color:#aaa;font-size:0.75em;margin-bottom:4px">
#{j+1} · {s.get('date','—')} · Similarity: <b style="color:#58a6ff">{sim_pct}%</b>
&nbsp;|&nbsp; Signal: <b style="color:{lbl_color}">{lbl}</b>
&nbsp;|&nbsp; Actual move: <b style="color:{move_color}">{move:+.2f}%</b>
</div>
<div style="color:#e6edf3">{s.get('headline','')}</div>
</div>""", unsafe_allow_html=True)
                except Exception as e:
                    err_str = str(e)
                    if "SSL" in err_str or "certificate" in err_str.lower() or "huggingface" in err_str.lower():
                        st.warning(
                            "⚠️ Cannot connect to HuggingFace to load the embedding model — "
                            "likely a corporate SSL/proxy issue. "
                            "Run this once on a network without SSL inspection to cache the model locally: "
                            "`python market_predictor/inference/vector_store.py`"
                        )
                    else:
                        st.error(f"Similarity search failed: {e}")

    st.divider()
    st.caption(
        "Historical accuracy note: similarity scores above 70% indicate strong precedent. "
        "Past price moves from similar events are shown for reference only and do not guarantee future outcomes."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Accuracy Report
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Accuracy Report")

    period_opts = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": 9999}
    period_sel = st.selectbox("Period", list(period_opts.keys()), index=1, key="t6_period")
    period_days = period_opts[period_sel]
    since6 = (date.today() - timedelta(days=period_days)).isoformat() if period_days < 9999 else "2000-01-01"

    df6 = query_db(
        "SELECT * FROM predictions WHERE prediction_date>=? ORDER BY prediction_date ASC",
        (since6,),
    )

    if df6.empty:
        st.info("No predictions found for the selected period.")
    else:
        resolved6 = df6[df6["outcome"].notna()]
        correct6 = (resolved6["outcome"] == "correct").sum() if not resolved6.empty else 0
        incorrect6 = (resolved6["outcome"] == "incorrect").sum() if not resolved6.empty else 0
        pending6 = df6["outcome"].isna().sum()
        total6 = len(df6)
        overall_acc = round(correct6 / len(resolved6) * 100, 1) if len(resolved6) > 0 else 0

        def _sig_acc(sig_label):
            sub = resolved6[resolved6["signal"] == sig_label] if not resolved6.empty else pd.DataFrame()
            return round((sub["outcome"] == "correct").sum() / len(sub) * 100, 1) if len(sub) > 0 else 0

        high_acc = _sig_acc("HIGH")
        med_acc = _sig_acc("MEDIUM")
        low_acc = _sig_acc("LOW")

        # KPI row
        k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
        k1.metric("Total", total6)
        k2.metric("Correct", int(correct6))
        k3.metric("Incorrect", int(incorrect6))
        k4.metric("Pending", int(pending6))
        k5.metric("Overall %", f"{overall_acc}%")
        k6.metric("HIGH %", f"{high_acc}%")
        k7.metric("MEDIUM %", f"{med_acc}%")
        k8.metric("LOW %", f"{low_acc}%")
        st.divider()

        if resolved6.empty:
            st.info("No resolved predictions yet — run backfill to populate exit prices and outcomes.")
        else:
            c1, c2 = st.columns(2)

            # Accuracy by signal bar chart with 80% target line
            with c1:
                sig_acc_df = pd.DataFrame({
                    "Signal": ["HIGH", "MEDIUM", "LOW"],
                    "Accuracy": [high_acc, med_acc, low_acc],
                })
                fig_sig = go.Figure()
                fig_sig.add_trace(go.Bar(
                    x=sig_acc_df["Signal"], y=sig_acc_df["Accuracy"],
                    marker_color=["#F44336", "#FFB300", "#00C853"],
                    name="Accuracy",
                ))
                fig_sig.add_hline(y=80, line_dash="dash", line_color="#58a6ff",
                                  annotation_text="80% target", annotation_position="top right")
                fig_sig.update_layout(title="Accuracy by Signal Level", yaxis_range=[0, 100],
                                      yaxis_title="Accuracy %", **PLOTLY_DEFAULTS)
                st.plotly_chart(fig_sig, use_container_width=True)

            # Accuracy by commodity bar chart
            with c2:
                acc_by_comm6 = (
                    resolved6.groupby("commodity")
                    .apply(lambda x: round((x["outcome"] == "correct").sum() / len(x) * 100, 1))
                    .reset_index(name="accuracy")
                )
                colors6 = acc_by_comm6["accuracy"].apply(
                    lambda v: "#00C853" if v >= 80 else ("#FFB300" if v >= 60 else "#F44336")).tolist()
                fig_comm6 = go.Figure(go.Bar(
                    x=acc_by_comm6["accuracy"], y=acc_by_comm6["commodity"],
                    orientation="h", marker_color=colors6,
                ))
                fig_comm6.add_vline(x=80, line_dash="dash", line_color="#58a6ff",
                                    annotation_text="80% target")
                fig_comm6.update_layout(title="Accuracy by Commodity", xaxis_range=[0, 100],
                                        xaxis_title="Accuracy %", **PLOTLY_DEFAULTS)
                st.plotly_chart(fig_comm6, use_container_width=True)

            # Cumulative P&L line chart
            if "pnl_usd" in df6.columns and df6["pnl_usd"].notna().any():
                st.markdown("**Cumulative P&L**")
                pnl_df = df6[df6["pnl_usd"].notna()].copy()
                pnl_df = pnl_df.sort_values("prediction_date")
                fig_pnl = go.Figure()
                # Per commodity
                for comm in COMM_NAMES:
                    sub_pnl = pnl_df[pnl_df["commodity"] == comm]
                    if sub_pnl.empty:
                        continue
                    sub_pnl = sub_pnl.copy()
                    sub_pnl["cum_pnl"] = sub_pnl["pnl_usd"].cumsum()
                    fig_pnl.add_trace(go.Scatter(
                        x=sub_pnl["prediction_date"], y=sub_pnl["cum_pnl"],
                        mode="lines", name=comm,
                        line=dict(color=COMMODITY_COLORS.get(comm, "#888"), width=1.5),
                    ))
                # Combined
                pnl_combined = pnl_df.groupby("prediction_date")["pnl_usd"].sum().cumsum().reset_index()
                fig_pnl.add_trace(go.Scatter(
                    x=pnl_combined["prediction_date"], y=pnl_combined["pnl_usd"],
                    mode="lines", name="Combined",
                    line=dict(color="#fff", width=2.5, dash="dot"),
                ))
                fig_pnl.update_layout(title="Cumulative P&L by Commodity",
                                      yaxis_title="P&L (USD)", **PLOTLY_DEFAULTS)
                st.plotly_chart(fig_pnl, use_container_width=True)

            # Best / worst calls
            if "pnl_usd" in resolved6.columns and resolved6["pnl_usd"].notna().any():
                bc, wc = st.columns(2)
                with bc:
                    st.markdown("**Top 5 Best Calls**")
                    best = resolved6[resolved6["pnl_usd"].notna()].nlargest(5, "pnl_usd")[
                        ["prediction_date", "commodity", "signal", "actual_move", "pnl_usd"]
                    ].copy()
                    best["pnl_usd"] = best["pnl_usd"].apply(lambda x: f"${x:+,.0f}")
                    best["actual_move"] = best["actual_move"].apply(
                        lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
                    st.dataframe(best, use_container_width=True, hide_index=True)
                with wc:
                    st.markdown("**Top 5 Worst Calls**")
                    worst = resolved6[resolved6["pnl_usd"].notna()].nsmallest(5, "pnl_usd")[
                        ["prediction_date", "commodity", "signal", "actual_move", "pnl_usd"]
                    ].copy()
                    worst["pnl_usd"] = worst["pnl_usd"].apply(lambda x: f"${x:+,.0f}")
                    worst["actual_move"] = worst["actual_move"].apply(
                        lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
                    st.dataframe(worst, use_container_width=True, hide_index=True)

    st.divider()

    # ── Risk Metrics Panel (Change 6e) ────────────────────────────────────────
    st.markdown("### Risk-Adjusted Performance")
    st.caption("For fundamental traders. Requires 5+ resolved trades to be meaningful.")

    if not df6.empty and "pnl_usd" in df6.columns:
        _pnl_series = df6["pnl_usd"].dropna().tolist()
        _wins_pnl = [p for p in _pnl_series if p > 0]
        _losses_pnl = [p for p in _pnl_series if p < 0]

        def _sharpe(series, rf=0.05):
            import numpy as np
            if len(series) < 5: return None
            r = np.array(series)
            # Normalise to % return (assume avg $80 entry × 1000 lot)
            avg_entry = df6["entry_price"].dropna().mean() or 80.0
            r_pct = r / (avg_entry * 1000) * 100
            daily_rf = rf / 252
            er = r_pct - daily_rf
            return round(float(np.mean(er) / er.std() * np.sqrt(252)), 2) if er.std() > 0 else None

        def _sortino(series, rf=0.05):
            import numpy as np
            if len(series) < 5: return None
            r = np.array(series)
            avg_entry = df6["entry_price"].dropna().mean() or 80.0
            r_pct = r / (avg_entry * 1000) * 100
            daily_rf = rf / 252
            er = r_pct - daily_rf
            down = er[er < 0]
            return round(float(np.mean(er) / down.std() * np.sqrt(252)), 2) if len(down) > 1 and down.std() > 0 else None

        def _wl_ratio(wins, losses):
            if not wins or not losses: return None
            return round(abs(sum(wins)/len(wins)) / abs(sum(losses)/len(losses)), 2)

        def _expectancy(wins, losses):
            if not wins and not losses: return None
            total = len(wins) + len(losses)
            wr = len(wins) / total; lr = len(losses) / total
            aw = sum(wins)/len(wins) if wins else 0
            al = abs(sum(losses)/len(losses)) if losses else 0
            return round(wr * aw - lr * al, 2)

        _sh = _sharpe(_pnl_series)
        _so = _sortino(_pnl_series)
        _wl = _wl_ratio(_wins_pnl, _losses_pnl)
        _ex = _expectancy(_wins_pnl, _losses_pnl)
        from inference.trade_filter import get_trades_this_month, MAX_TRADES_PER_MONTH
        _tm = get_trades_this_month()

        _rm1, _rm2, _rm3, _rm4, _rm5 = st.columns(5)
        _rm1.metric("Sharpe Ratio",
                    f"{_sh:.2f}" if _sh else "—",
                    delta="Good > 1.0" if _sh and _sh > 1.0 else ("Need 5+ trades" if not _sh else "Below 1.0"))
        _rm2.metric("Sortino Ratio",
                    f"{_so:.2f}" if _so else "—",
                    delta="Good > 1.5" if _so and _so > 1.5 else None)
        _rm3.metric("Win/Loss Ratio",
                    f"{_wl:.1f}x" if _wl else "—",
                    delta="Good > 1.5x" if _wl and _wl > 1.5 else ("Below 1.5x" if _wl else None))
        _rm4.metric("Expectancy",
                    f"${_ex:,.0f}/trade" if _ex is not None else "—",
                    delta="Positive edge" if _ex and _ex > 0 else "No edge yet")
        _rm5.metric("Trades this month",
                    f"{_tm} / {MAX_TRADES_PER_MONTH}",
                    delta=f"{MAX_TRADES_PER_MONTH - _tm} remaining")

    st.divider()
    st.caption(
        "For fundamental analysts. Accuracy = correct direction predictions / resolved predictions. "
        "Sharpe/Sortino use daily P&L per trade, annualised (×√252). "
        "Expectancy = average expected P&L per trade. Positive = model has edge."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Prediction Accuracy (ML Classification Evaluation)
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    from inference.evaluation import load_evaluation_df, compute_accuracy_metrics

    st.markdown("## 📊 Prediction Accuracy (Model Evaluation)")
    st.caption(
        "Evaluates: **Did the model correctly predict the market movement direction?** "
        "This is pure ML classification evaluation — no P&L, no LONG/SHORT trades."
    )

    _ev_conn = sqlite3.connect(DB_PATH)
    df_ev = load_evaluation_df(_ev_conn)
    _ev_conn.close()

    acc_m = compute_accuracy_metrics(df_ev)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    acc_color = "#00C853" if acc_m["accuracy_pct"] >= 55 else ("#FFB300" if acc_m["accuracy_pct"] >= 45 else "#F44336")
    hp_color  = "#00C853" if acc_m["high_precision_pct"] >= 60 else ("#FFB300" if acc_m["high_precision_pct"] >= 45 else "#F44336")

    def _ev_kpi(label, value, subtitle, color="#e0e0e0"):
        return (
            f'<div style="background:#0a0a0a;border:1px solid #1a1a2e;border-radius:6px;'
            f'padding:12px 16px;text-align:center">'
            f'<div style="color:#666;font-size:0.72em;text-transform:uppercase;letter-spacing:1px">{label}</div>'
            f'<div style="color:{color};font-size:1.8em;font-weight:bold;font-family:monospace">{value}</div>'
            f'<div style="color:#444;font-size:0.7em">{subtitle}</div>'
            f'</div>'
        )

    ek1, ek2, ek3, ek4, ek5 = st.columns(5)
    with ek1:
        st.markdown(_ev_kpi("Evaluated", str(acc_m["total_evaluated"]), "with entry+exit price"), unsafe_allow_html=True)
    with ek2:
        st.markdown(_ev_kpi("Accuracy %", f"{acc_m['accuracy_pct']}%", "correct direction", acc_color), unsafe_allow_html=True)
    with ek3:
        st.markdown(_ev_kpi("HIGH Precision", f"{acc_m['high_precision_pct']}%", "of HIGH predictions correct", hp_color), unsafe_allow_html=True)
    with ek4:
        st.markdown(_ev_kpi("HIGH Recall", f"{acc_m['high_recall_pct']}%", "large moves caught", "#e0e0e0"), unsafe_allow_html=True)
    with ek5:
        st.markdown(_ev_kpi("Pending", str(acc_m["total_pending"]), "missing prices", "#FFB300"), unsafe_allow_html=True)

    st.info(
        "**Accuracy** = predicted direction matches actual direction. "
        "Actual direction: >+1% = rise, <-1% = fall, else = flat. "
        "This is independent of trading profitability."
    )
    st.divider()

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = acc_m.get("confusion_matrix", {})
    if cm:
        st.markdown("### Confusion Matrix")
        st.caption("Rows = actual movement · Columns = predicted direction")
        labels_cm = ["rise", "flat", "fall"]
        cm_data = {
            "Actual \\ Predicted": labels_cm,
        }
        for pred in labels_cm:
            cm_data[f"Pred: {pred}"] = [cm.get(actual, {}).get(pred, 0) for actual in labels_cm]
        cm_df = pd.DataFrame(cm_data).set_index("Actual \\ Predicted")

        # Heatmap via Plotly
        z_vals = [[cm.get(a, {}).get(p, 0) for p in labels_cm] for a in labels_cm]
        fig_cm = go.Figure(go.Heatmap(
            z=z_vals,
            x=[f"Pred: {p}" for p in labels_cm],
            y=[f"Actual: {a}" for a in labels_cm],
            colorscale=[[0, "#0a0a0a"], [0.5, "#1a3a5c"], [1, "#00C853"]],
            text=z_vals,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            height=280,
            **PLOTLY_DEFAULTS,
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    st.divider()

    # ── Accuracy by Commodity ─────────────────────────────────────────────────
    by_comm = acc_m.get("by_commodity", {})
    by_sig  = acc_m.get("by_signal", {})

    if by_comm or by_sig:
        bc_col, bs_col = st.columns(2)

        with bc_col:
            st.markdown("### Accuracy by Commodity")
            if by_comm:
                comm_rows = [
                    {"Commodity": k, "Correct": v["correct"], "Total": v["total"], "Accuracy %": v["accuracy"]}
                    for k, v in by_comm.items()
                ]
                comm_df = pd.DataFrame(comm_rows).sort_values("Accuracy %", ascending=False)
                bar_colors = comm_df["Accuracy %"].apply(
                    lambda v: "#00C853" if v >= 55 else ("#FFB300" if v >= 45 else "#F44336")
                ).tolist()
                fig_bc = go.Figure(go.Bar(
                    x=comm_df["Accuracy %"], y=comm_df["Commodity"],
                    orientation="h", marker_color=bar_colors,
                    text=comm_df["Accuracy %"].apply(lambda v: f"{v}%"),
                    textposition="outside",
                ))
                fig_bc.add_vline(x=50, line_dash="dash", line_color="#555",
                                 annotation_text="50% baseline")
                fig_bc.update_layout(
                    xaxis_range=[0, 100], xaxis_title="Accuracy %",
                    height=280, **PLOTLY_DEFAULTS,
                )
                st.plotly_chart(fig_bc, use_container_width=True)

        with bs_col:
            st.markdown("### Accuracy by Signal Level")
            if by_sig:
                sig_rows = [
                    {"Signal": k, "Correct": v["correct"], "Total": v["total"], "Accuracy %": v["accuracy"]}
                    for k, v in by_sig.items()
                ]
                sig_df = pd.DataFrame(sig_rows)
                sig_colors = {"HIGH": "#F44336", "MEDIUM": "#FFB300", "LOW": "#00C853"}
                fig_bs = go.Figure(go.Bar(
                    x=sig_df["Signal"],
                    y=sig_df["Accuracy %"],
                    marker_color=[sig_colors.get(s, "#888") for s in sig_df["Signal"]],
                    text=sig_df["Accuracy %"].apply(lambda v: f"{v}%"),
                    textposition="outside",
                ))
                fig_bs.add_hline(y=50, line_dash="dash", line_color="#555",
                                 annotation_text="50% baseline")
                fig_bs.update_layout(
                    yaxis_range=[0, 100], yaxis_title="Accuracy %",
                    height=280, **PLOTLY_DEFAULTS,
                )
                st.plotly_chart(fig_bs, use_container_width=True)

    st.divider()

    # ── Accuracy over time ────────────────────────────────────────────────────
    if not df_ev.empty:
        ev_time = df_ev[df_ev["status"] == "evaluated"].copy()
        if len(ev_time) > 5 and "prediction_date" in ev_time.columns:
            st.markdown("### Accuracy Over Time (7-day rolling)")
            ev_time = ev_time.sort_values("prediction_date")
            ev_time["rolling_acc"] = (
                ev_time["is_correct"].astype(float).rolling(7, min_periods=1).mean() * 100
            )
            fig_time = go.Figure(go.Scatter(
                x=ev_time["prediction_date"], y=ev_time["rolling_acc"],
                mode="lines", line=dict(color="#58a6ff", width=2),
                hovertemplate="%{x}<br>Rolling Accuracy: %{y:.1f}%<extra></extra>",
            ))
            fig_time.add_hline(y=50, line_dash="dash", line_color="#555",
                               annotation_text="50% baseline")
            fig_time.update_layout(
                yaxis_title="Accuracy %", yaxis_range=[0, 100],
                height=280, **PLOTLY_DEFAULTS,
            )
            st.plotly_chart(fig_time, use_container_width=True)
            st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("### Prediction Detail Table")
    ef1, ef2, ef3, ef4, ef5 = st.columns([1.5, 1, 1, 1, 1])
    with ef1:
        ev_comm = st.selectbox("Commodity", ["All"] + COMM_NAMES, key="ev_comm")
    with ef2:
        ev_from = st.date_input("From", value=date.today() - timedelta(days=30), key="ev_from")
    with ef3:
        ev_to = st.date_input("To", value=date.today(), key="ev_to")
    with ef4:
        ev_sig = st.selectbox("Signal", ["All", "HIGH", "MEDIUM", "LOW"], key="ev_sig")
    with ef5:
        ev_correct = st.selectbox("Result", ["All", "Correct", "Wrong", "Pending"], key="ev_correct")

    df_ev_filt = df_ev.copy() if not df_ev.empty else pd.DataFrame()
    if not df_ev_filt.empty:
        if ev_comm != "All":
            df_ev_filt = df_ev_filt[df_ev_filt["commodity"] == ev_comm]
        if ev_from:
            df_ev_filt = df_ev_filt[df_ev_filt["prediction_date"] >= str(ev_from)]
        if ev_to:
            df_ev_filt = df_ev_filt[df_ev_filt["prediction_date"] <= str(ev_to)]
        if ev_sig != "All":
            df_ev_filt = df_ev_filt[df_ev_filt["signal"] == ev_sig]
        if ev_correct == "Correct":
            df_ev_filt = df_ev_filt[df_ev_filt["is_correct"] == True]
        elif ev_correct == "Wrong":
            df_ev_filt = df_ev_filt[df_ev_filt["is_correct"] == False]
        elif ev_correct == "Pending":
            df_ev_filt = df_ev_filt[df_ev_filt["status"] == "pending"]

    if df_ev_filt.empty:
        st.info("No predictions match the current filters.")
    else:
        tbl_ev_cols = ["prediction_date", "commodity", "signal", "direction",
                       "actual_direction", "return_pct", "confidence", "is_correct"]
        tbl_ev_cols = [c for c in tbl_ev_cols if c in df_ev_filt.columns]
        disp_ev = df_ev_filt[tbl_ev_cols].copy()

        # Format
        if "return_pct" in disp_ev.columns:
            disp_ev["return_pct"] = disp_ev["return_pct"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "Pending")
        if "confidence" in disp_ev.columns:
            disp_ev["confidence"] = disp_ev["confidence"].apply(
                lambda x: f"{x:.1f}%" if (pd.notna(x) and x > 1.0) else (
                    f"{x*100:.1f}%" if pd.notna(x) else "—"))
        if "is_correct" in disp_ev.columns:
            disp_ev["is_correct"] = disp_ev["is_correct"].apply(
                lambda x: "✓ Correct" if x is True else (
                    "✗ Wrong" if x is False else "⏳ Pending"))

        disp_ev.columns = [
            {"prediction_date": "Date", "commodity": "Commodity", "signal": "Signal",
             "direction": "Predicted", "actual_direction": "Actual",
             "return_pct": "Return %", "confidence": "Confidence",
             "is_correct": "Result"}.get(c, c)
            for c in disp_ev.columns
        ]

        def _style_ev(row):
            styles = [""] * len(row)
            cols_l = list(row.index)
            ri = cols_l.index("Result") if "Result" in cols_l else None
            if ri is not None:
                v = str(row.iloc[ri])
                if "Correct" in v:  styles[ri] = "background-color:#001a09;color:#00C853"
                elif "Wrong" in v:  styles[ri] = "background-color:#1a0000;color:#F44336"
                else:               styles[ri] = "background-color:#1a1400;color:#FFB300"
            return styles

        st.dataframe(
            disp_ev.style.apply(_style_ev, axis=1),
            use_container_width=True, height=420,
        )

    st.caption(
        "Predicted direction: rise/fall/flat from model signal. "
        "Actual direction: >+1% = rise, <-1% = fall, else = flat. "
        "Accuracy is independent of P&L — a correct direction prediction can still lose money."
    )
