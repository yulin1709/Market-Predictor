"""
End-to-end headline prediction pipeline.

Supports two model modes:
  - text_tfidf_logreg: local text-only baseline
  - legacy embedding/entity model artifacts if they already exist

Usage:
    python inference/pipeline.py --headline "OPEC agrees to cut production"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SAVE_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"

EVENT_TYPES = [
    "pipeline_outage",
    "sanctions",
    "refinery_maintenance",
    "shipping_disruption",
    "demand_change",
    "opec_decision",
    "geopolitical",
    "other",
]
REGIONS = ["europe", "middle_east", "us", "russia", "asia", "global"]

_models: dict = {}
_commodity_models: dict[str, object] = {}   # commodity_group → fitted pipeline
_direction_model: dict = {}
_finbert_model: dict = {}
_ensemble_weights: dict = {}
ENABLE_SIMILARITY_SEARCH = os.getenv("ENABLE_SIMILARITY_SEARCH", "").strip().lower() in {"1", "true", "yes"}

# Commodity group keyword map — mirrors train.py COMMODITY_GROUPS
_COMMODITY_KEYWORDS: dict[str, list[str]] = {
    "dubai": ["dubai", "oman", "saudi", "abu dhabi", "kuwait", "iraq", "iran",
              "middle east", "gulf", "opec", "arab", "murban", "basrah",
              "pcaat", "pcaas", "arabian", "sour crude"],
    "brent": ["brent", "north sea", "dated brent", "forties", "oseberg", "ekofisk",
              "europe", "urals", "russia", "norway", "uk crude", "rotterdam",
              "mediterranean", "cif", "fob rotterdam"],
    "wti":   ["wti", "west texas", "cushing", "shale", "permian", "eagle ford",
              "bakken", "us crude", "gulf coast", "usgc", "houston", "midland",
              "light sweet", "american crude"],
    "lng":   ["lng", "natural gas", "liquefied", "jkm", "henry hub", "ttf",
              "nbp", "regasification", "cargo", "tanker", "vessel", "shipping",
              "gas price", "spot gas", "lng terminal", "lng export"],
}

# Map commodity display names → commodity group key
_COMMODITY_NAME_TO_GROUP: dict[str, str] = {
    "Dubai Crude (PCAAT00)": "dubai",
    "Dubai Crude":           "dubai",
    "Brent Dated (PCAAS00)": "brent",
    "Brent":                 "brent",
    "WTI (PCACG00)":         "wti",
    "WTI":                   "wti",
    "LNG / Nat Gas (JKM)":   "lng",
    "LNG":                   "lng",
}


def _detect_commodity_group(headline: str) -> str:
    """Detect commodity group from headline keywords."""
    text = headline.lower()
    scores = {g: sum(1 for kw in kws if kw in text)
              for g, kws in _COMMODITY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"



def _load_ensemble_weights() -> dict:
    """
    Change 4: Load per-commodity historical win rates from predictions table.
    Used to weight each commodity model's vote in the ensemble.
    Falls back to equal weights if < 10 predictions per commodity.
    """
    try:
        import sqlite3 as _sq
        db_path = Path(__file__).resolve().parent.parent / "data" / ("articles.db" if (Path(__file__).resolve().parent.parent / "data" / "articles.db").exists() else "articles_deploy.db")
        conn = _sq.connect(db_path)
        rows = conn.execute("""
            SELECT commodity,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END) as correct
            FROM predictions
            WHERE outcome IN ('correct','incorrect')
            GROUP BY commodity
        """).fetchall()
        conn.close()
        weights = {}
        for comm, total, correct in rows:
            if total >= 10:
                group = _COMMODITY_NAME_TO_GROUP.get(comm, "")
                if group:
                    weights[group] = round(correct / total, 3)
        return weights
    except Exception:
        return {}


def _load_models() -> dict:
    global _models, _commodity_models, _direction_model, _finbert_model, _ensemble_weights
    if _models:
        return _models

    bundle = joblib.load(SAVE_DIR / "classifier.pkl")
    model_info_path = SAVE_DIR / "model_info.pkl"
    model_info = joblib.load(model_info_path) if model_info_path.exists() else {}

    if isinstance(bundle, dict):
        _models = {**bundle, "model_info": model_info}
    else:
        _models = {"clf": bundle, "tfidf": None, "scaler": None,
                   "model_type": "text_tfidf_logreg", "model_info": model_info}

    # Load commodity-specific models
    for commodity in ("dubai", "brent", "wti", "lng"):
        comm_path = SAVE_DIR / f"classifier_{commodity}.pkl"
        if comm_path.exists():
            try:
                cb = joblib.load(comm_path)
                _commodity_models[commodity] = cb if isinstance(cb, dict) else {"clf": cb, "tfidf": None, "scaler": None}
            except Exception as e:
                print(f"[pipeline] Could not load {commodity} model: {e}")

    # Change 3: Load directional model
    dir_path = SAVE_DIR / "classifier_direction.pkl"
    if dir_path.exists():
        try:
            _direction_model.update(joblib.load(dir_path))
            print("[pipeline] Loaded directional model")
        except Exception as e:
            print(f"[pipeline] Could not load directional model: {e}")

    # Change 1: Load FinBERT if available
    finbert_dir = SAVE_DIR / "classifier_finbert"
    if finbert_dir.exists() and (finbert_dir / "config.json").exists():
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            import torch
            _finbert_model["tokenizer"] = BertTokenizer.from_pretrained(str(finbert_dir))
            _finbert_model["model"] = BertForSequenceClassification.from_pretrained(str(finbert_dir))
            _finbert_model["model"].eval()
            meta = joblib.load(finbert_dir / "meta.pkl")
            _finbert_model["label_encoder"] = meta["label_encoder"]
            _finbert_model["labels"] = meta["labels"]
            print("[pipeline] Loaded FinBERT model")
        except Exception as e:
            print(f"[pipeline] FinBERT unavailable: {e}")

    # Change 4: Load ensemble weights
    _ensemble_weights.update(_load_ensemble_weights())

    loaded_comm = list(_commodity_models.keys())
    model_type = _models.get("model_type") or model_info.get("model_type", "unknown")
    print(f"[pipeline] Loaded model type: {model_type}")
    if loaded_comm:
        print(f"[pipeline] Commodity-specific models: {loaded_comm}")
    if _ensemble_weights:
        print(f"[pipeline] Ensemble weights: {_ensemble_weights}")
    return _models


def _default_entities(headline: str) -> dict:
    text = headline.lower()
    event_type = "other"
    region = "global"
    sentiment = 0.0
    supply_impact = False

    if "opec" in text:
        event_type = "opec_decision"
    elif "sanction" in text:
        event_type = "sanctions"
        supply_impact = True
    elif "refinery" in text and ("outage" in text or "maintenance" in text):
        event_type = "refinery_maintenance"
    elif any(word in text for word in ["pipeline", "terminal", "port shutdown"]):
        event_type = "pipeline_outage"
        supply_impact = True
    elif any(word in text for word in ["shipping", "tanker", "freight", "strait"]):
        event_type = "shipping_disruption"
    elif any(word in text for word in ["demand", "import", "export", "buying", "tender"]):
        event_type = "demand_change"

    if any(word in text for word in ["dubai", "oman", "saudi", "abu dhabi", "middle east", "gulf"]):
        region = "middle_east"
    elif any(word in text for word in ["china", "singapore", "japan", "korea", "asia"]):
        region = "asia"
    elif any(word in text for word in ["us", "wti", "gulf coast", "cushing"]):
        region = "us"
    elif any(word in text for word in ["russia", "urals", "moscow"]):
        region = "russia"
    elif any(word in text for word in ["europe", "brent", "north sea", "rotterdam"]):
        region = "europe"

    if any(word in text for word in ["cut", "outage", "disruption", "sanction", "attack", "tight"]):
        sentiment = 0.35
    elif any(word in text for word in ["build", "restart", "surplus", "weak", "slower"]):
        sentiment = -0.25

    return {
        "event_type": event_type,
        "region": region,
        "sentiment": sentiment,
        "supply_impact": supply_impact,
    }


def _one_hot(val: str, cats: list[str]) -> list[int]:
    return [1 if val == c else 0 for c in cats]


def _build_legacy_vector(headline: str, entities: dict, models: dict) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = embed_model.encode([headline], normalize_embeddings=True)
    compressed = models["pca"].transform(embedding)
    structured = (
        _one_hot(entities.get("event_type", "other"), EVENT_TYPES)
        + _one_hot(entities.get("region", "global"), REGIONS)
        + [float(entities.get("sentiment", 0.0))]
        + [int(bool(entities.get("supply_impact", False)))]
    )
    return np.hstack([structured, compressed[0]]).astype(np.float32).reshape(1, -1)


def _get_numeric_features() -> list[float]:
    """
    Fetch current price momentum features for inference.
    Returns list matching NUMERIC_COLS order.
    Falls back to zeros if unavailable.
    """
    try:
        import sqlite3 as _sq
        from datetime import date as _date
        db_path = Path(__file__).resolve().parent.parent / "data" / ("articles.db" if (Path(__file__).resolve().parent.parent / "data" / "articles.db").exists() else "articles_deploy.db")
        conn = _sq.connect(db_path)
        rows = conn.execute("""
            SELECT price FROM prices WHERE symbol='Dubai'
            ORDER BY date DESC LIMIT 11
        """).fetchall()
        art_count = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE date(published_at)=date('now')"
        ).fetchone()[0]
        conn.close()
        if len(rows) >= 6:
            prices = [r[0] for r in rows]
            mom_5d  = (prices[0] - prices[5])  / prices[5]  * 100 if prices[5]  else 0.0
            mom_10d = (prices[0] - prices[-1]) / prices[-1] * 100 if prices[-1] else 0.0
        else:
            mom_5d = mom_10d = 0.0
        import datetime as _dt
        dow = _dt.date.today().weekday()  # 0=Mon
        dow_vec = [1.0 if dow == i else 0.0 for i in range(5)]
        return [mom_5d, mom_10d] + dow_vec + [float(art_count)]
    except Exception:
        return [0.0] * 8  # len(NUMERIC_COLS)


# Cache for FinBERT embedding model (separate from fine-tuned FinBERT classifier)
_finbert_embed_model: dict = {}


def _embed_texts_finbert_inference(texts: list) -> "np.ndarray":
    """
    Extract FinBERT [CLS] embeddings for inference.
    Uses pre-trained ProsusAI/finbert weights — no fine-tuning.
    Cached after first call.
    """
    global _finbert_embed_model
    try:
        import torch
        from transformers import BertTokenizer, BertModel

        if not _finbert_embed_model:
            print("[pipeline] Loading FinBERT for embeddings...")
            _finbert_embed_model["tokenizer"] = BertTokenizer.from_pretrained("ProsusAI/finbert")
            _finbert_embed_model["model"] = BertModel.from_pretrained("ProsusAI/finbert")
            _finbert_embed_model["model"].eval()
            print("[pipeline] FinBERT embedder loaded.")

        tok = _finbert_embed_model["tokenizer"]
        model = _finbert_embed_model["model"]
        inputs = tok(list(texts), return_tensors="pt", truncation=True,
                     max_length=128, padding=True)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:, 0, :].numpy()
    except Exception as e:
        print(f"[pipeline] FinBERT embed error: {e}")
        return np.zeros((len(texts), 768))

def _predict_with_bundle(headline: str, bundle: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Run inference on a single model bundle. Returns (proba, classes) or None."""
    clf = bundle.get("clf")
    if clf is None:
        return None

    tfidf = bundle.get("tfidf")
    scaler = bundle.get("scaler")
    selector = bundle.get("selector")  # XGBoost uses SelectKBest
    le = bundle.get("le")              # XGBoost uses LabelEncoder

    # XGBoost bundle: tfidf + selector + scaler + le
    if selector is not None and le is not None and tfidf is not None:
        try:
            X_text = tfidf.transform([headline])
            X_text_sel = selector.transform(X_text).toarray()
            X_num = np.array(_get_numeric_features()).reshape(1, -1)
            if scaler is not None:
                X_num = scaler.transform(X_num)
            X = np.hstack([X_text_sel, X_num])
            proba = clf.predict_proba(X)[0]
            return proba, np.array(le.classes_)
        except Exception as e:
            print(f"[pipeline] XGBoost inference error: {e}")
            return None

    # LogReg bundle with tfidf + scaler + numeric features
    if tfidf is not None and scaler is not None:
        import scipy.sparse as sp
        X_text = tfidf.transform([headline])
        X_num = scaler.transform([_get_numeric_features()])
        X = sp.hstack([X_text, sp.csr_matrix(X_num)])
        proba = clf.predict_proba(X)[0]
        return proba, np.array(clf.classes_)

    # Bundle with tfidf only
    if tfidf is not None:
        proba = clf.predict_proba(tfidf.transform([headline]))[0]
        return proba, np.array(clf.classes_)

    # sklearn Pipeline
    if hasattr(clf, "steps"):
        proba = clf.predict_proba([headline])[0]
        return proba, np.array(clf.classes_)

    # Direct LogReg
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba([headline])[0]
        return proba, np.array(clf.classes_)

    return None


def _predict_finbert(headline: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Run FinBERT inference. Returns (proba, classes) or None."""
    if not _finbert_model:
        return None
    try:
        import torch
        tokenizer = _finbert_model["tokenizer"]
        model = _finbert_model["model"]
        le = _finbert_model["label_encoder"]
        labels = _finbert_model["labels"]
        inputs = tokenizer(headline, return_tensors="pt", truncation=True,
                           max_length=128, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[0].numpy()
        return proba, np.array(labels)
    except Exception as e:
        print(f"[pipeline] FinBERT inference error: {e}")
        return None


def _predict_probabilities(headline: str, entities: dict, models: dict,
                           commodity: str = "") -> tuple[np.ndarray, np.ndarray]:
    """
    Change 4: Ensemble voting across all available models.
    Weights each model by historical win rate (from predictions table).
    Falls back to equal weights if insufficient history.
    """
    # Collect all model bundles to vote
    bundles: list[tuple[dict, float]] = []  # (bundle, weight)

    # General model
    general_weight = 1.0
    bundles.append((_models, general_weight))

    # Commodity-specific models
    if _commodity_models:
        group = _COMMODITY_NAME_TO_GROUP.get(commodity, "") if commodity else ""
        if not group or group not in _commodity_models:
            group = _detect_commodity_group(headline)
        for comm_group, comm_bundle in _commodity_models.items():
            w = _ensemble_weights.get(comm_group, 1.0)
            bundles.append((comm_bundle if isinstance(comm_bundle, dict) else {"clf": comm_bundle}, w))

    # Collect probabilities
    all_probas: list[np.ndarray] = []
    all_weights: list[float] = []
    classes_ref: np.ndarray | None = None

    for bundle, weight in bundles:
        result = _predict_with_bundle(headline, bundle)
        if result is None:
            continue
        proba, classes = result
        if classes_ref is None:
            classes_ref = classes
        # Align class order
        if list(classes) != list(classes_ref):
            aligned = np.zeros(len(classes_ref))
            for i, c in enumerate(classes_ref):
                idx = np.where(classes == c)[0]
                if len(idx):
                    aligned[i] = proba[idx[0]]
            proba = aligned
        all_probas.append(proba)
        all_weights.append(weight)

    # FinBERT vote (Change 1)
    finbert_result = _predict_finbert(headline)
    if finbert_result is not None:
        fb_proba, fb_classes = finbert_result
        if classes_ref is None:
            classes_ref = fb_classes
        all_probas.append(fb_proba)
        all_weights.append(1.5)  # FinBERT gets slightly higher weight

    if not all_probas:
        # Ultimate fallback
        classes_ref = np.array(["HIGH", "LOW", "MEDIUM"])
        return np.array([0.33, 0.34, 0.33]), classes_ref

    # Weighted average
    total_w = sum(all_weights)
    weights_norm = [w / total_w for w in all_weights]
    ensemble_proba = sum(p * w for p, w in zip(all_probas, weights_norm))

    return ensemble_proba, classes_ref


def predict(headline: str, skip_explanation: bool = False,
            commodity: str = "") -> dict:
    t0 = time.time()
    models = _load_models()
    entities = _default_entities(headline)
    proba, classes = _predict_probabilities(headline, entities, models, commodity=commodity)

    label_int = int(np.argmax(proba))
    label = str(classes[label_int])
    score = float(proba[label_int])
    class_probs = {str(c): float(p) for c, p in zip(classes, proba)}

    # Change 3: Use directional model for direction field if available
    direction_from_model = None
    if _direction_model:
        dir_result = _predict_with_bundle(headline, _direction_model)
        if dir_result is not None:
            dir_proba, dir_classes = dir_result
            dir_label = str(dir_classes[int(np.argmax(dir_proba))])
            direction_from_model = dir_label  # "rise", "fall", or "flat"

    similar = []
    chroma_path = Path(__file__).resolve().parent.parent / "chroma_db"
    if ENABLE_SIMILARITY_SEARCH and chroma_path.exists() and any(chroma_path.iterdir()):
        try:
            from inference.vector_store import find_similar
            similar = find_similar(headline, n=3)
        except Exception as exc:
            print(f"[pipeline] Similarity lookup unavailable: {exc}")

    explanation = ""
    if not skip_explanation:
        try:
            from inference.explain import generate_explanation
            explanation = generate_explanation(headline, label, score, entities, similar)
        except Exception as exc:
            explanation = f"Explanation unavailable: {exc}"

    latency_ms = round((time.time() - t0) * 1000, 1)
    comm_group = _COMMODITY_NAME_TO_GROUP.get(commodity, _detect_commodity_group(headline)) if commodity else _detect_commodity_group(headline)

    # Map direction to Bullish/Bearish/Neutral language
    DIRECTION_MAP = {"rise": "bullish", "fall": "bearish", "flat": "neutral"}
    raw_direction = direction_from_model or entities.get("event_type", "neutral")
    # direction_from_model comes from directional classifier (rise/fall/flat) — map it
    if direction_from_model:
        mapped_direction = DIRECTION_MAP.get(direction_from_model, direction_from_model)
    else:
        mapped_direction = None

    return {
        "headline": headline,
        "label": label,
        "score": score,
        "high_prob": class_probs.get("HIGH", 0.0),
        "medium_prob": class_probs.get("MEDIUM", 0.0),
        "low_prob": class_probs.get("LOW", 0.0),
        "explanation": explanation,
        "entities": entities,
        "similar_events": similar,
        "latency_ms": latency_ms,
        "model_type": models["model_info"].get("model_type", "legacy"),
        "commodity_model_used": comm_group if comm_group in _commodity_models else "general",
        "direction_from_model": mapped_direction,  # bullish/bearish/neutral
        "ensemble_weights": dict(_ensemble_weights),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline", required=True)
    parser.add_argument("--commodity", default="", help="Commodity name for model routing")
    args = parser.parse_args()

    result = predict(args.headline, commodity=args.commodity)
    print(json.dumps(result, indent=2))


# ── Article count guard ───────────────────────────────────────────────────────

MIN_ARTICLES_TO_PREDICT = 5  # below this, model output is unreliable (near-uniform probs)


def get_article_count_today() -> int:
    """
    Return the number of articles from the most recent date that has enough articles.
    Falls back up to 7 days to handle weekends and public holidays.
    """
    import sqlite3
    from datetime import date as _date, timedelta
    db_path = Path(__file__).resolve().parent.parent / "data" / ("articles.db" if (Path(__file__).resolve().parent.parent / "data" / "articles.db").exists() else "articles_deploy.db")
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        # Find the most recent date with at least MIN_ARTICLES_TO_PREDICT articles
        for days_back in range(0, 8):
            check_date = (_date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            count = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE date(published_at) = ?", (check_date,)
            ).fetchone()[0]
            if count >= MIN_ARTICLES_TO_PREDICT:
                conn.close()
                return int(count)
        # Return whatever we have from the most recent day
        row = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE date(published_at) = date('now', '-1 day')"
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def predict_with_guard(headline: str, article_count: int,
                       skip_explanation: bool = True,
                       commodity: str = "") -> dict | None:
    """
    Wrapper around predict() that returns None if article count is too low.
    """
    if article_count < MIN_ARTICLES_TO_PREDICT:
        return None
    return predict(headline, skip_explanation=skip_explanation, commodity=commodity)


def log_prediction(commodity: str, ticker: str, result: dict,
                   headline: str | None = None,
                   article_count: int = MIN_ARTICLES_TO_PREDICT) -> None:
    """
    Persist a prediction to the predictions table in articles.db.
    Fetches the current closing price for entry_price via fetch_price.py.
    Uses INSERT OR IGNORE to prevent duplicates (UNIQUE on prediction_date, commodity).

    Refuses to write a row if article_count < MIN_ARTICLES_TO_PREDICT — on low-news
    days the model output is unreliable and should not be stored as a real prediction.

    Args:
        commodity: Display name e.g. 'Dubai Crude (PCAAT00)'
        ticker: Platts symbol e.g. 'PCAAT00'
        result: dict from predict() or commodity signal dict
        headline: Top driving headline (optional)
        article_count: Number of articles fetched today (default = threshold so existing
                       callers that don't pass this arg still work)
    """
    if article_count < MIN_ARTICLES_TO_PREDICT:
        print(
            f"[log_prediction] Skipping {commodity} — only {article_count} articles today "
            f"(minimum {MIN_ARTICLES_TO_PREDICT} required). No row written."
        )
        return
    import sqlite3
    from datetime import date as date_cls
    from pathlib import Path

    db_path = Path(__file__).resolve().parent.parent / "data" / ("articles.db" if (Path(__file__).resolve().parent.parent / "data" / "articles.db").exists() else "articles_deploy.db")
    today = date_cls.today().isoformat()

    # Ensure table exists with all columns
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction_date  DATE NOT NULL,
            commodity        TEXT NOT NULL,
            ticker           TEXT NOT NULL,
            ticker_yf        TEXT,
            signal           TEXT NOT NULL,
            direction        TEXT NOT NULL,
            confidence       REAL NOT NULL,
            expected_move    TEXT NOT NULL,
            headline         TEXT,
            entry_price      REAL,
            exit_price       REAL,
            price_currency   TEXT DEFAULT 'USD',
            price_unit       TEXT,
            actual_move      REAL,
            outcome          TEXT,
            pnl_usd          REAL,
            UNIQUE(prediction_date, commodity)
        )
    """)

    # Migrate any missing columns
    for col, col_type in [("entry_price","REAL"),("exit_price","REAL"),
                           ("price_currency","TEXT DEFAULT 'USD'"),
                           ("price_unit","TEXT"),("ticker_yf","TEXT"),
                           ("data_quality","TEXT"),
                           ("entry_price_source","TEXT"),
                           ("exit_price_source","TEXT"),
                           ("trade_status","TEXT DEFAULT 'HOLD'"),
                           ("trade_number","INTEGER"),
                           ("event_type","TEXT"),
                           ("cut_loss_price","REAL"),
                           ("impact_score","INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    # Look up ticker config
    try:
        from data.ticker_config import COMMODITY_CONFIG
        cfg = COMMODITY_CONFIG.get(commodity, {})
        ticker_yf = cfg.get("ticker_yf", "")
        price_unit = cfg.get("unit", "barrel")
    except ImportError:
        ticker_yf = ""
        price_unit = "barrel"

    # Fetch entry price (two-tier: Platts → yfinance)
    entry_price = None
    entry_price_source = None
    try:
        from data.fetch_price import get_closing_price
        price_result = get_closing_price(commodity, today)
        if price_result:
            entry_price = price_result["price"]
            entry_price_source = price_result["source"]
    except Exception as e:
        print(f"[log_prediction] Price fetch failed for {commodity}: {e}")

    # Extract signal values — handle both pipeline result dict and commodity sig dict
    signal = result.get("label", result.get("signal", "LOW"))
    direction = result.get("direction", result.get("direction_from_model", "neutral"))
    # Normalise direction to bullish/bearish/neutral
    _dir_map = {"rise": "bullish", "fall": "bearish", "flat": "neutral"}
    direction = _dir_map.get(direction, direction)

    # Use raw model probability (high_prob) as confidence — not the scaled display value.
    # scaled_conf is an artificial 35-95% display number; high_prob is the actual model output.
    confidence = result.get("high_prob", result.get("confidence", 0))
    if confidence is None or confidence == 0:
        confidence = result.get("scaled_conf", 0)
    if confidence > 1.0:
        confidence = confidence / 100.0
    expected_move = result.get("move", result.get("expected_move", "<1%"))

    # Extract event_type and impact_score from entities
    entities = result.get("entities", {})
    event_type = entities.get("event_type", "other")
    impact_score = entities.get("impact_score", 2)

    # Compute cut_loss_price from entry_price
    cut_loss_price = None
    if entry_price:
        try:
            from data.ticker_config import STOP_LOSS_PCT
            sl_pct = STOP_LOSS_PCT.get(commodity, 1.5) / 100.0
            if direction == "bullish":
                cut_loss_price = round(entry_price * (1 - sl_pct), 4)
            elif direction == "bearish":
                cut_loss_price = round(entry_price * (1 + sl_pct), 4)
        except Exception:
            pass

    # Determine trade_status via quality filter
    try:
        from inference.trade_filter import should_trade
        trade_decision = should_trade(
            signal=signal,
            direction=direction,
            confidence=confidence,
            expected_move=expected_move,
            event_type=event_type,
        )
        trade_status = trade_decision["status"]
        trade_number = trade_decision.get("trade_number")
    except Exception:
        trade_status = "HOLD"
        trade_number = None

    conn.execute("""
        INSERT OR IGNORE INTO predictions
            (prediction_date, commodity, ticker, ticker_yf, signal, direction,
             confidence, expected_move, headline, entry_price, entry_price_source,
             price_unit, data_quality, trade_status, trade_number, event_type,
             cut_loss_price, impact_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?)
    """, (
        today, commodity, ticker, ticker_yf, signal, direction,
        confidence, expected_move, headline, entry_price, entry_price_source,
        price_unit, trade_status, trade_number, event_type,
        cut_loss_price, impact_score,
    ))
    conn.commit()
    conn.close()





