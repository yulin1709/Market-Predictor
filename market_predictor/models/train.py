"""
Train market-impact classifiers on labelled articles.

Flags:
  --model-type  tfidf (default) | finbert
  --label-type  magnitude (default, HIGH/MEDIUM/LOW) | direction (rise/flat/fall)
  --commodity   dubai | brent | wti | lng | "" (all)
  --no-commodity-models

Model variants saved to models/saved/:
  classifier.pkl              general TF-IDF model (magnitude)
  classifier_direction.pkl    TF-IDF directional model
  classifier_finbert/         FinBERT fine-tuned model directory
  classifier_{comm}.pkl       commodity-specific TF-IDF models
  model_info.pkl              CV metrics + metadata for all variants
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.db"
SAVE_DIR = Path(__file__).resolve().parent / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BODY_TEXT_CHARS = 400
MIN_SAMPLES = 40
MIN_FOLD_TRAIN = 50
MIN_FOLD_TEST = 20
N_CV_FOLDS = 4

COMMODITY_GROUPS: dict[str, list[str]] = {
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


# ── Non-text feature engineering (Change 2) ───────────────────────────────────

def _load_price_momentum(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Compute non-text features for each date in the prices table:
      - mom_5d:   5-day rolling return of Dubai Crude
      - mom_10d:  10-day rolling return of Dubai Crude
      - dow_mon .. dow_fri: one-hot day-of-week (5 features)
      - article_count: number of articles published on that date
    Returns DataFrame indexed by date string.
    """
    prices = pd.read_sql(
        "SELECT date, price FROM prices WHERE symbol='Dubai' ORDER BY date",
        conn
    )
    if prices.empty:
        return pd.DataFrame()

    prices["date_dt"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date_dt").reset_index(drop=True)
    prices["mom_5d"]  = prices["price"].pct_change(5).fillna(0.0) * 100
    prices["mom_10d"] = prices["price"].pct_change(10).fillna(0.0) * 100

    # Day-of-week one-hot
    prices["dow"] = prices["date_dt"].dt.dayofweek  # 0=Mon, 4=Fri
    for i, name in enumerate(["dow_mon","dow_tue","dow_wed","dow_thu","dow_fri"]):
        prices[name] = (prices["dow"] == i).astype(float)

    # Article count per date
    art_counts = pd.read_sql(
        "SELECT date(published_at) as date, COUNT(*) as article_count FROM articles GROUP BY 1",
        conn
    )
    prices = prices.merge(art_counts, on="date", how="left")
    prices["article_count"] = prices["article_count"].fillna(0.0)

    return prices.set_index("date")[["mom_5d","mom_10d",
                                     "dow_mon","dow_tue","dow_wed","dow_thu","dow_fri",
                                     "article_count"]]


NUMERIC_COLS = ["mom_5d","mom_10d","dow_mon","dow_tue","dow_wed","dow_thu","dow_fri","article_count"]


def _add_numeric_features(df: pd.DataFrame, momentum_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price momentum + day-of-week + article count into the article DataFrame."""
    if momentum_df.empty:
        for col in NUMERIC_COLS:
            df[col] = 0.0
        return df
    df["date_str"] = df["aligned_date"].dt.strftime("%Y-%m-%d")
    df = df.merge(momentum_df.reset_index().rename(columns={"date": "date_str"}),
                  on="date_str", how="left")
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    return df


# ── Data loading ──────────────────────────────────────────────────────────────

# Generic report names that appear equally across all labels — zero signal
_GENERIC_PATTERNS = [
    "crude oil marketwire", "oilgram price report", "latin american wire",
    "north american crude", "arab gulf marketscan", "european marketscan",
    "asia-pacific", "bunkerwire", "us marketscan", "lpgaswire",
    "clean tankerwire", "gas daily", "energy trader", "megawatt daily",
    "fiber marketscan", "petrochemicalscan", "weekly oil", "weekly analysis",
    "market pulse", "doe weekly", "japan weekly", "iea oil report",
    "[application/pdf]",
]

def _is_generic_headline(headline: str) -> bool:
    hl = headline.lower().strip()
    return any(p in hl for p in _GENERIC_PATTERNS) or len(hl) < 10


def load_data(min_body_chars: int = 0, label_type: str = "magnitude") -> pd.DataFrame:
    """
    Load labelled articles with text + numeric features.

    CRITICAL FILTER: Generic report names ("Crude Oil Marketwire", "Oilgram Price Report")
    appear equally across HIGH/MEDIUM/LOW — they carry zero signal. We only keep articles
    that have EITHER meaningful body text OR a specific headline.

    label_type: "magnitude" (HIGH/MEDIUM/LOW) | "direction" (rise/flat/fall)
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT la.id, la.headline, la.body_text, la.aligned_date,
               la.label, la.price_change
        FROM labelled_articles la
        ORDER BY la.aligned_date ASC, la.id ASC
    """, conn)
    momentum_df = _load_price_momentum(conn)
    conn.close()

    df["headline"] = df["headline"].fillna("")
    df["body_text"] = df["body_text"].fillna("")
    df["aligned_date"] = pd.to_datetime(df["aligned_date"], errors="coerce")
    df = df[df["aligned_date"].notna()].copy()

    if min_body_chars > 0:
        df = df[df["body_text"].str.len() >= min_body_chars].copy()

    # Body text: narrative only — with hard length cap to avoid regex hangs
    try:
        from data.cleaner import is_narrative_text, extract_narrative
        def _body_snippet(body: str) -> str:
            if not body or len(body) < 50:
                return ""
            body_capped = body[:5000]
            if not is_narrative_text(body_capped):
                return ""
            return extract_narrative(body_capped)[:BODY_TEXT_CHARS]
    except ImportError:
        def _body_snippet(body: str) -> str:
            if not body or len(body) < 50:
                return ""
            return body[:BODY_TEXT_CHARS] if sum(c.isalpha() for c in body[:500]) / 500 > 0.4 else ""

    df["body_snippet"] = df["body_text"].apply(_body_snippet)

    # Filter: keep only articles with real signal
    df["has_body"] = df["body_snippet"].str.len() > 200
    df["has_specific_headline"] = ~df["headline"].apply(_is_generic_headline)
    df_filtered = df[df["has_body"] | df["has_specific_headline"]].copy()

    print(f"[train] Total labelled: {len(df)} | After signal filter: {len(df_filtered)}")
    print(f"[train]   With body text: {df['has_body'].sum()} | Specific headlines: {df['has_specific_headline'].sum()}")
    df = df_filtered

    # Text = body snippet (primary) + headline (secondary)
    df["text"] = df.apply(
        lambda r: (r["body_snippet"].strip() + " " + r["headline"].strip()).strip()
        if r["body_snippet"] else r["headline"].strip(), axis=1,
    )
    df = df[df["text"] != ""].copy()

    # Add numeric features
    df = _add_numeric_features(df, momentum_df)

    # Derive direction labels
    if label_type == "direction":
        def _dir_label(pct):
            if pct is None or pd.isna(pct):
                return "flat"
            if float(pct) > 1.0:
                return "rise"
            if float(pct) < -1.0:
                return "fall"
            return "flat"
        df["label"] = df["price_change"].apply(_dir_label)

    df["commodity_group"] = df["headline"].apply(_assign_commodity_group)

    print(f"[train] Final training set: {len(df)} | labels: {df['label'].value_counts().to_dict()}")
    if not df.empty:
        print(f"[train] Date range: {df['aligned_date'].min().date()} -> {df['aligned_date'].max().date()}")
    return df


def _assign_commodity_group(headline: str) -> str:
    text = headline.lower()
    scores = {g: sum(1 for kw in kws if kw in text)
              for g, kws in COMMODITY_GROUPS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── TF-IDF + numeric feature matrix ──────────────────────────────────────────

def build_tfidf_features(df: pd.DataFrame,
                          tfidf: TfidfVectorizer | None = None,
                          scaler: StandardScaler | None = None,
                          fit: bool = True):
    """
    Build combined feature matrix: TF-IDF(text) + scaled numeric features.
    Returns (X_sparse, tfidf, scaler).
    """
    if fit:
        tfidf = TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), max_features=12000,
            min_df=2, sublinear_tf=True,
        )
        X_text = tfidf.fit_transform(df["text"])
    else:
        X_text = tfidf.transform(df["text"])

    X_num_raw = df[NUMERIC_COLS].values.astype(float)
    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num_raw)
    else:
        X_num = scaler.transform(X_num_raw)

    X = sp.hstack([X_text, sp.csr_matrix(X_num)])
    return X, tfidf, scaler


def build_tfidf_pipeline(max_features: int = 12000) -> Pipeline:
    """Simple TF-IDF pipeline (no numeric features) for commodity sub-models."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), max_features=max_features,
            min_df=2, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, random_state=42,
        )),
    ])


def _build_xgb_features(df: pd.DataFrame,
                         tfidf: TfidfVectorizer | None = None,
                         selector: SelectKBest | None = None,
                         scaler: StandardScaler | None = None,
                         le: LabelEncoder | None = None,
                         fit: bool = True,
                         top_k: int = 500):
    """
    Build dense feature matrix for XGBoost:
      TF-IDF → SelectKBest(top_k) → dense + numeric features
    XGBoost requires dense input; we reduce TF-IDF to top_k features first.
    Returns (X_dense, tfidf, selector, scaler, le).
    """
    if fit:
        tfidf = TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), max_features=15000,
            min_df=2, sublinear_tf=True,
        )
        X_text_sparse = tfidf.fit_transform(df["text"])
        le = LabelEncoder()
        y = le.fit_transform(df["label"])
        selector = SelectKBest(chi2, k=min(top_k, X_text_sparse.shape[1]))
        X_text_selected = selector.fit_transform(X_text_sparse, y)
    else:
        X_text_sparse = tfidf.transform(df["text"])
        X_text_selected = selector.transform(X_text_sparse)

    X_text_dense = X_text_selected.toarray()

    X_num_raw = df[NUMERIC_COLS].values.astype(float)
    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num_raw)
    else:
        X_num = scaler.transform(X_num_raw)

    X = np.hstack([X_text_dense, X_num])
    return X, tfidf, selector, scaler, le


def _xgb_available() -> bool:
    try:
        import xgboost
        return True
    except ImportError:
        return False


# ── FinBERT embeddings (no fine-tuning) ──────────────────────────────────────

_finbert_embed_cache: dict = {}  # cache model+tokenizer so we load once


def _get_finbert_embedder():
    """Load FinBERT for embedding extraction (no fine-tuning needed)."""
    if _finbert_embed_cache:
        return _finbert_embed_cache["tokenizer"], _finbert_embed_cache["model"]
    try:
        import torch
        from transformers import BertTokenizer, BertModel
        print("[train] Loading FinBERT for embeddings...")
        tok = BertTokenizer.from_pretrained("ProsusAI/finbert")
        model = BertModel.from_pretrained("ProsusAI/finbert")
        model.eval()
        _finbert_embed_cache["tokenizer"] = tok
        _finbert_embed_cache["model"] = model
        print("[train] FinBERT loaded.")
        return tok, model
    except Exception as e:
        print(f"[train] FinBERT unavailable: {e}")
        return None, None


def _embed_texts_finbert(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Extract FinBERT [CLS] embeddings for a list of texts.
    Returns array of shape (n_texts, 768).
    Uses pre-trained weights — no fine-tuning required.
    """
    import torch
    tok, model = _get_finbert_embedder()
    if tok is None:
        return np.zeros((len(texts), 768))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tok(batch, return_tensors="pt", truncation=True,
                     max_length=128, padding=True)
        with torch.no_grad():
            out = model(**inputs)
        # [CLS] token embedding = first token of last hidden state
        cls_embeddings = out.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)
        if (i // batch_size) % 5 == 0:
            print(f"  [embed] {min(i+batch_size, len(texts))}/{len(texts)} texts embedded")
    return np.vstack(all_embeddings)


def _build_finbert_xgb_features(df: pd.DataFrame,
                                  tfidf: TfidfVectorizer | None = None,
                                  selector: SelectKBest | None = None,
                                  scaler: StandardScaler | None = None,
                                  embed_scaler: StandardScaler | None = None,
                                  le: LabelEncoder | None = None,
                                  fit: bool = True,
                                  top_k: int = 300):
    """
    Build feature matrix combining:
      - FinBERT [CLS] embeddings (768 dims) — captures financial language semantics
      - TF-IDF top-k features (300 dims) — captures specific keywords
      - Numeric features (8 dims) — price momentum, day-of-week, article count

    Total: 768 + 300 + 8 = 1,076 features per article.
    XGBoost handles this well and learns interactions between all three.
    """
    # FinBERT embeddings
    print(f"  [features] Extracting FinBERT embeddings for {len(df)} texts...")
    embeddings = _embed_texts_finbert(df["text"].tolist())

    # TF-IDF features
    if fit:
        tfidf = TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), max_features=10000,
            min_df=2, sublinear_tf=True,
        )
        X_text_sparse = tfidf.fit_transform(df["text"])
        le = LabelEncoder()
        y = le.fit_transform(df["label"])
        selector = SelectKBest(chi2, k=min(top_k, X_text_sparse.shape[1]))
        X_text_selected = selector.fit_transform(X_text_sparse, y)
    else:
        X_text_sparse = tfidf.transform(df["text"])
        X_text_selected = selector.transform(X_text_sparse)

    X_text_dense = X_text_selected.toarray()

    # Numeric features
    X_num_raw = df[NUMERIC_COLS].values.astype(float)
    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num_raw)
        embed_scaler = StandardScaler()
        embeddings_scaled = embed_scaler.fit_transform(embeddings)
    else:
        X_num = scaler.transform(X_num_raw)
        embeddings_scaled = embed_scaler.transform(embeddings)

    # Combine all features
    X = np.hstack([embeddings_scaled, X_text_dense, X_num])
    print(f"  [features] Feature matrix: {X.shape} (embed={embeddings_scaled.shape[1]}, "
          f"tfidf={X_text_dense.shape[1]}, numeric={X_num.shape[1]})")
    return X, tfidf, selector, scaler, embed_scaler, le


# ── FinBERT fine-tuning (Change 1) ────────────────────────────────────────────

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_DIR = SAVE_DIR / "classifier_finbert"


def _finbert_available() -> bool:
    try:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        return True
    except ImportError:
        return False


def train_finbert(df: pd.DataFrame, label_type: str = "magnitude") -> dict:
    """
    Fine-tune ProsusAI/finbert on the labelled articles.
    Uses the same walk-forward CV structure for evaluation, then trains
    a final model on all data.

    Returns cv_metrics dict. Saves model to models/saved/classifier_finbert/.
    """
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import Dataset, DataLoader
        from transformers import (BertTokenizer, BertForSequenceClassification,
                                   get_linear_schedule_with_warmup)
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("[train:finbert] transformers/torch not installed. "
              "Run: pip install transformers torch")
        return {}

    labels_sorted = sorted(df["label"].dropna().unique().tolist())
    le = LabelEncoder()
    le.fit(labels_sorted)

    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train:finbert] Device: {device} | Labels: {labels_sorted}")

    class ArticleDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.encodings = tokenizer(
                list(texts), truncation=True, padding=True,
                max_length=max_len, return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]

    # CPU: smaller batch, fewer epochs to keep training time reasonable
    BATCH_SIZE = 8 if not torch.cuda.is_available() else 16
    EPOCHS_CV  = 1 if not torch.cuda.is_available() else 2   # 1 epoch on CPU for CV
    EPOCHS_FINAL = 2 if not torch.cuda.is_available() else 3  # 2 epochs for final model

    def _train_fold(train_df, test_df, epochs=EPOCHS_CV, batch_size=BATCH_SIZE, lr=2e-5):
        model = BertForSequenceClassification.from_pretrained(
            FINBERT_MODEL, num_labels=len(labels_sorted),
            ignore_mismatched_sizes=True
        ).to(device)

        train_ds = ArticleDataset(train_df["text"], le.transform(train_df["label"]), tokenizer)
        test_ds  = ArticleDataset(test_df["text"],  le.transform(test_df["label"]),  tokenizer)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size)

        optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = len(train_dl) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for step, (batch, labels_b) in enumerate(train_dl):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels_b = labels_b.to(device)
                outputs = model(**batch, labels=labels_b)
                outputs.loss.backward()
                total_loss += outputs.loss.item()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                if (step + 1) % 20 == 0:
                    print(f"    epoch {epoch+1}/{epochs} step {step+1}/{len(train_dl)} loss={total_loss/(step+1):.4f}")

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch, labels_b in test_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels_b.numpy())

        acc = accuracy_score(all_true, all_preds)
        report = classification_report(
            all_true, all_preds,
            labels=list(range(len(labels_sorted))),
            target_names=labels_sorted,
            zero_division=0, output_dict=True
        )
        return acc, report, model

    # Walk-forward CV
    df_sorted = df.sort_values("aligned_date").reset_index(drop=True)
    unique_dates = sorted(df_sorted["aligned_date"].dt.normalize().unique())
    segment_size = max(1, len(unique_dates) // (N_CV_FOLDS + 1))
    fold_results = []

    for fold in range(N_CV_FOLDS):
        train_cutoff = unique_dates[min((fold+1)*segment_size, len(unique_dates)-1)]
        test_end     = unique_dates[min((fold+2)*segment_size, len(unique_dates)-1)]
        train_df = df_sorted[df_sorted["aligned_date"] <= train_cutoff]
        test_df  = df_sorted[(df_sorted["aligned_date"] > train_cutoff) &
                              (df_sorted["aligned_date"] <= test_end)]
        if len(train_df) < MIN_FOLD_TRAIN or len(test_df) < MIN_FOLD_TEST:
            continue
        if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
            continue

        print(f"  [cv:finbert] Fold {fold+1}: train={len(train_df)} test={len(test_df)}")
        acc, report, _ = _train_fold(train_df, test_df)
        fold_results.append({
            "fold": fold+1, "accuracy": acc,
            "high_precision": report.get("HIGH", {}).get("precision", 0),
            "high_recall":    report.get("HIGH", {}).get("recall", 0),
        })
        print(f"  [cv:finbert] Fold {fold+1}: acc={acc*100:.1f}%")

    if not fold_results:
        print("[train:finbert] No valid CV folds.")
        return {}

    avg_acc = float(np.mean([r["accuracy"] for r in fold_results]))
    avg_hp  = float(np.mean([r["high_precision"] for r in fold_results]))
    avg_hr  = float(np.mean([r["high_recall"] for r in fold_results]))
    print(f"  [cv:finbert] CV avg: acc={avg_acc*100:.1f}% HIGH_prec={avg_hp*100:.0f}%")

    # Train final model on all data
    print("[train:finbert] Training final model on all data...")
    _, _, final_model = _train_fold(df_sorted, df_sorted.tail(max(1, len(df_sorted)//10)),
                                    epochs=EPOCHS_FINAL)

    # Save
    FINBERT_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(str(FINBERT_DIR))
    tokenizer.save_pretrained(str(FINBERT_DIR))
    joblib.dump({"label_encoder": le, "labels": labels_sorted,
                 "label_type": label_type},
                FINBERT_DIR / "meta.pkl")
    print(f"[train:finbert] Saved to {FINBERT_DIR}")

    return {
        "cv_accuracy": avg_acc, "cv_high_precision": avg_hp,
        "cv_high_recall": avg_hr, "cv_folds": fold_results,
        "n_folds": len(fold_results),
    }


# ── Walk-forward CV (TF-IDF) ──────────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, n_folds: int = N_CV_FOLDS,
                    name: str = "general", use_numeric: bool = True,
                    use_xgb: bool = False, use_finbert_xgb: bool = False) -> dict:
    df = df.sort_values("aligned_date").reset_index(drop=True)
    unique_dates = sorted(df["aligned_date"].dt.normalize().unique())

    if len(unique_dates) < n_folds + 1:
        return _single_split_eval(df, name, use_numeric, use_xgb, use_finbert_xgb)

    segment_size = len(unique_dates) // (n_folds + 1)
    fold_results = []

    for fold in range(n_folds):
        train_cutoff = unique_dates[min((fold+1)*segment_size, len(unique_dates)-1)]
        test_end     = unique_dates[min((fold+2)*segment_size, len(unique_dates)-1)]
        train_df = df[df["aligned_date"] <= train_cutoff].copy()
        test_df  = df[(df["aligned_date"] > train_cutoff) &
                      (df["aligned_date"] <= test_end)].copy()

        if (len(train_df) < MIN_FOLD_TRAIN or len(test_df) < MIN_FOLD_TEST
                or train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2):
            continue

        if use_finbert_xgb and _xgb_available():
            from xgboost import XGBClassifier
            X_train, tfidf, selector, scaler, embed_scaler, le = _build_finbert_xgb_features(
                train_df, fit=True)
            X_test, _, _, _, _, _ = _build_finbert_xgb_features(
                test_df, tfidf=tfidf, selector=selector, scaler=scaler,
                embed_scaler=embed_scaler, le=le, fit=False)
            y_train = le.transform(train_df["label"])
            y_test  = le.transform(test_df["label"])
            clf = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            pred_enc = clf.predict(X_test)
            pred = le.inverse_transform(pred_enc)
            test_labels = le.inverse_transform(y_test)
        elif use_xgb and _xgb_available():
            from xgboost import XGBClassifier
            X_train, tfidf, selector, scaler, le = _build_xgb_features(train_df, fit=True)
            X_test, _, _, _, _ = _build_xgb_features(test_df, tfidf=tfidf, selector=selector,
                                                       scaler=scaler, le=le, fit=False)
            y_train = le.transform(train_df["label"])
            y_test  = le.transform(test_df["label"])
            clf = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            pred_enc = clf.predict(X_test)
            pred = le.inverse_transform(pred_enc)
            test_labels = le.inverse_transform(y_test)
        elif use_numeric:
            X_train, tfidf, scaler = build_tfidf_features(train_df, fit=True)
            X_test, _, _ = build_tfidf_features(test_df, tfidf=tfidf, scaler=scaler, fit=False)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     C=1.0, random_state=42)
            clf.fit(X_train, train_df["label"])
            pred = clf.predict(X_test)
            test_labels = test_df["label"]
        else:
            model = build_tfidf_pipeline()
            model.fit(train_df["text"], train_df["label"])
            pred = model.predict(test_df["text"])
            test_labels = test_df["label"]

        acc = accuracy_score(test_labels, pred)
        labels = sorted(df["label"].dropna().unique().tolist())
        report = classification_report(test_labels, pred, labels=labels,
                                       zero_division=0, output_dict=True)
        fold_results.append({
            "fold": fold+1, "train_size": len(train_df), "test_size": len(test_df),
            "accuracy": acc,
            "high_precision": report.get("HIGH", report.get("rise", {})).get("precision", 0),
            "high_recall":    report.get("HIGH", report.get("rise", {})).get("recall", 0),
            "train_cutoff": str(train_cutoff)[:10], "test_end": str(test_end)[:10],
        })
        print(f"  [cv:{name}] Fold {fold+1}: train={len(train_df)} ({str(train_cutoff)[:10]}) "
              f"test={len(test_df)} ({str(test_end)[:10]}) acc={acc*100:.1f}%"
              f"{' [XGB]' if use_xgb else ''}{' [FinBERT+XGB]' if use_finbert_xgb else ''}")

    if not fold_results:
        return _single_split_eval(df, name, use_numeric, use_xgb, use_finbert_xgb)

    avg_acc = float(np.mean([r["accuracy"] for r in fold_results]))
    avg_hp  = float(np.mean([r["high_precision"] for r in fold_results]))
    avg_hr  = float(np.mean([r["high_recall"] for r in fold_results]))
    print(f"  [cv:{name}] CV avg: acc={avg_acc*100:.1f}% prec={avg_hp*100:.0f}% recall={avg_hr*100:.0f}%")
    return {"cv_accuracy": avg_acc, "cv_high_precision": avg_hp,
            "cv_high_recall": avg_hr, "cv_folds": fold_results, "n_folds": len(fold_results)}


def _single_split_eval(df: pd.DataFrame, name: str,
                        use_numeric: bool = True, use_xgb: bool = False,
                        use_finbert_xgb: bool = False) -> dict:
    df = df.sort_values("aligned_date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    if len(train_df) < MIN_FOLD_TRAIN or len(test_df) < MIN_FOLD_TEST:
        return {"cv_accuracy": 0.0, "cv_high_precision": 0.0,
                "cv_high_recall": 0.0, "cv_folds": [], "n_folds": 0}
    if use_finbert_xgb and _xgb_available():
        from xgboost import XGBClassifier
        X_train, tfidf, selector, scaler, embed_scaler, le = _build_finbert_xgb_features(
            train_df, fit=True)
        X_test, _, _, _, _, _ = _build_finbert_xgb_features(
            test_df, tfidf=tfidf, selector=selector, scaler=scaler,
            embed_scaler=embed_scaler, le=le, fit=False)
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                            use_label_encoder=False, eval_metric="mlogloss",
                            random_state=42, n_jobs=-1)
        clf.fit(X_train, le.transform(train_df["label"]))
        pred = le.inverse_transform(clf.predict(X_test))
        test_labels = test_df["label"]
    elif use_xgb and _xgb_available():
        from xgboost import XGBClassifier
        X_train, tfidf, selector, scaler, le = _build_xgb_features(train_df, fit=True)
        X_test, _, _, _, _ = _build_xgb_features(test_df, tfidf=tfidf, selector=selector,
                                                   scaler=scaler, le=le, fit=False)
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="mlogloss",
                            random_state=42, n_jobs=-1)
        clf.fit(X_train, le.transform(train_df["label"]))
        pred = le.inverse_transform(clf.predict(X_test))
        test_labels = test_df["label"]
    elif use_numeric:
        X_train, tfidf, scaler = build_tfidf_features(train_df, fit=True)
        X_test, _, _ = build_tfidf_features(test_df, tfidf=tfidf, scaler=scaler, fit=False)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(X_train, train_df["label"])
        pred = clf.predict(X_test)
        test_labels = test_df["label"]
    else:
        model = build_tfidf_pipeline()
        model.fit(train_df["text"], train_df["label"])
        pred = model.predict(test_df["text"])
        test_labels = test_df["label"]
    acc = accuracy_score(test_labels, pred)
    labels = sorted(df["label"].dropna().unique().tolist())
    report = classification_report(test_labels, pred, labels=labels,
                                   zero_division=0, output_dict=True)
    print(f"  [cv:{name}] Single split: acc={acc*100:.1f}%")
    return {"cv_accuracy": acc,
            "cv_high_precision": report.get("HIGH", report.get("rise", {})).get("precision", 0),
            "cv_high_recall":    report.get("HIGH", report.get("rise", {})).get("recall", 0),
            "cv_folds": [], "n_folds": 1}


# ── Train final models ────────────────────────────────────────────────────────

def train_final_tfidf(df: pd.DataFrame, use_xgb: bool = False,
                       use_finbert_xgb: bool = False) -> tuple:
    """Train on ALL data. Returns appropriate tuple based on model type."""
    if use_finbert_xgb and _xgb_available():
        from xgboost import XGBClassifier
        X, tfidf, selector, scaler, embed_scaler, le = _build_finbert_xgb_features(df, fit=True)
        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X, le.transform(df["label"]))
        return clf, tfidf, selector, scaler, embed_scaler, le
    if use_xgb and _xgb_available():
        from xgboost import XGBClassifier
        X, tfidf, selector, scaler, le = _build_xgb_features(df, fit=True)
        clf = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X, le.transform(df["label"]))
        return clf, tfidf, selector, scaler, le
    X, tfidf, scaler = build_tfidf_features(df, fit=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             C=1.0, random_state=42)
    clf.fit(X, df["label"])
    return clf, tfidf, scaler


def train_final_pipeline(df: pd.DataFrame, max_features: int = 12000) -> Pipeline:
    """Train simple Pipeline (for commodity sub-models)."""
    model = build_tfidf_pipeline(max_features)
    model.fit(df["text"], df["label"])
    return model


# ── Save artifacts ────────────────────────────────────────────────────────────

def _make_info(cv_metrics: dict, df: pd.DataFrame, model_type: str,
               label_type: str, extra: dict | None = None) -> dict:
    labels = sorted(df["label"].dropna().unique().tolist())
    top_label = "HIGH" if label_type == "magnitude" else "rise"
    # test_samples = total samples evaluated across all CV folds
    test_samples = sum(f.get("test_size", 0) for f in cv_metrics.get("cv_folds", []))
    info = {
        "model_type":            model_type,
        "label_type":            label_type,
        "labels":                labels,
        "train_samples":         len(df),
        "test_samples":          test_samples,
        "n_folds":               cv_metrics.get("n_folds", 0),
        "split_mode":            f"walk-forward CV ({cv_metrics.get('n_folds',0)} folds)",
        "classification_report": {
            "accuracy": cv_metrics["cv_accuracy"],
            top_label: {"precision": cv_metrics["cv_high_precision"],
                        "recall":    cv_metrics["cv_high_recall"]},
        },
        "accuracy":              cv_metrics["cv_accuracy"],
        "trained_at":            datetime.now().strftime("%Y-%m-%d %H:%M"),
        "features":              "TF-IDF(headline+body) + price_momentum + day_of_week + article_count",
        "cv_folds":              cv_metrics.get("cv_folds", []),
    }
    if extra:
        info.update(extra)
    return info


def save_general(clf, tfidf, scaler_or_selector, df: pd.DataFrame, cv_metrics: dict,
                 per_commodity: dict, label_type: str,
                 direction_cv: dict | None = None,
                 finbert_cv: dict | None = None,
                 use_xgb: bool = False,
                 xgb_extra: dict | None = None,
                 use_finbert_xgb: bool = False) -> None:
    suffix = "" if label_type == "magnitude" else "_direction"
    fname = f"classifier{suffix}.pkl"

    if use_finbert_xgb and xgb_extra:
        model_type_str = "finbert_xgb"
        bundle = {"clf": clf, "tfidf": tfidf,
                  "selector": xgb_extra.get("selector"),
                  "scaler": xgb_extra.get("scaler"),
                  "embed_scaler": xgb_extra.get("embed_scaler"),
                  "le": xgb_extra.get("le"),
                  "model_type": model_type_str,
                  "numeric_cols": NUMERIC_COLS,
                  "label_type": label_type}
    elif use_xgb and xgb_extra:
        model_type_str = "tfidf_xgb"
        bundle = {"clf": clf, "tfidf": tfidf,
                  "selector": xgb_extra.get("selector"),
                  "scaler": xgb_extra.get("scaler"),
                  "le": xgb_extra.get("le"),
                  "model_type": model_type_str,
                  "numeric_cols": NUMERIC_COLS,
                  "label_type": label_type}
    else:
        model_type_str = "tfidf_numeric_logreg"
        bundle = {"clf": clf, "tfidf": tfidf, "scaler": scaler_or_selector,
                  "model_type": model_type_str,
                  "numeric_cols": NUMERIC_COLS,
                  "label_type": label_type}

    joblib.dump(bundle, SAVE_DIR / fname)
    info = _make_info(cv_metrics, df, model_type_str, label_type, {
        "per_commodity_metrics": per_commodity,
        "commodity_models":      list(per_commodity.keys()),
        "direction_model_cv":    direction_cv or {},
        "finbert_cv":            finbert_cv or {},
    })
    joblib.dump(info, SAVE_DIR / "model_info.pkl")
    print(f"\n[train] Saved {fname} -> {SAVE_DIR}")


def save_commodity_model(model: Pipeline, df: pd.DataFrame,
                         cv_metrics: dict, commodity: str, label_type: str) -> None:
    joblib.dump(model, SAVE_DIR / f"classifier_{commodity}.pkl")
    info = _make_info(cv_metrics, df, f"tfidf_logreg_{commodity}", label_type)
    info["commodity"] = commodity
    joblib.dump(info, SAVE_DIR / f"model_info_{commodity}.pkl")
    print(f"[train] Saved classifier_{commodity}.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(min_body_chars: int = 0, commodity_filter: str = "",
         no_commodity_models: bool = False,
         model_type: str = "tfidf",
         label_type: str = "magnitude") -> None:

    use_xgb = (model_type == "xgb")
    use_finbert_xgb = (model_type == "finbert-xgb")
    print(f"\n[train] model_type={model_type} | label_type={label_type}"
          f"{' [XGBoost]' if use_xgb else ''}"
          f"{' [FinBERT+XGBoost]' if use_finbert_xgb else ''}")

    # ── FinBERT path (Change 1) ───────────────────────────────────────────────
    if model_type == "finbert":
        if not _finbert_available():
            print("[train] ERROR: transformers/torch not installed.")
            print("  Run: pip install transformers torch")
            return
        df = load_data(min_body_chars=min_body_chars, label_type=label_type)
        if len(df) < MIN_SAMPLES:
            print(f"[train] ERROR: Only {len(df)} samples.")
            return
        cv = train_finbert(df, label_type=label_type)
        if cv:
            # Update model_info.pkl with FinBERT metrics
            info_path = SAVE_DIR / "model_info.pkl"
            if info_path.exists():
                info = joblib.load(info_path)
            else:
                info = {}
            info["finbert_cv"] = cv
            info["finbert_trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            joblib.dump(info, info_path)
            print(f"\n[train] FinBERT CV accuracy: {cv['cv_accuracy']*100:.1f}%")
        return

    # ── TF-IDF path ───────────────────────────────────────────────────────────
    df = load_data(min_body_chars=min_body_chars, label_type=label_type)
    if len(df) < MIN_SAMPLES:
        print(f"[train] ERROR: Only {len(df)} samples. Need {MIN_SAMPLES}+.")
        return

    print("\n" + "="*60)
    print(f"STEP 1 — Walk-forward CV (general, label_type={label_type})")
    print("="*60)
    cv_metrics = walk_forward_cv(df, n_folds=N_CV_FOLDS, name="general",
                                  use_numeric=True, use_xgb=use_xgb,
                                  use_finbert_xgb=use_finbert_xgb)

    print("\n[train] Training final model on all data...")
    final_result = train_final_tfidf(df, use_xgb=use_xgb, use_finbert_xgb=use_finbert_xgb)
    if use_finbert_xgb:
        clf, tfidf, selector, scaler, embed_scaler, le = final_result
        xgb_extra = {"selector": selector, "scaler": scaler,
                     "embed_scaler": embed_scaler, "le": le}
    elif use_xgb:
        clf, tfidf, selector, scaler, le = final_result
        xgb_extra = {"selector": selector, "scaler": scaler, "le": le}
    else:
        clf, tfidf, scaler = final_result
        xgb_extra = None

    # ── Directional model — train alongside magnitude ─────────────────────────
    direction_cv = {}
    if label_type == "magnitude":
        print("\n" + "="*60)
        print("STEP 1b — Directional model (rise/flat/fall)")
        print("="*60)
        df_dir = load_data(min_body_chars=min_body_chars, label_type="direction")
        if len(df_dir) >= MIN_SAMPLES:
            direction_cv = walk_forward_cv(df_dir, n_folds=N_CV_FOLDS,
                                           name="direction", use_numeric=True,
                                           use_xgb=use_xgb,
                                           use_finbert_xgb=use_finbert_xgb)
            dir_result = train_final_tfidf(df_dir, use_xgb=use_xgb,
                                           use_finbert_xgb=use_finbert_xgb)
            if use_finbert_xgb:
                dir_clf, dir_tfidf, dir_selector, dir_scaler, dir_embed_scaler, dir_le = dir_result
                dir_bundle = {"clf": dir_clf, "tfidf": dir_tfidf,
                              "selector": dir_selector, "scaler": dir_scaler,
                              "embed_scaler": dir_embed_scaler, "le": dir_le,
                              "model_type": "finbert_xgb_direction",
                              "numeric_cols": NUMERIC_COLS, "label_type": "direction"}
            elif use_xgb:
                dir_clf, dir_tfidf, dir_selector, dir_scaler, dir_le = dir_result
                dir_bundle = {"clf": dir_clf, "tfidf": dir_tfidf,
                              "selector": dir_selector, "scaler": dir_scaler,
                              "le": dir_le, "model_type": "tfidf_xgb_direction",
                              "numeric_cols": NUMERIC_COLS, "label_type": "direction"}
            else:
                dir_clf, dir_tfidf, dir_scaler = dir_result
                dir_bundle = {"clf": dir_clf, "tfidf": dir_tfidf, "scaler": dir_scaler,
                              "model_type": "tfidf_numeric_logreg_direction",
                              "numeric_cols": NUMERIC_COLS, "label_type": "direction"}
            joblib.dump(dir_bundle, SAVE_DIR / "classifier_direction.pkl")
            print(f"[train] Saved classifier_direction.pkl | CV acc={direction_cv['cv_accuracy']*100:.1f}%")

    # ── Commodity-specific models ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Commodity-specific models")
    print("="*60)
    per_commodity_metrics: dict[str, dict] = {}

    if not no_commodity_models:
        commodity_names = [commodity_filter] if commodity_filter else list(COMMODITY_GROUPS.keys())
        for commodity in commodity_names:
            df_comm = df[df["commodity_group"] == commodity].copy()
            df_gen  = df[df["commodity_group"] == "general"].copy()
            df_combined = pd.concat([df_comm, df_gen], ignore_index=True).drop_duplicates("id")
            print(f"\n  [{commodity}] {len(df_comm)} specific + {len(df_gen)} general = {len(df_combined)} total")
            if len(df_combined) < MIN_SAMPLES:
                print(f"  [{commodity}] Skipping — not enough samples")
                continue
            comm_cv = walk_forward_cv(df_combined, n_folds=N_CV_FOLDS,
                                      name=commodity, use_numeric=False)
            comm_model = train_final_pipeline(df_combined, max_features=8000)
            save_commodity_model(comm_model, df_combined, comm_cv, commodity, label_type)
            per_commodity_metrics[commodity] = {
                "accuracy":       comm_cv["cv_accuracy"],
                "train_samples":  len(df_combined),
                "high_precision": comm_cv["cv_high_precision"],
                "high_recall":    comm_cv["cv_high_recall"],
            }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_general(clf, tfidf, scaler if not (use_xgb or use_finbert_xgb) else None,
                 df, cv_metrics, per_commodity_metrics, label_type,
                 direction_cv=direction_cv, use_xgb=use_xgb,
                 xgb_extra=xgb_extra, use_finbert_xgb=use_finbert_xgb)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"General model (walk-forward CV avg):")
    print(f"  Accuracy:       {cv_metrics['cv_accuracy']*100:.1f}%")
    print(f"  Precision:      {cv_metrics['cv_high_precision']*100:.1f}%")
    print(f"  Recall:         {cv_metrics['cv_high_recall']*100:.1f}%")
    print(f"  Folds:          {cv_metrics['n_folds']}")
    print(f"  Final model on: {len(df)} articles (all data)")
    if direction_cv:
        print(f"Directional model CV acc: {direction_cv.get('cv_accuracy',0)*100:.1f}%")
    if per_commodity_metrics:
        print("\nCommodity models:")
        for comm, m in per_commodity_metrics.items():
            print(f"  {comm:10s}: acc={m['accuracy']*100:.1f}%  "
                  f"prec={m['high_precision']*100:.0f}%  "
                  f"({m['train_samples']} samples)")
    print("\n[train] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-body-chars", type=int, default=0)
    parser.add_argument("--commodity", type=str, default="",
                        choices=["dubai", "brent", "wti", "lng", ""])
    parser.add_argument("--no-commodity-models", action="store_true")
    parser.add_argument("--model-type", type=str, default="tfidf",
                        choices=["tfidf", "xgb", "finbert-xgb", "finbert"],
                        help="tfidf | xgb (XGBoost) | finbert-xgb (FinBERT embeddings + XGBoost, best accuracy) | finbert (fine-tune, needs GPU)")
    parser.add_argument("--label-type", type=str, default="magnitude",
                        choices=["magnitude", "direction"],
                        help="magnitude (HIGH/MEDIUM/LOW) | direction (rise/flat/fall)")
    args = parser.parse_args()
    main(min_body_chars=args.min_body_chars,
         commodity_filter=args.commodity,
         no_commodity_models=args.no_commodity_models,
         model_type=args.model_type,
         label_type=args.label_type)





