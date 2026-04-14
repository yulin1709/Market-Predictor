"""
inference/vector_store.py — ChromaDB vector store for historical article similarity search.

Embedding model: sentence-transformers all-MiniLM-L6-v2
Persistent store: ./chroma_db (relative to market_predictor root)

Usage:
    python inference/vector_store.py          # index all labelled articles
    python inference/vector_store.py --query "OPEC cuts production"
"""
import argparse
import os
import sqlite3
import ssl
import sys

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.db_path import DB_PATH as _DB_PATH; DB_PATH = _DB_PATH
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "energy_articles"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── SSL fix for corporate networks ────────────────────────────────────────────
# Disable SSL verification for HuggingFace downloads (corporate proxy issue).
# The model is cached locally after first download — this only affects the
# version-check request that happens on each load.
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass


def _load_embed_model() -> SentenceTransformer:
    """Load sentence-transformer model, using local cache if available."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Try local cache first (no network)
            return SentenceTransformer(EMBED_MODEL, local_files_only=True)
        except Exception:
            # Fall back to download with SSL verification disabled
            return SentenceTransformer(EMBED_MODEL)


class _LocalEmbeddingFunction:
    """Wraps a local SentenceTransformer for ChromaDB."""
    def __init__(self):
        self._model = None

    def name(self) -> str:
        return "local_sentence_transformer"

    def _get_model(self):
        if self._model is None:
            self._model = _load_embed_model()
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._get_model().encode(input, show_progress_bar=False).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_model().encode([text], show_progress_bar=False)[0].tolist()


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = _LocalEmbeddingFunction()
    # Delete and recreate if embedding function name conflicts with persisted config
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    except ValueError as e:
        if "conflict" in str(e).lower() or "embedding function" in str(e).lower():
            # Old collection used a different embedding function name — recreate it
            client.delete_collection(COLLECTION_NAME)
            return client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        raise


def index_articles(df=None) -> None:
    """Index all labelled articles into ChromaDB."""
    if df is None:
        conn = sqlite3.connect(DB_PATH)
        import pandas as pd
        df = pd.read_sql("""
            SELECT la.id, la.headline, la.aligned_date, la.label, la.price_change,
                   e.event_type, e.region
            FROM labelled_articles la
            LEFT JOIN entities e ON la.id = e.article_id
        """, conn)
        conn.close()

    if df.empty:
        print("[vector_store] No articles to index.")
        return

    collection = get_collection()
    print(f"[vector_store] Indexing {len(df)} articles...")

    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        ids = batch["id"].tolist()
        docs = batch["headline"].fillna("").tolist()
        metas = []
        for _, row in batch.iterrows():
            metas.append({
                "headline": str(row["headline"] or "")[:300],
                "date": str(row.get("aligned_date", "") or ""),
                "label": str(row.get("label", "") or ""),
                "price_change": float(row.get("price_change", 0) or 0),
                "event_type": str(row.get("event_type", "") or ""),
                "region": str(row.get("region", "") or ""),
            })
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"  Indexed batch {i // batch_size + 1} ({len(batch)} articles)")

    print(f"[vector_store] Done. Total indexed: {collection.count()}")


def find_similar(text: str, n: int = 3) -> list[dict]:
    """Return n most similar historical articles to text."""
    # Skip generic report names — they produce meaningless similarity results
    if not text or len(text.strip()) < 15:
        return []

    collection = get_collection()
    if collection.count() == 0:
        return []

    # Embed the query ourselves — avoids ChromaDB calling embed_query with kwargs
    ef = _LocalEmbeddingFunction()
    query_embedding = ef.embed_query(text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, collection.count()),
        include=["metadatas", "distances"],
    )

    similar = []
    for j in range(len(results["ids"][0])):
        meta = results["metadatas"][0][j]
        similar.append({
            "headline": meta.get("headline", ""),
            "date": meta.get("date", ""),
            "label": meta.get("label", ""),
            "price_change": meta.get("price_change", 0),
            "event_type": meta.get("event_type", ""),
            "region": meta.get("region", ""),
            "similarity": round(1 - results["distances"][0][j], 3),
        })
    return similar


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if args.query:
        results = find_similar(args.query)
        print(f"\nTop {len(results)} similar articles for: '{args.query}'\n")
        for r in results:
            print(f"  [{r['label']}] {r['headline'][:80]}")
            print(f"    Date: {r['date']}  Move: {r['price_change']:+.2f}%  Sim: {r['similarity']}")
    else:
        index_articles()
