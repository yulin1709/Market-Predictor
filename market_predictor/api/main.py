"""
api/main.py — FastAPI REST API for the AI Market Impact Predictor.

Endpoints:
  GET /predict?headline=...  → run full prediction pipeline
  GET /feed                  → latest 50 labelled articles
  GET /health                → {"status": "ok"}

Run with:
    uvicorn api.main:app --reload
    (from market_predictor/ directory)
"""
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")

app = FastAPI(
    title="AI Market Impact Predictor",
    description="Petronas Trading Digital — Predicts energy news market impact",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


class SimilarEvent(BaseModel):
    headline: str
    date: str
    label: str
    price_change: float
    similarity: float


class PredictResponse(BaseModel):
    headline: str
    label: str
    score: float
    high_prob: float
    medium_prob: float
    low_prob: float
    explanation: str
    entities: dict
    similar_events: list[SimilarEvent]
    latency_ms: float
    timestamp: str


class ArticleResponse(BaseModel):
    id: str
    headline: str
    body_text: str | None = None
    source: str | None = None
    published_at: str | None = None
    url: str | None = None
    content_id: str | None = None
    content_type: str | None = None
    report_name: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/predict", response_model=PredictResponse)
def predict_endpoint(headline: str = Query(..., min_length=5)):
    if not headline.strip():
        raise HTTPException(status_code=400, detail="headline cannot be empty")
    try:
        from inference.pipeline import predict
        result = predict(headline)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictResponse(
        **{k: result[k] for k in [
            "headline", "label", "score", "high_prob", "medium_prob",
            "low_prob", "explanation", "entities", "latency_ms"
        ]},
        similar_events=[SimilarEvent(**s) for s in result["similar_events"]],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/feed")
def feed(limit: int = 50):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT headline, source, published_at, aligned_date, label, price_change, url
            FROM labelled_articles
            ORDER BY published_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        {
            "headline": r[0], "source": r[1], "published_at": r[2],
            "aligned_date": r[3], "label": r[4],
            "price_change": r[5], "url": r[6],
        }
        for r in rows
    ]


@app.get("/reports", response_model=list[ArticleResponse])
def reports(
    report_name: Optional[str] = None,
    content_id: Optional[str] = None,
    limit: int = 50,
):
    try:
        conn = sqlite3.connect(DB_PATH)
        columns = table_columns(conn, "articles")
        select_content_id = "content_id" if "content_id" in columns else "NULL AS content_id"
        select_content_type = "content_type" if "content_type" in columns else "NULL AS content_type"
        select_report_name = "report_name" if "report_name" in columns else "NULL AS report_name"

        query = f"""
            SELECT id, headline, body_text, source, published_at, url,
                   {select_content_id}, {select_content_type}, {select_report_name}
            FROM articles
        """
        clauses = []
        params: list[object] = []

        if report_name and "report_name" in columns:
            clauses.append("LOWER(report_name) = LOWER(?)")
            params.append(report_name)
        if content_id and "content_id" in columns:
            clauses.append("content_id = ?")
            params.append(content_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        ArticleResponse(
            id=row[0],
            headline=row[1],
            body_text=row[2],
            source=row[3],
            published_at=row[4],
            url=row[5],
            content_id=row[6],
            content_type=row[7],
            report_name=row[8],
        )
        for row in rows
    ]
