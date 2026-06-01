"""FastAPI app serving sentiment classification and BM25 search."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover - clear error for missing runtime deps.
    raise RuntimeError("Install API dependencies with `uv sync` or `pip install -r requirements.txt`.") from exc

from src.search import SentimentSearchEngine, load_engine


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/reductions/amazon_bow_stem.joblib"))
SEARCH_INDEX_PATH = Path(os.getenv("SEARCH_INDEX_PATH", "models/amazon_bm25_search.joblib"))

app = FastAPI(title="Sentiment Analysis Lab API", version="0.1.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    label: int
    sentiment: str
    probability: float | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    sentiment: str | None = Field(default=None, pattern="^(positive|negative)$")
    top_k: int = Field(default=10, ge=1, le=50)


def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model not found: {MODEL_PATH}")
    if not hasattr(app.state, "model"):
        app.state.model = joblib.load(MODEL_PATH)
    return app.state.model


def get_search_engine() -> SentimentSearchEngine:
    if not SEARCH_INDEX_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Search index not found: {SEARCH_INDEX_PATH}")
    if not hasattr(app.state, "search_engine"):
        app.state.search_engine = load_engine(SEARCH_INDEX_PATH)
    return app.state.search_engine


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "search_available": SEARCH_INDEX_PATH.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model = get_model()
    label = int(model.predict([payload.text])[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([payload.text])[0][label])
    return PredictResponse(
        label=label,
        sentiment="positive" if label == 1 else "negative",
        probability=probability,
    )


@app.post("/search")
def search(payload: SearchRequest) -> dict[str, object]:
    engine = get_search_engine()
    results = engine.search(payload.query, sentiment=payload.sentiment, top_k=payload.top_k)
    return {"query": payload.query, "count": len(results), "results": [r.__dict__ for r in results]}
