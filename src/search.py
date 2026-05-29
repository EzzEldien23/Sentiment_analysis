"""BM25 retrieval with optional sentiment filtering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_amazon
from src.preprocessing import TextPreprocessor
from src.vectorization import BM25Vectorizer


@dataclass
class SearchResult:
    doc_id: str
    score: float
    sentiment: str
    text: str


class SentimentSearchEngine:
    def __init__(
        self,
        preprocessor: TextPreprocessor,
        vectorizer: BM25Vectorizer,
        matrix,
        documents: pd.DataFrame,
    ) -> None:
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.documents = documents.reset_index(drop=True)

    def search(self, query: str, sentiment: str | None = None, top_k: int = 10) -> list[SearchResult]:
        clean_query = self.preprocessor.transform([query])
        query_vector = self.vectorizer.transform(clean_query)
        scores = (self.matrix @ query_vector.T).toarray().ravel()

        mask = np.ones(len(scores), dtype=bool)
        if sentiment in {"positive", "negative"}:
            wanted = 1 if sentiment == "positive" else 0
            mask = self.documents["label"].to_numpy() == wanted
        scores = np.where(mask, scores, -np.inf)
        limit = min(top_k, len(scores))
        candidate_idx = np.argpartition(scores, -limit)[-limit:]
        ranked_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

        results = []
        for idx in ranked_idx:
            if not np.isfinite(scores[idx]) or scores[idx] <= 0:
                continue
            row = self.documents.iloc[int(idx)]
            results.append(
                SearchResult(
                    doc_id=str(row["doc_id"]),
                    score=float(scores[idx]),
                    sentiment="positive" if int(row["label"]) == 1 else "negative",
                    text=str(row["text"])[:500],
                )
            )
        return results


def build_index(args: argparse.Namespace) -> Path:
    df = load_amazon(args.input, sample=args.sample, random_state=args.random_state)
    preprocessor = TextPreprocessor(mode="standard", reduction=args.reduction)
    clean_text = preprocessor.transform(df["text"])
    vectorizer = BM25Vectorizer(max_features=args.max_features, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(clean_text)
    payload = {
        "preprocessor": preprocessor,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": df[["doc_id", "text", "label"]],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output)
    return output


def load_engine(path: str | Path) -> SentimentSearchEngine:
    payload = joblib.load(path)
    if isinstance(payload, SentimentSearchEngine):
        return payload
    return SentimentSearchEngine(
        payload["preprocessor"],
        payload["vectorizer"],
        payload["matrix"],
        payload["documents"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/Reviews.csv")
    parser.add_argument("--output", default="models/amazon_bm25_search.joblib")
    parser.add_argument("--sample", type=int, default=500_000)
    parser.add_argument("--max-features", type=int, default=150_000)
    parser.add_argument("--reduction", choices=["none", "stem", "lemma"], default="lemma")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(build_index(parse_args()))
