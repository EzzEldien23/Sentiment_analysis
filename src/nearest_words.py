"""Inspect nearest terms from the learned distributional embedding space."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import load_dataset
from src.preprocessing import TextPreprocessor
from src.vectorization import DistributionalEmbeddingVectorizer


def nearest_words(args: argparse.Namespace) -> pd.DataFrame:
    df = load_dataset(args.dataset, args.data_dir, sample=args.sample, random_state=args.random_state)
    mode = "tweet" if args.dataset == "sentiment140" else "standard"
    clean_text = TextPreprocessor(mode=mode, reduction=args.reduction).transform(df["text"])
    vectorizer = DistributionalEmbeddingVectorizer(
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        max_features=args.max_features,
        min_df=2,
        random_state=args.random_state,
    ).fit(clean_text)

    rows = []
    for term in args.terms:
        for neighbor, similarity in vectorizer.nearest_terms(term, top_k=args.top_k):
            rows.append(
                {
                    "dataset": args.dataset,
                    "term": term,
                    "neighbor": neighbor,
                    "cosine_similarity": similarity,
                }
            )
    results = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["amazon", "sentiment140"], default="amazon")
    parser.add_argument("--terms", nargs="+", default=["good", "bad", "love", "hate"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--sample", type=int, default=10_000)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reduction", choices=["none", "stem", "lemma"], default="lemma")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="reports/nearest_words.csv")
    return parser.parse_args()


if __name__ == "__main__":
    print(nearest_words(parse_args()).to_string(index=False))
