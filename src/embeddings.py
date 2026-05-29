"""Create simple distributional word embeddings and a 2D visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from src.data import load_dataset
from src.preprocessing import TextPreprocessor


def build_embeddings(args: argparse.Namespace) -> Path:
    df = load_dataset(args.dataset, args.data_dir, sample=args.sample, random_state=args.random_state)
    mode = "tweet" if args.dataset == "sentiment140" else "standard"
    preprocessor = TextPreprocessor(mode=mode, reduction=args.reduction)
    clean_text = preprocessor.transform(df["text"])

    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=2)
    document_term = vectorizer.fit_transform(clean_text)
    term_document = document_term.T
    components = min(args.dimensions, max(2, term_document.shape[1] - 1))
    embeddings = TruncatedSVD(n_components=components, random_state=args.random_state).fit_transform(
        term_document
    )

    terms = np.array(vectorizer.get_feature_names_out())
    norms = np.linalg.norm(embeddings, axis=1)
    top_idx = np.argsort(norms)[-args.plot_terms :]
    perplexity = min(30, max(2, len(top_idx) // 3))
    points = TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=args.random_state,
    ).fit_transform(embeddings[top_idx])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "term": terms,
            **{f"dim_{i}": embeddings[:, i] for i in range(embeddings.shape[1])},
        }
    ).to_parquet(output.with_suffix(".parquet"), index=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], s=18, alpha=0.75)
    for term, (x, y) in zip(terms[top_idx], points, strict=True):
        plt.text(x, y, term, fontsize=8)
    plt.title(f"{args.dataset} distributional word embeddings")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["amazon", "sentiment140"], default="amazon")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--dimensions", type=int, default=100)
    parser.add_argument("--plot-terms", type=int, default=150)
    parser.add_argument("--reduction", choices=["none", "stem", "lemma"], default="lemma")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="reports/figures/amazon_embeddings.png")
    return parser.parse_args()


if __name__ == "__main__":
    print(build_embeddings(parse_args()))
