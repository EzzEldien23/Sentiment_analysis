"""Train and track sentiment classifiers across datasets and vectorizers."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.preprocessing import TextPreprocessor
from src.vectorization import BM25Vectorizer, DistributionalEmbeddingVectorizer


def maybe_import_mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def build_pipeline(dataset: str, vectorizer: str, reduction: str) -> Pipeline:
    mode = "tweet" if dataset == "sentiment140" else "standard"
    if vectorizer == "bow":
        vectorizer_step = CountVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2)
    elif vectorizer == "tfidf":
        vectorizer_step = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2)
    elif vectorizer == "bm25":
        vectorizer_step = BM25Vectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2)
    elif vectorizer == "embedding":
        vectorizer_step = DistributionalEmbeddingVectorizer(
            embedding_dim=100,
            window_size=4,
            max_features=20_000,
            min_df=2,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported vectorizer: {vectorizer}")

    return Pipeline(
        [
            ("preprocess", TextPreprocessor(mode=mode, reduction=reduction)),
            ("vectorize", vectorizer_step),
            ("classifier", LogisticRegression(max_iter=1_000, n_jobs=1, class_weight="balanced")),
        ]
    )


def train(args: argparse.Namespace) -> dict[str, float | str]:
    df = load_dataset(args.dataset, args.data_dir, sample=args.sample, random_state=args.random_state)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label"],
        random_state=args.random_state,
    )
    pipeline = build_pipeline(args.dataset, args.vectorizer, args.reduction)

    mlflow = maybe_import_mlflow()
    if mlflow:
        mlflow.set_experiment(args.experiment_name)

    started = time.time()
    run_context = mlflow.start_run(run_name=args.run_name) if mlflow else nullcontext()
    with run_context:
        pipeline.fit(train_df["text"], train_df["label"])
        predictions = pipeline.predict(test_df["text"])
        vectorizer = pipeline.named_steps["vectorize"]
        vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}))
        metrics = {
            "accuracy": accuracy_score(test_df["label"], predictions),
            "precision": precision_score(test_df["label"], predictions, zero_division=0),
            "recall": recall_score(test_df["label"], predictions, zero_division=0),
            "f1": f1_score(test_df["label"], predictions),
            "vocabulary_size": vocabulary_size,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "duration_seconds": time.time() - started,
        }
        params = {
            "dataset": args.dataset,
            "vectorizer": args.vectorizer,
            "reduction": args.reduction,
            "sample": args.sample or "all",
            "test_size": args.test_size,
        }
        if mlflow:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")

        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{args.dataset}_{args.vectorizer}_{args.reduction}.joblib"
        joblib.dump(pipeline, model_path)

        report_path = model_dir / f"{args.dataset}_{args.vectorizer}_{args.reduction}_report.txt"
        report_path.write_text(
            classification_report(test_df["label"], predictions, target_names=["negative", "positive"]),
            encoding="utf-8",
        )

    return {**params, **metrics, "model_path": str(model_path), "report_path": str(report_path)}


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["amazon", "sentiment140"], required=True)
    parser.add_argument("--vectorizer", choices=["bow", "tfidf", "bm25", "embedding"], default="tfidf")
    parser.add_argument("--reduction", choices=["none", "stem", "lemma"], default="lemma")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", default="sentiment-lab")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(pd.Series(train(parse_args())).to_string())
