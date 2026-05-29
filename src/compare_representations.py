"""Run comparable sentiment experiments for sparse and embedding representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.train import train


def run_comparison(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for dataset in args.datasets:
        for vectorizer in args.vectorizers:
            run_args = argparse.Namespace(
                dataset=dataset,
                vectorizer=vectorizer,
                reduction=args.reduction,
                data_dir=args.data_dir,
                model_dir=args.model_dir,
                sample=args.sample,
                test_size=args.test_size,
                random_state=args.random_state,
                experiment_name=args.experiment_name,
                run_name=f"{dataset}-{vectorizer}-{args.reduction}",
            )
            rows.append(train(run_args))
    results = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    results.to_markdown(output.with_suffix(".md"), index=False)
    metric_summary = {
        "rows": int(len(results)),
        "best_f1": float(results["f1"].max()),
        "best_accuracy": float(results["accuracy"].max()),
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["amazon", "sentiment140"], default=["amazon", "sentiment140"])
    parser.add_argument(
        "--vectorizers",
        nargs="+",
        choices=["bow", "tfidf", "bm25", "embedding"],
        default=["bow", "tfidf", "bm25", "embedding"],
    )
    parser.add_argument("--reduction", choices=["none", "stem", "lemma"], default="lemma")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--sample", type=int, default=5_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", default="sentiment-lab")
    parser.add_argument("--output", default="reports/model_comparison.csv")
    return parser.parse_args()


if __name__ == "__main__":
    print(run_comparison(parse_args()).to_string(index=False))
