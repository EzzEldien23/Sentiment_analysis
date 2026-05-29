"""Download NLTK resources used by reproducible preprocessing stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import nltk


def prepare_nltk_data(output_dir: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nltk.download("wordnet", download_dir=str(output), quiet=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/external/nltk_data")
    return parser.parse_args()


if __name__ == "__main__":
    print(prepare_nltk_data(parse_args().output_dir))
