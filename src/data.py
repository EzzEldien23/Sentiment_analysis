"""Dataset loading utilities for the sentiment lab."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


DatasetName = Literal["amazon", "sentiment140"]


def load_amazon(path: str | Path, sample: int | None = None, random_state: int = 42) -> pd.DataFrame:
    columns = ["Id", "Score", "Summary", "Text"]
    nrows = sample * 3 if sample else None
    df = pd.read_csv(path, usecols=columns, nrows=nrows)
    df = df[df["Score"] != 3].copy()
    df["label"] = (df["Score"] >= 4).astype(int)
    df["text"] = (df["Summary"].fillna("") + " " + df["Text"].fillna("")).str.strip()
    df["dataset"] = "amazon"
    if sample and sample < len(df):
        df = df.sample(sample, random_state=random_state)
    return df[["dataset", "Id", "text", "label", "Score"]].rename(columns={"Id": "doc_id"})


def load_sentiment140(
    path: str | Path, sample: int | None = None, random_state: int = 42
) -> pd.DataFrame:
    names = ["target", "tweet_id", "date", "flag", "user", "text"]
    if sample:
        half = max(sample // 2, 1)
        negative = pd.read_csv(path, encoding="latin-1", header=None, names=names, nrows=half)
        positive = pd.read_csv(
            path,
            encoding="latin-1",
            header=None,
            names=names,
            skiprows=range(800_000),
            nrows=sample - half,
        )
        df = pd.concat([negative, positive], ignore_index=True)
    else:
        df = pd.read_csv(path, encoding="latin-1", header=None, names=names)
    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = (df["target"] == 4).astype(int)
    df["dataset"] = "sentiment140"
    if sample and sample < len(df):
        df = df.sample(sample, random_state=random_state)
    return df[["dataset", "tweet_id", "text", "label", "target"]].rename(
        columns={"tweet_id": "doc_id"}
    )


def load_dataset(
    name: DatasetName,
    data_dir: str | Path = "data/raw",
    sample: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if name == "amazon":
        return load_amazon(data_dir / "Reviews.csv", sample, random_state)
    if name == "sentiment140":
        return load_sentiment140(
            data_dir / "training.1600000.processed.noemoticon.csv", sample, random_state
        )
    raise ValueError(f"Unknown dataset: {name}")
