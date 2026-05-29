"""Reusable text preprocessing for food reviews and tweets."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from pathlib import Path
import re
import string
from typing import Iterable, Literal

from sklearn.base import BaseEstimator, TransformerMixin

try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    nltk.data.path.append(str(Path("data/external/nltk_data").resolve()))
except ImportError:  # pragma: no cover - dependency is declared, guard keeps imports friendly.
    nltk = None
    PorterStemmer = None
    WordNetLemmatizer = None


NormalizationMode = Literal["minimal", "standard", "tweet"]
ReductionMode = Literal["none", "stem", "lemma"]


_CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}


@dataclass
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Configurable normalizer/tokenizer usable inside sklearn pipelines.

    The same class supports both datasets. Amazon reviews usually benefit from
    standard punctuation cleanup, while tweets need URL, mention, and hashtag
    handling that preserves sentiment-bearing markers such as exclamation marks.
    """

    mode: NormalizationMode = "standard"
    reduction: ReductionMode = "lemma"
    lowercase: bool = True
    remove_stopwords: bool = True
    keep_negations: bool = True
    min_token_length: int = 2
    extra_stopwords: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._stemmer = PorterStemmer() if PorterStemmer and self.reduction == "stem" else None
        self._lemmatizer = self._make_lemmatizer()
        self._stopwords = self._build_stopwords()

    def fit(self, X: Iterable[str], y: object = None) -> "TextPreprocessor":
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [self.normalize(text) for text in X]

    def normalize(self, text: object) -> str:
        text = "" if text is None else str(text)
        text = html.unescape(text)
        if self.lowercase:
            text = text.lower()

        text = self._expand_contractions(text)
        if self.mode == "tweet":
            text = self._normalize_tweet(text)
        else:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self.mode != "minimal":
            text = text.translate(str.maketrans({p: " " for p in string.punctuation}))
            text = re.sub(r"\d+", " <num> ", text)

        tokens = self.tokenize(text)
        return " ".join(tokens)

    def tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z_<>']+", text)
        cleaned: list[str] = []
        for token in tokens:
            token = token.strip("'_")
            if not token or len(token) < self.min_token_length:
                continue
            if self.remove_stopwords and token in self._stopwords:
                continue
            cleaned.append(self._reduce(token))
        return cleaned

    def _normalize_tweet(self, text: str) -> str:
        text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
        text = re.sub(r"@\w+", " <user> ", text)
        text = re.sub(r"#(\w+)", r" \1 ", text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        return text

    def _expand_contractions(self, text: str) -> str:
        for source, target in _CONTRACTIONS.items():
            text = text.replace(source, target)
        return text

    def _reduce(self, token: str) -> str:
        if self._stemmer:
            return self._stemmer.stem(token)
        if self._lemmatizer:
            try:
                return self._lemmatizer.lemmatize(token)
            except LookupError:
                return token
        return token

    def _make_lemmatizer(self):
        if not WordNetLemmatizer or self.reduction != "lemma":
            return None
        if nltk:
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                return None
        return WordNetLemmatizer()

    def _build_stopwords(self) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
        }
        if not self.keep_negations:
            stopwords.update({"no", "nor", "not"})
        return stopwords | set(self.extra_stopwords)
