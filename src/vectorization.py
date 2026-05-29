"""Vectorizers used in the lab, including BM25 and distributional embeddings."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """Okapi BM25 document-term matrix transformer with sklearn conventions."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: int | None = 100_000,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 2,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

    def fit(self, raw_documents: list[str], y: object = None) -> "BM25Vectorizer":
        self.count_vectorizer_ = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        counts = self.count_vectorizer_.fit_transform(raw_documents).astype(np.float64)
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
        self.avgdl_ = float(doc_lengths.mean()) if len(doc_lengths) else 0.0
        df = np.diff(counts.tocsc().indptr)
        n_docs = counts.shape[0]
        self.idf_ = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self.vocabulary_ = self.count_vectorizer_.vocabulary_
        return self

    def transform(self, raw_documents: list[str]) -> sparse.csr_matrix:
        counts = self.count_vectorizer_.transform(raw_documents).astype(np.float64).tocsr()
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
        rows, cols = counts.nonzero()
        data = counts.data
        denom = data + self.k1 * (
            1.0 - self.b + self.b * doc_lengths[rows] / max(self.avgdl_, 1e-12)
        )
        counts.data = self.idf_[cols] * data * (self.k1 + 1.0) / denom
        return counts

    def fit_transform(self, raw_documents: list[str], y: object = None) -> sparse.csr_matrix:
        return self.fit(raw_documents).transform(raw_documents)


class DistributionalEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Document vectors built from PPMI/SVD word embeddings.

    Fit learns dense word vectors from local context co-occurrence. Transform
    represents each document as the mean of the embeddings for known tokens.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 4,
        max_features: int | None = 20_000,
        min_df: int | float = 2,
        random_state: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.max_features = max_features
        self.min_df = min_df
        self.random_state = random_state

    def fit(self, raw_documents: list[str], y: object = None) -> "DistributionalEmbeddingVectorizer":
        self.count_vectorizer_ = CountVectorizer(max_features=self.max_features, min_df=self.min_df)
        self.count_vectorizer_.fit(raw_documents)
        self.vocabulary_ = self.count_vectorizer_.vocabulary_
        self.terms_ = np.array(self.count_vectorizer_.get_feature_names_out())
        cooccurrence = self._build_cooccurrence(raw_documents)
        ppmi = self._to_ppmi(cooccurrence)
        n_components = min(self.embedding_dim, max(1, ppmi.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.embeddings_ = normalize(self.svd_.fit_transform(ppmi))
        return self

    def transform(self, raw_documents: list[str]) -> np.ndarray:
        vectors = np.zeros((len(raw_documents), self.embeddings_.shape[1]), dtype=np.float64)
        for row, document in enumerate(raw_documents):
            indices = [self.vocabulary_[token] for token in document.split() if token in self.vocabulary_]
            if indices:
                vectors[row] = self.embeddings_[indices].mean(axis=0)
        return vectors

    def fit_transform(self, raw_documents: list[str], y: object = None) -> np.ndarray:
        return self.fit(raw_documents).transform(raw_documents)

    def nearest_terms(self, term: str, top_k: int = 10) -> list[tuple[str, float]]:
        if term not in self.vocabulary_:
            return []
        idx = self.vocabulary_[term]
        scores = self.embeddings_ @ self.embeddings_[idx]
        order = np.argsort(scores)[::-1]
        neighbors = []
        for neighbor_idx in order:
            if neighbor_idx == idx:
                continue
            neighbors.append((str(self.terms_[neighbor_idx]), float(scores[neighbor_idx])))
            if len(neighbors) == top_k:
                break
        return neighbors

    def _build_cooccurrence(self, raw_documents: list[str]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for document in raw_documents:
            tokens = [token for token in document.split() if token in self.vocabulary_]
            ids = [self.vocabulary_[token] for token in tokens]
            for center_pos, center_id in enumerate(ids):
                start = max(0, center_pos - self.window_size)
                end = min(len(ids), center_pos + self.window_size + 1)
                for context_pos in range(start, end):
                    if context_pos == center_pos:
                        continue
                    rows.append(center_id)
                    cols.append(ids[context_pos])
                    data.append(1.0 / abs(context_pos - center_pos))
        size = len(self.vocabulary_)
        return sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

    def _to_ppmi(self, cooccurrence: sparse.csr_matrix) -> sparse.csr_matrix:
        cooccurrence = cooccurrence.astype(np.float64).tocoo()
        total = cooccurrence.data.sum()
        if total == 0:
            return cooccurrence.tocsr()
        row_sums = np.asarray(cooccurrence.sum(axis=1)).ravel()
        col_sums = np.asarray(cooccurrence.sum(axis=0)).ravel()
        expected = row_sums[cooccurrence.row] * col_sums[cooccurrence.col]
        pmi = np.log((cooccurrence.data * total) / np.maximum(expected, 1e-12))
        ppmi_data = np.maximum(pmi, 0.0)
        keep = ppmi_data > 0
        return sparse.coo_matrix(
            (ppmi_data[keep], (cooccurrence.row[keep], cooccurrence.col[keep])),
            shape=cooccurrence.shape,
        ).tocsr()
