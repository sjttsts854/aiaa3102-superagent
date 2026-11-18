"""Train-only shared text features for ordered pairs of arbitrary documents."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted


class PairedTfidfTransformer(TransformerMixin, BaseEstimator):
    """Represent A/B using one vocabulary fitted on both training columns.

    Difference features preserve pair order and negate exactly when A/B swap.
    Slots retain independent A/B coefficients while still sharing the vocabulary.
    Neither transform nor prediction updates document frequencies or vocabulary.
    """

    def __init__(self, columns, analyzer="word", ngram_range=(1, 1),
                 max_features=10000, min_df=1, representation="difference"):
        self.columns = columns
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.representation = representation

    def _texts(self, X):
        if len(self.columns) != 2 or len(set(self.columns)) != 2:
            raise ValueError("Paired text needs two distinct ordered columns")
        frame = pd.DataFrame(X)
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError("Paired text input is missing a configured column")
        return [frame[column].fillna("").astype(str).to_numpy() for column in self.columns]

    def fit(self, X, y=None):
        if self.representation not in ("difference", "slots"):
            raise ValueError("representation must be difference or slots")
        left, right = self._texts(X)
        self.vectorizer_ = TfidfVectorizer(
            analyzer=self.analyzer, ngram_range=self.ngram_range,
            max_features=self.max_features, min_df=self.min_df, sublinear_tf=True,
        ).fit(np.concatenate([left, right]))
        self.n_features_in_ = len(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "vectorizer_")
        left, right = self._texts(X)
        left_matrix = self.vectorizer_.transform(left)
        right_matrix = self.vectorizer_.transform(right)
        if self.representation == "difference":
            return (left_matrix - right_matrix).tocsr()
        return sparse.hstack([left_matrix, right_matrix], format="csr")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "vectorizer_")
        names = self.vectorizer_.get_feature_names_out()
        if self.representation == "difference":
            return np.asarray([f"difference__{name}" for name in names], dtype=object)
        return np.asarray([f"{slot}__{name}" for slot in ("left", "right") for name in names], dtype=object)
