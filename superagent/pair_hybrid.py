"""Generic ordered-pair text features with train-only vocabulary and scaling."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.validation import check_is_fitted

from .pair_features import PairedTfidfTransformer


_STYLE_NAMES = (
    "character_count", "word_count", "mean_word_length", "unique_word_fraction",
    "uppercase_fraction", "digit_count", "punctuation_count", "question_count",
    "exclamation_count", "comma_count", "period_count", "colon_count",
    "quote_count", "hyphen_count", "parenthesis_count", "whitespace_count",
)


def _style_values(texts):
    """Measure surface form only: no dataset vocabulary or label-specific rules."""
    rows = []
    for text in texts:
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        word_count = len(words)
        alpha_count = sum(character.isalpha() for character in text)
        rows.append([
            len(text), word_count,
            sum(map(len, words)) / max(word_count, 1),
            len({word.casefold() for word in words}) / max(word_count, 1),
            sum(character.isupper() for character in text) / max(alpha_count, 1),
            sum(character.isdigit() for character in text),
            sum(unicodedata.category(character).startswith("P") for character in text),
            sum(text.count(character) for character in "?？"),
            sum(text.count(character) for character in "!！"),
            sum(text.count(character) for character in ",，"),
            sum(text.count(character) for character in ".。"),
            sum(text.count(character) for character in ":：;；"),
            sum(text.count(character) for character in "\"'‘’“”"),
            sum(text.count(character) for character in "-–—"),
            sum(text.count(character) for character in "()（）[]【】"),
            sum(character.isspace() for character in text),
        ])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(_STYLE_NAMES))


class PairedHybridTransformer(TransformerMixin, BaseEstimator):
    """Combine word, character, and optional surface-form A-minus-B features.

    The word and character vocabularies are shared across the two training slots.
    Each TF-IDF difference block has unit L2 norm when nonzero. The style block
    uses training-difference scales without centering. Thus swapping the two
    documents negates every feature, including for previously unseen documents.
    """

    def __init__(self, columns, word_max_features=20000, char_max_features=30000,
                 char_ngram_range=(3, 5), min_df=2, style_features=True):
        self.columns = columns
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.style_features = style_features

    def _style_difference(self, X):
        left, right = self.word_._texts(X)
        return _style_values(left) - _style_values(right)

    def fit(self, X, y=None):
        if type(self.style_features) is not bool:
            raise ValueError("style_features must be a boolean")
        for name in ("word_max_features", "char_max_features", "min_df"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        ngram = self.char_ngram_range
        if (not isinstance(ngram, (list, tuple)) or len(ngram) != 2 or
                any(type(value) is not int for value in ngram) or
                not 1 <= ngram[0] <= ngram[1] <= 5):
            raise ValueError("char_ngram_range must satisfy 1 <= low <= high <= 5")
        self.word_ = PairedTfidfTransformer(
            columns=self.columns, analyzer="word", ngram_range=(1, 2),
            max_features=self.word_max_features, min_df=self.min_df,
            representation="difference",
        ).fit(X)
        self.char_ = PairedTfidfTransformer(
            columns=self.columns, analyzer="char", ngram_range=tuple(ngram),
            max_features=self.char_max_features, min_df=self.min_df,
            representation="difference",
        ).fit(X)
        self.style_scaler_ = (StandardScaler(with_mean=False).fit(self._style_difference(X))
                              if self.style_features else None)
        self.n_features_in_ = self.word_.n_features_in_
        return self

    def transform(self, X):
        check_is_fitted(self, ["word_", "char_", "style_scaler_"])
        blocks = [normalize(transformer.transform(X), norm="l2", copy=False)
                  for transformer in (self.word_, self.char_)]
        if self.style_scaler_ is not None:
            blocks.append(sparse.csr_matrix(self.style_scaler_.transform(self._style_difference(X))))
        return sparse.hstack(blocks, format="csr")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["word_", "char_", "style_scaler_"])
        names = [f"{prefix}__{name}" for prefix, transformer in (("word", self.word_), ("char", self.char_))
                 for name in transformer.get_feature_names_out()]
        if self.style_scaler_ is not None:
            names.extend(f"style_difference__{name}" for name in _STYLE_NAMES)
        return np.asarray(names, dtype=object)
