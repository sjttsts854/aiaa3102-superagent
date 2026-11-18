"""Bounded, general-purpose sklearn candidates with serializable preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .data import TaskSpec
from .pair_features import PairedTfidfTransformer
from .pair_hybrid import PairedHybridTransformer
from .image_features import ImageGridFeatures


_TABULAR = {
    "logistic_regression": {"C": "float [0.001, 100] (default 1)", "class_weight": "null or balanced"},
    "random_forest": {"n_estimators": "integer [10, 400] (default 100)", "max_depth": "null or integer [1, 40]", "min_samples_leaf": "integer [1, 50]", "class_weight": "null or balanced"},
    "extra_trees": {"n_estimators": "integer [10, 400] (default 100)", "max_depth": "null or integer [1, 40]", "min_samples_leaf": "integer [1, 50]", "class_weight": "null or balanced"},
    "catboost": {
        "iterations": "integer [100, 1000] (default 500); native categorical boosting on CPU",
        "depth": "integer [3, 10] (default 6)",
        "learning_rate": "float [0.01, 0.3] (default 0.05)",
        "l2_leaf_reg": "float [1, 20] (default 3)",
        "class_weight": "null or balanced",
    },
    "hist_gradient_boosting": {
        "learning_rate": "float [0.01, 0.3] (default 0.1)",
        "max_iter": "integer [50, 600] (default 200)",
        "max_leaf_nodes": "integer [7, 63] (default 31)",
        "min_samples_leaf": "integer [5, 100] (default 20)",
        "l2_regularization": "float [0, 10] (default 0)",
        "class_weight": "null or balanced",
    },
}
_TFIDF = {"analyzer": "word or char (default word)", "ngram_range": "two-element integer array [low, high], 1 <= low <= high <= 5 (default [1, 1])", "max_features": "integer [100, 50000] (default 10000)", "min_df": "integer [1, 10] (default 1)"}
_TEXT = {
    "text_logistic": {**_TFIDF, "C": "float [0.001, 100] (default 1)", "class_weight": "null or balanced"},
    "text_naive_bayes": {**_TFIDF, "alpha": "float [0.001, 10] (default 1)"},
}
_PAIRED_TEXT = {
    "paired_text_hybrid": {
        "C": "float [0.001, 100] (default 1)",
        "class_weight": "null or balanced",
        "word_max_features": "integer [100, 50000] (default 20000); word unigrams and bigrams",
        "char_max_features": "integer [100, 50000] (default 30000)",
        "char_ngram_range": "two-element integer array [low, high], 1 <= low <= high <= 5 (default [3, 5])",
        "min_df": "integer [1, 10] (default 2)",
        "style_features": "boolean (default true); train-scaled generic length/punctuation/format differences",
    },
    "paired_text_logistic": {
        **_TFIDF,
        "C": "float [0.001, 100] (default 1)",
        "class_weight": "null or balanced",
        "representation": "difference (default, shared TF-IDF A minus B, no intercept) or slots (separate A/B coefficients with intercept)",
    },
}


_IMAGE = {
    "image_logistic": {
        "C": "float [0.001, 100] (default 1)",
        "class_weight": "null or balanced",
        "grid_size": "integer [2, 16] (default 8); at most image height/width; RGB grid averages and channel moments",
    },
}


def candidate_catalog(data_type: str) -> dict[str, dict[str, str]]:
    sources = {"tabular": _TABULAR, "text": _TEXT, "paired_text": _PAIRED_TEXT, "image": _IMAGE}
    if data_type not in sources:
        raise ValueError("Unsupported data type")
    source = sources[data_type]
    return {name: dict(params) for name, params in source.items()}


def _number(params: dict, key: str, default: Any, low: float, high: float, integer: bool = False):
    value = params.get(key, default)
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not low <= value <= high:
        raise ValueError(f"{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")
    return value


def _class_weight(params: dict):
    value = params.get("class_weight")
    if value not in (None, "balanced"):
        raise ValueError("class_weight must be null or balanced")
    return value


def _categorical_strings(X):
    """Avoid dtype mismatches and retain columns consisting entirely of missing values."""
    frame = pd.DataFrame(X).astype(object)
    return frame.where(frame.notna(), "__missing__").astype(str).to_numpy()


def _text_values(X):
    frame = pd.DataFrame(X)
    if frame.shape[1] != 1:
        raise ValueError("Text preprocessing expects exactly one text column")
    return frame.iloc[:, 0].fillna("").astype(str).to_numpy()


def build_pipeline(task: TaskSpec, candidate: dict, X_train: pd.DataFrame) -> Pipeline:
    """Construct an unfitted pipeline; fit it only on the intended training split."""
    if not isinstance(candidate, dict) or set(candidate) - {"algorithm", "params", "reason"}:
        raise ValueError("Candidate must contain algorithm and optional params/reason")
    algorithm = candidate.get("algorithm")
    catalog = candidate_catalog(task.data_type)
    if not isinstance(algorithm, str) or algorithm not in catalog:
        raise ValueError(f"Unsupported algorithm for {task.data_type}: {algorithm}")
    params = candidate.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("Candidate params must be an object")
    extra = set(params) - set(catalog[algorithm])
    if extra:
        raise ValueError(f"Unsupported parameters for {algorithm}: {sorted(extra)}")
    if task.data_type == "image":
        return Pipeline([
            ("image_features", ImageGridFeatures(grid_size=_number(params, "grid_size", 8, 2, 16, integer=True))),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=_number(params, "C", 1.0, .001, 100),
                                              class_weight=_class_weight(params),
                                              max_iter=1000, random_state=task.seed)),
        ])
    if task.data_type == "tabular":
        numeric = [c for c in X_train if pd.api.types.is_numeric_dtype(X_train[c])]
        categorical = [c for c in X_train if c not in numeric]
        if not numeric and not categorical:
            raise ValueError("No usable feature columns")
        if algorithm == "catboost":
            from .categorical_boost import CategoricalBoostClassifier
            classifier = CategoricalBoostClassifier(
                iterations=_number(params, "iterations", 500, 100, 1000, integer=True),
                depth=_number(params, "depth", 6, 3, 10, integer=True),
                learning_rate=_number(params, "learning_rate", .05, .01, .3),
                l2_leaf_reg=_number(params, "l2_leaf_reg", 3.0, 1, 20),
                class_weight=_class_weight(params), random_state=task.seed,
            )
            return Pipeline([("classifier", classifier)])
        if algorithm == "hist_gradient_boosting":
            transformers = []
            if numeric:
                # Histogram boosting handles missing numeric values directly.
                transformers.append(("numeric", "passthrough", numeric))
            if categorical:
                transformers.append(("categorical", Pipeline([
                    ("strings", FunctionTransformer(_categorical_strings)),
                    ("encode", OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan,
                        max_categories=254,
                    )),
                ]), categorical))
            preprocess = ColumnTransformer(transformers, sparse_threshold=0.0)
            classifier = HistGradientBoostingClassifier(
                learning_rate=_number(params, "learning_rate", .1, .01, .3),
                max_iter=_number(params, "max_iter", 200, 50, 600, integer=True),
                max_leaf_nodes=_number(params, "max_leaf_nodes", 31, 7, 63, integer=True),
                min_samples_leaf=_number(params, "min_samples_leaf", 20, 5, 100, integer=True),
                l2_regularization=_number(params, "l2_regularization", 0.0, 0, 10),
                categorical_features=[False] * len(numeric) + [True] * len(categorical),
                max_bins=255, early_stopping=False, class_weight=_class_weight(params),
                random_state=task.seed,
            )
            return Pipeline([("preprocess", preprocess), ("classifier", classifier)])
        transformers = []
        if numeric:
            transformers.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([
                ("strings", FunctionTransformer(_categorical_strings)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), categorical))
        if not transformers:
            raise ValueError("No usable feature columns")
        preprocess = ColumnTransformer(transformers, sparse_threshold=1.0)
        if algorithm == "logistic_regression":
            classifier = LogisticRegression(C=_number(params, "C", 1.0, .001, 100), class_weight=_class_weight(params), max_iter=1000, random_state=task.seed)
        else:
            depth = params.get("max_depth")
            if depth is not None:
                depth = _number(params, "max_depth", None, 1, 40, integer=True)
            cls = RandomForestClassifier if algorithm == "random_forest" else ExtraTreesClassifier
            classifier = cls(
                n_estimators=_number(params, "n_estimators", 100, 10, 400, integer=True),
                max_depth=depth,
                min_samples_leaf=_number(params, "min_samples_leaf", 1, 1, 50, integer=True),
                class_weight=_class_weight(params), random_state=task.seed, n_jobs=1,
            )
        return Pipeline([("preprocess", preprocess), ("classifier", classifier)])

    if algorithm == "paired_text_hybrid":
        ngram = params.get("char_ngram_range", [3, 5])
        if not isinstance(ngram, (list, tuple)) or len(ngram) != 2 or any(type(v) is not int for v in ngram) or not 1 <= ngram[0] <= ngram[1] <= 5:
            raise ValueError("char_ngram_range must contain integers satisfying 1 <= low <= high <= 5")
        style = params.get("style_features", True)
        if type(style) is not bool:
            raise ValueError("style_features must be a boolean")
        features = PairedHybridTransformer(
            columns=task.text_columns,
            word_max_features=_number(params, "word_max_features", 20000, 100, 50000, integer=True),
            char_max_features=_number(params, "char_max_features", 30000, 100, 50000, integer=True),
            char_ngram_range=tuple(ngram), min_df=_number(params, "min_df", 2, 1, 10, integer=True),
            style_features=style,
        )
        classifier = LogisticRegression(
            C=_number(params, "C", 1.0, .001, 100), class_weight=_class_weight(params),
            fit_intercept=False, max_iter=1000, random_state=task.seed,
        )
        return Pipeline([("paired_hybrid", features), ("classifier", classifier)])

    analyzer = params.get("analyzer", "word")
    if analyzer not in ("word", "char"):
        raise ValueError("analyzer must be word or char")
    ngram = params.get("ngram_range", [1, 1])
    if not isinstance(ngram, (list, tuple)) or len(ngram) != 2 or any(type(v) is not int for v in ngram) or not 1 <= ngram[0] <= ngram[1] <= 5:
        raise ValueError("ngram_range must contain integers satisfying 1 <= low <= high <= 5")
    max_features = _number(params, "max_features", 10000, 100, 50000, integer=True)
    min_df = _number(params, "min_df", 1, 1, 10, integer=True)
    if task.data_type == "paired_text":
        representation = params.get("representation", "difference")
        if representation not in ("difference", "slots"):
            raise ValueError("representation must be difference or slots")
        features = PairedTfidfTransformer(
            columns=task.text_columns, analyzer=analyzer, ngram_range=tuple(ngram),
            max_features=max_features, min_df=min_df, representation=representation,
        )
        classifier = LogisticRegression(
            C=_number(params, "C", 1.0, .001, 100), class_weight=_class_weight(params),
            fit_intercept=representation == "slots", max_iter=1000, random_state=task.seed,
        )
        return Pipeline([("paired_tfidf", features), ("classifier", classifier)])
    vectorizer = TfidfVectorizer(
        analyzer=analyzer, ngram_range=tuple(ngram),
        max_features=max_features, min_df=min_df, sublinear_tf=True,
    )
    if algorithm == "text_logistic":
        classifier = LogisticRegression(C=_number(params, "C", 1.0, .001, 100), class_weight=_class_weight(params), max_iter=1000, random_state=task.seed)
    else:
        classifier = MultinomialNB(alpha=_number(params, "alpha", 1.0, .001, 10))
    return Pipeline([("text", FunctionTransformer(_text_values)), ("tfidf", vectorizer), ("classifier", classifier)])


def score_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: str) -> float:
    if metric == "roc_auc":
        classes = pipeline.classes_
        if len(classes) != 2 or len(pd.unique(y)) != 2:
            raise ValueError("roc_auc requires two classes in both training and evaluation")
        if not set(pd.unique(y)).issubset(set(classes)):
            raise ValueError("Evaluation contains unknown class labels")
        # This agrees with sklearn's binary convention: sorted classes_[1] is positive.
        probability = pipeline.predict_proba(X)[:, 1]
        result = roc_auc_score(np.asarray(y) == classes[1], probability)
    else:
        prediction = pipeline.predict(X)
        if metric == "accuracy":
            result = accuracy_score(y, prediction)
        elif metric == "balanced_accuracy":
            result = balanced_accuracy_score(y, prediction)
        elif metric == "f1_macro":
            result = f1_score(y, prediction, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    if not np.isfinite(result):
        raise ValueError("Metric returned a nonfinite value")
    return float(result)
