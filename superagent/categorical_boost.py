"""A bounded, train-only CatBoost classifier with serializable input handling."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted


class CategoricalBoostClassifier(ClassifierMixin, BaseEstimator):
    """Fit native categorical boosting without accessing a validation split.

    Feature types and order come only from the fit frame. The fitted schema is
    reused for prediction, so a changed inference dtype cannot turn a categorical
    column into a numeric one. CatBoost is imported only when fit is requested.
    """

    def __init__(self, iterations=500, depth=6, learning_rate=.05,
                 l2_leaf_reg=3., class_weight=None, random_state=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.class_weight = class_weight
        self.random_state = random_state

    def _validate_parameters(self):
        for name, low, high in (("iterations", 100, 1000), ("depth", 3, 10),
                                ("random_state", 0, 2**32 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, low, high in (("learning_rate", .01, .3), ("l2_leaf_reg", 1., 20.)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be null or balanced")

    def _frame(self, X, *, fitted):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            values = np.asarray(X)
            if values.ndim != 2:
                raise ValueError("Features must be a two-dimensional matrix")
            if fitted and values.shape[1] != len(self.feature_columns_):
                raise ValueError("Prediction feature width differs from the fitted schema")
            frame = pd.DataFrame(values, columns=self.feature_columns_ if fitted else None)
        if not frame.columns.is_unique:
            raise ValueError("Features must have distinct columns")
        if fitted:
            if set(self.feature_columns_) - set(frame.columns):
                raise ValueError("Prediction input is missing a fitted feature column")
            frame = frame.loc[:, self.feature_columns_]
        if frame.empty:
            raise ValueError("Features must have rows and nonempty columns")
        return frame

    def _prepare(self, frame):
        prepared = frame.copy()
        for index, column in enumerate(self.feature_columns_):
            if index in self.categorical_indices_:
                values = prepared[column].astype(object)
                prepared[column] = values.where(values.notna(), "__missing__").astype(str)
            else:
                values = pd.to_numeric(prepared[column], errors="raise")
                prepared[column] = values.to_numpy(dtype=np.float64, na_value=np.nan)
                prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
        return prepared

    def fit(self, X, y):
        self._validate_parameters()
        frame = self._frame(X, fitted=False)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(frame):
            raise ValueError("Labels must be one-dimensional and align with feature rows")
        if pd.isna(labels).any() or type_of_target(labels) not in ("binary", "multiclass"):
            raise ValueError("Labels must be nonmissing binary or multiclass values")
        if len(pd.unique(labels)) < 2:
            raise ValueError("Training labels must contain at least two classes")

        # Existing models remain loadable in environments without this optional
        # backend; selecting it requires installing the declared dependency.
        from catboost import CatBoostClassifier

        for attribute in ("model_", "classes_", "feature_names_in_"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        self.feature_columns_ = list(frame.columns)
        self.categorical_indices_ = [index for index, dtype in enumerate(frame.dtypes)
                                     if not pd.api.types.is_numeric_dtype(dtype)]
        self.n_features_in_ = len(self.feature_columns_)
        if all(isinstance(name, str) for name in self.feature_columns_):
            self.feature_names_in_ = np.asarray(self.feature_columns_, dtype=object)
        settings = {
            "iterations": self.iterations, "depth": self.depth,
            "learning_rate": self.learning_rate, "l2_leaf_reg": self.l2_leaf_reg,
            "random_seed": self.random_state,
            "loss_function": "Logloss" if len(pd.unique(labels)) == 2 else "MultiClass",
            "task_type": "CPU", "thread_count": 1,
            "allow_writing_files": False, "verbose": False,
        }
        if self.class_weight == "balanced":
            settings["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(**settings)
        # No eval_set, use_best_model, or early stopping: all training decisions
        # depend exclusively on this fit call's training rows and fixed settings.
        model.fit(self._prepare(frame), labels, cat_features=self.categorical_indices_)
        self.model_ = model
        self.classes_ = np.asarray(model.classes_).copy()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        frame = self._prepare(self._frame(X, fitted=True))
        return np.asarray(self.model_.predict(frame)).reshape(-1)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        frame = self._prepare(self._frame(X, fitted=True))
        return np.asarray(self.model_.predict_proba(frame))
