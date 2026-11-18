import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from superagent.data import TaskSpec
from superagent.models import build_pipeline


def build(tmp_path, X, **params):
    task = TaskSpec(data_type="tabular", train_path=tmp_path / "train.csv", target_column="label")
    return build_pipeline(task, {"algorithm": "hist_gradient_boosting", "params": {
        "max_iter": 50, "max_leaf_nodes": 7, "min_samples_leaf": 5, **params,
    }}, X)


@pytest.mark.parametrize("mode", ["numeric", "categorical", "mixed"])
def test_histogram_types_missing_unknown_and_serialization(tmp_path, mode):
    numeric = np.tile([-2., -1., 1., 2.], 30)
    categorical = np.tile(["blue", "blue", "red", "red"], 30).astype(object)
    X = pd.DataFrame({"category": categorical, "number": numeric})
    # Both supported input types include missing values; retain a numeric signal.
    X.loc[0, "number"] = np.nan
    X.loc[1, "category"] = None
    if mode == "numeric":
        X = X[["number"]]
    elif mode == "categorical":
        X = X[["category"]]
    y = (numeric > 0).astype(int)
    pipeline = build(tmp_path, X, class_weight="balanced")
    with threadpool_limits(limits=1):
        pipeline.fit(X, y)
        assert np.mean(pipeline.predict(X) == y) > .95
        valid = X.iloc[:3].copy()
        if "category" in valid:
            valid.loc[0, "category"] = "never_seen_during_training"
            valid.loc[2, "category"] = None
            encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"]
            original_categories = [values.copy() for values in encoder.categories_]
        expected = pipeline.predict_proba(valid)
        assert np.isfinite(expected).all()
        np.testing.assert_allclose(expected.sum(axis=1), 1)
        if "category" in valid:
            for before, after in zip(original_categories, encoder.categories_):
                np.testing.assert_array_equal(before, after)
            transformed = pipeline.named_steps["preprocess"].transform(valid)
            # Numeric columns precede categories irrespective of original frame order.
            assert np.isnan(transformed[0, -1])
        classifier = pipeline.named_steps["classifier"]
        assert classifier.early_stopping is False
        assert classifier.n_iter_ == 50
        expected_mask = ([False] if "number" in X else []) + ([True] if "category" in X else [])
        np.testing.assert_array_equal(classifier.is_categorical_, expected_mask if "category" in X else None)
        model_file = tmp_path / "hist.joblib"
        joblib.dump(pipeline, model_file)
        np.testing.assert_array_equal(expected, joblib.load(model_file).predict_proba(valid))


def test_high_cardinality_categories_bounded_to_native_limit(tmp_path):
    X = pd.DataFrame({"category": [f"value_{i}" for i in range(300)] * 2})
    y = np.tile(np.arange(300) % 2, 2)
    pipeline = build(tmp_path, X)
    with threadpool_limits(limits=1):
        pipeline.fit(X, y)
        encoded = pipeline.named_steps["preprocess"].transform(X)
        assert len(np.unique(encoded[:, 0])) <= 254
        assert encoded.max() < 255
        assert np.isfinite(pipeline.predict_proba(pd.DataFrame({"category": ["new", None]}))).all()


@pytest.mark.parametrize("params", [
    {"learning_rate": .001}, {"learning_rate": .31}, {"max_iter": 49},
    {"max_iter": 601}, {"max_leaf_nodes": 6}, {"max_leaf_nodes": 64},
    {"min_samples_leaf": 4}, {"min_samples_leaf": 101},
    {"l2_regularization": -1}, {"l2_regularization": 11},
    {"class_weight": "other"}, {"max_iter": True}, {"early_stopping": True},
])
def test_histogram_params_are_bounded(tmp_path, params):
    with pytest.raises(ValueError):
        build(tmp_path, pd.DataFrame({"x": [1, 2]}), **params)
