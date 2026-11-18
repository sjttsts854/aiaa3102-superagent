import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError

from superagent.categorical_boost import CategoricalBoostClassifier


def classifier(**kwargs):
    return CategoricalBoostClassifier(iterations=100, depth=3, learning_rate=.1, **kwargs)


@pytest.mark.parametrize("mode", ["numeric", "categorical", "mixed"])
def test_binary_schema_missing_unknown_and_roundtrip(tmp_path, mode):
    values = np.tile([-3., -1., 1., 3.], 30)
    frame = pd.DataFrame({"category": np.where(values > 0, "right", "left").astype(object),
                          "number": values, "always_missing": np.full(len(values), np.nan)})
    frame.loc[0, "category"] = None
    frame.loc[1, "number"] = np.nan
    if mode == "numeric":
        frame = frame[["number", "always_missing"]]
    elif mode == "categorical":
        frame = frame[["category"]]
    labels = (values > 0).astype(int)
    model = classifier(class_weight="balanced")
    assert is_classifier(model)
    assert clone(model).get_params() == model.get_params()
    model.fit(frame, labels)
    assert model.predict(frame).shape == (len(frame),)
    assert model.score(frame, labels) > .95
    assert model.n_features_in_ == frame.shape[1]
    np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
    probe = frame.iloc[:4].copy()
    if "category" in probe:
        probe.loc[0, "category"] = "not_seen_during_training"
        probe.loc[1, "category"] = None
    before = model.model_.tree_count_
    expected = model.predict_proba(probe)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(expected.sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict_proba(probe[probe.columns[::-1]]), expected)
    assert model.model_.tree_count_ == before == 100
    filename = tmp_path / "category.joblib"
    joblib.dump(model, filename)
    loaded = joblib.load(filename)
    np.testing.assert_array_equal(loaded.predict_proba(probe), expected)
    assert loaded.model_.get_params()["allow_writing_files"] is False
    assert loaded.model_.get_params()["thread_count"] == 1
    with pytest.raises(ValueError, match="missing a fitted feature"):
        loaded.predict(probe.drop(columns=[probe.columns[0]]))


def test_multiclass_preserves_labels_and_flattens_prediction():
    frame = pd.DataFrame({"category": np.tile(["red", "blue", "green"], 30),
                          "number": np.tile([0., 1., 2.], 30)})
    labels = np.tile(["first", "second", "third"], 30)
    model = classifier().fit(frame, labels)
    assert model.predict(frame).shape == labels.shape
    assert set(model.classes_) == set(labels)
    assert model.predict_proba(frame).shape == (90, 3)
    np.testing.assert_array_equal(model.predict(frame), labels)


def test_training_dtype_stays_categorical_when_prediction_values_are_numbers():
    frame = pd.DataFrame({"category": pd.Series([1, 2] * 30, dtype=object)})
    model = classifier().fit(frame, np.tile([0, 1], 30))
    original_types = list(model.categorical_indices_)
    np.testing.assert_array_equal(model.predict_proba(pd.DataFrame({"category": [1, 2]})),
                                  model.predict_proba(pd.DataFrame({"category": ["1", "2"]})))
    assert model.categorical_indices_ == original_types == [0]


def test_numpy_input_and_nullable_numeric():
    frame = pd.DataFrame({"n": pd.Series([1, 2, 3, 4] * 15, dtype="Int64")})
    frame.loc[0, "n"] = pd.NA
    labels = np.tile([0, 0, 1, 1], 15)
    model = classifier().fit(frame, labels)
    assert np.isfinite(model.predict_proba(frame)).all()
    numeric = frame.to_numpy(dtype=float, na_value=np.nan)
    np.testing.assert_array_equal(model.predict(numeric), model.predict(frame))
    fresh = classifier().fit(numeric, labels)
    assert fresh.predict(numeric).shape == (60,)


@pytest.mark.parametrize("updates", [
    {"iterations": 99}, {"iterations": 1001}, {"iterations": True},
    {"depth": 2}, {"depth": 11}, {"learning_rate": .001},
    {"learning_rate": float("nan")}, {"l2_leaf_reg": 0},
    {"l2_leaf_reg": 21}, {"class_weight": "not_supported"},
])
def test_invalid_configuration_rejected_before_fit(updates):
    model = classifier().set_params(**updates)
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame({"number": [1., 2.]}), [0, 1])


def test_not_fitted_and_bad_training_shapes():
    model = classifier()
    with pytest.raises(NotFittedError):
        model.predict(pd.DataFrame({"number": [1., 2.]}))
    for labels in ([0], [[0], [1]], [0, None], [1, 1]):
        with pytest.raises(ValueError):
            model.fit(pd.DataFrame({"number": [1., 2.]}), labels)
