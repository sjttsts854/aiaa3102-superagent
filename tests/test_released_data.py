"""Exercise generic formats with synthetic fixtures; official test data stays sealed."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superagent.data import TaskSpec, load_data, load_prediction_data, load_task


def npy_task(tmp_path, **overrides):
    np.save(tmp_path / "features.npy", np.array([[1, "first"], [2, None], [3, "third"], [4, "fourth"]], dtype=object))
    np.save(tmp_path / "labels.npy", np.array([1, 0, 1, 0]))
    np.save(tmp_path / "valid.npy", np.array([[98, "valid-only"], [99, None]], dtype=object))
    np.save(tmp_path / "valid_labels.npy", np.array([0, 1]))
    config = dict(data_type="tabular", data_format="npy", train_path=tmp_path / "features.npy",
                  train_labels_path=tmp_path / "labels.npy", validation_path=tmp_path / "valid.npy",
                  validation_labels_path=tmp_path / "valid_labels.npy", target_column="target",
                  feature_names=["code", "category"], categorical_columns=["code", "category"])
    config.update(overrides)
    return TaskSpec(**config)


def test_npy_preserves_rows_categories_validation_and_sealed_prediction(tmp_path, monkeypatch):
    task = npy_task(tmp_path, predict_path=tmp_path / "sealed.npy")
    loaded_paths = []
    original_load = np.load

    def record_load(path, **kwargs):
        loaded_paths.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, "load", record_load)
    data = load_data(task)
    assert loaded_paths == [task.train_path, task.train_labels_path, task.validation_path, task.validation_labels_path]
    assert data.X_train["code"].tolist() == [1, 2, 3, 4]
    assert data.y_train.tolist() == [1, 0, 1, 0]
    assert data.X_valid["code"].tolist() == [98, 99]
    assert data.y_valid.tolist() == [0, 1]
    assert all(dtype == object for dtype in data.X_train.dtypes)
    assert len(data.X_refit) == 4
    assert "valid-only" not in data.X_refit["category"].tolist()
    assert data.profile["categorical_feature_count"] == 2
    assert data.profile["numeric_feature_count"] == 0
    assert data.split_manifest["train_labels_source"] == str(task.train_labels_path)
    assert data.split_manifest["validation_labels_source"] == str(task.validation_labels_path)
    assert "category" not in json.dumps(data.profile)


def test_npy_prediction_does_not_load_training_or_any_labels(tmp_path, monkeypatch):
    task = npy_task(tmp_path, predict_path=tmp_path / "prediction.npy")
    np.save(task.predict_path, np.array([[50, "unseen"], [51, None]], dtype=object))
    original_load = np.load

    def prediction_only(path, **kwargs):
        assert path == task.predict_path
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, "load", prediction_only)
    X, ids = load_prediction_data(task, ["code", "category"])
    assert X["code"].tolist() == [50, 51]
    assert ids is None
    assert X["code"].dtype == object


@pytest.mark.parametrize("invalid_array,expected", [
    (np.zeros(4), "two-dimensional"),
    (np.zeros((0, 2)), "two-dimensional"),
    (np.zeros((4, 3)), "feature width"),
])
def test_npy_rejects_invalid_feature_shape(tmp_path, invalid_array, expected):
    task = npy_task(tmp_path)
    np.save(task.train_path, invalid_array)
    with pytest.raises(ValueError, match=expected):
        load_data(task)


@pytest.mark.parametrize("invalid_labels,expected", [
    (np.array([0, 1]), "row counts"),
    (np.array([[0], [1], [0], [1]]), "one-dimensional"),
    (np.array([0., 1., np.nan, 0.]), "must not be missing"),
])
def test_npy_rejects_invalid_labels(tmp_path, invalid_labels, expected):
    task = npy_task(tmp_path)
    np.save(task.train_labels_path, invalid_labels)
    with pytest.raises(ValueError, match=expected):
        load_data(task)


def test_npy_manifest_resolves_all_paths_and_rejects_overlap(tmp_path):
    task = npy_task(tmp_path)
    manifest = tmp_path / "task.json"
    config = dict(data_type="tabular", data_format="npy", train_path="features.npy",
                  train_labels_path="labels.npy", validation_path="valid.npy",
                  validation_labels_path="valid_labels.npy", feature_names=["code", "category"],
                  target_column="target")
    manifest.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_task(manifest)
    assert loaded.train_labels_path == task.train_labels_path
    assert loaded.validation_labels_path == task.validation_labels_path
    with pytest.raises(ValueError, match="distinct paths"):
        replace(task, validation_labels_path=task.train_labels_path)
    with pytest.raises(ValueError, match="distinct paths"):
        replace(task, predict_path=task.train_labels_path)
    with pytest.raises(ValueError, match="provided together"):
        replace(task, validation_labels_path=None)


@pytest.mark.parametrize("overrides", [
    {"feature_names": None},
    {"feature_names": ["same", "same"]},
    {"feature_names": ["code", "target"]},
    {"train_labels_path": None},
    {"data_format": "unknown"},
    {"categorical_columns": ["target"]},
])
def test_npy_invalid_manifest_fields(tmp_path, overrides):
    with pytest.raises(ValueError):
        npy_task(tmp_path, **overrides)


def paired_task(tmp_path, **overrides):
    pd.DataFrame({"right": ["right one", "right two", None, "right four"],
                  "target": [1, 0, 1, 0], "left": ["left one", "left two", "left three", "left four"],
                  "irrelevant": ["ignore"] * 4}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"left": ["held-out left", "held-out other"], "right": ["held-out right", "held-out last"],
                  "target": [0, 1]}).to_csv(tmp_path / "valid.csv", index=False)
    config = dict(data_type="paired_text", train_path=tmp_path / "train.csv",
                  validation_path=tmp_path / "valid.csv", target_column="target", text_columns=["left", "right"])
    config.update(overrides)
    return TaskSpec(**config)


def test_paired_text_preserves_order_and_profiles_training_only(tmp_path):
    task = paired_task(tmp_path, predict_path=tmp_path / "sealed.csv")
    data = load_data(task)
    assert data.feature_columns == ["left", "right"]
    assert data.X_train.columns.tolist() == ["left", "right"]
    assert data.X_valid.columns.tolist() == ["left", "right"]
    assert data.X_train.iloc[0].tolist() == ["left one", "right one"]
    assert data.profile["empty_text_fractions"] == [0., .25]
    assert data.profile["training_rows"] == len(data.X_refit) == 4
    assert "held-out" not in json.dumps(data.profile)
    assert "left" not in json.dumps(data.profile)
    assert "irrelevant" not in data.X_train


def test_paired_prediction_ignores_labels_and_retains_roles(tmp_path):
    task = paired_task(tmp_path, predict_path=tmp_path / "predict.csv")
    pd.DataFrame({"right": ["second"], "target": ["sealed-label"], "left": ["first"]}).to_csv(task.predict_path, index=False)
    X, ids = load_prediction_data(task, ["left", "right"])
    assert X.iloc[0].tolist() == ["first", "second"]
    assert ids is None


@pytest.mark.parametrize("overrides", [
    {"text_columns": None}, {"text_columns": ["left"]},
    {"text_columns": ["left", "left"]}, {"text_columns": ["target", "right"]},
    {"text_columns": ["left", "right"], "id_column": "left"},
    {"text_column": "left"}, {"categorical_columns": ["left"]},
])
def test_paired_text_rejects_invalid_roles(tmp_path, overrides):
    with pytest.raises(ValueError):
        paired_task(tmp_path, **overrides)


def test_missing_paired_column_fails_clearly(tmp_path):
    task = paired_task(tmp_path, text_columns=["missing", "right"])
    with pytest.raises(ValueError, match="Missing feature columns"):
        load_data(task)


def test_csv_categorical_declaration_overrides_numeric_inference(tmp_path):
    pd.DataFrame({"code": [1, 2] * 20, "target": [0, 1] * 20}).to_csv(tmp_path / "train.csv", index=False)
    task = TaskSpec(data_type="tabular", train_path=tmp_path / "train.csv", target_column="target", categorical_columns=["code"])
    data = load_data(task)
    assert data.X_train["code"].dtype == object
    assert data.profile["numeric_feature_count"] == 0
    with pytest.raises(ValueError, match="Missing categorical columns"):
        load_data(replace(task, categorical_columns=["missing"]))
