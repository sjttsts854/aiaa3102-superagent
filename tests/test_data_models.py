import json

import joblib
import numpy as np
import pandas as pd
import pytest

from superagent.data import TaskSpec, load_data, load_prediction_data, load_task
from superagent.models import build_pipeline, candidate_catalog, score_pipeline


def make_task(tmp_path, **kwargs):
    frame = pd.DataFrame({"id": [f"{i:03}" for i in range(40)], "number": list(range(40)), "category": ["a", "b"] * 20, "label": [0, 1] * 20})
    frame.to_csv(tmp_path / "train.csv", index=False)
    return TaskSpec(data_type="tabular", train_path=tmp_path / "train.csv", target_column="label", id_column="id", **kwargs)


def test_prediction_deferred_and_id_target_excluded(tmp_path):
    task = make_task(tmp_path, predict_path=tmp_path / "not_created.csv")
    data = load_data(task)
    assert data.feature_columns == ["number", "category"]
    assert len(data.X_refit) == 40
    assert set(data.split_manifest["train_indices"]).isdisjoint(data.split_manifest["validation_indices"])
    assert "train.csv" not in json.dumps(data.profile)
    assert "category" not in data.profile
    with pytest.raises(FileNotFoundError):
        load_prediction_data(task, data.feature_columns)


def test_pipeline_learns_only_train_and_handles_unseen_category(tmp_path):
    task = make_task(tmp_path)
    X_train = pd.DataFrame({"number": [0., 2., np.nan, 4.], "category": ["a", "b", None, "a"]})
    pipeline = build_pipeline(task, {"algorithm": "logistic_regression"}, X_train)
    pipeline.fit(X_train, [0, 1, 0, 1])
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == 2.0
    valid = pd.DataFrame({"number": [1e9, np.nan], "category": ["unseen", None]})
    original = imputer.statistics_.copy()
    assert len(pipeline.predict(valid)) == 2
    np.testing.assert_array_equal(original, imputer.statistics_)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    np.testing.assert_array_equal(pipeline.predict(valid), joblib.load(model_path).predict(valid))


def test_prediction_labels_ignored_ids_preserved(tmp_path):
    task = make_task(tmp_path, predict_path=tmp_path / "predict.csv")
    pd.DataFrame({"id": ["001", "002"], "number": [1, 2], "category": ["new", "a"], "label": ["secret", "secret"]}).to_csv(task.predict_path, index=False)
    X, ids = load_prediction_data(task, load_data(task).feature_columns)
    assert "label" not in X
    assert ids.tolist() == ["001", "002"]


@pytest.mark.parametrize("algorithm", ["text_logistic", "text_naive_bayes"])
def test_text_missing_serialize_and_binary_auc(tmp_path, algorithm):
    task = TaskSpec(data_type="text", train_path=tmp_path / "train.csv", target_column="label", text_column="message", metric="roc_auc")
    X = pd.DataFrame({"message": ["great cheerful", None, "awful sad", "great warm", "awful cold", "sad bad"]})
    y = pd.Series(["yes", "no", "no", "yes", "no", "no"])
    pipeline = build_pipeline(task, {"algorithm": algorithm}, X).fit(X, y)
    assert 0 <= score_pipeline(pipeline, X, y, "roc_auc") <= 1
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    np.testing.assert_array_equal(pipeline.predict(X), joblib.load(model_path).predict(X))


def test_provided_validation_not_in_refit_or_profile(tmp_path):
    task = make_task(tmp_path, validation_path=tmp_path / "valid.csv")
    pd.DataFrame({"id": ["101", "102"], "number": [1e9, 1e9], "category": ["leak", "leak"], "label": [0, 1]}).to_csv(task.validation_path, index=False)
    data = load_data(task)
    assert len(data.X_train) == len(data.X_refit) == 40
    assert data.profile["training_rows"] == 40
    assert data.X_refit["number"].max() == 39


@pytest.mark.parametrize("candidate", [
    {"algorithm": "random_forest", "params": {"n_estimators": 999999}},
    {"algorithm": "logistic_regression", "params": {"C": True}},
    {"algorithm": "logistic_regression", "params": {"C": float("nan")}},
    {"algorithm": "logistic_regression", "params": {"unknown": 1}},
    {"algorithm": "text_logistic"},
])
def test_invalid_candidates_rejected(tmp_path, candidate):
    task = make_task(tmp_path)
    with pytest.raises(ValueError):
        build_pipeline(task, candidate, load_data(task).X_train)


def test_manifest_paths_validation_and_supported_catalog(tmp_path):
    task = make_task(tmp_path)
    manifest = tmp_path / "task.json"
    config = {"data_type": "tabular", "train_path": "train.csv", "target_column": "label"}
    manifest.write_text(json.dumps(config), encoding="utf-8")
    assert load_task(manifest).train_path == task.train_path
    config["max_experiments"] = True
    manifest.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_task(manifest)
    config.pop("max_experiments")
    config["test_path"] = "secret.csv"
    manifest.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_task(manifest)
    assert len(candidate_catalog("tabular")) == 5
    with pytest.raises(ValueError):
        TaskSpec(data_type="image", train_path=task.train_path, target_column="label")
    with pytest.raises(ValueError):
        TaskSpec(data_type="tabular", train_path=task.train_path, predict_path=task.train_path, target_column="label")


def test_small_classes_cannot_silently_drop_stratification(tmp_path):
    task = make_task(tmp_path)
    frame = pd.read_csv(task.train_path).iloc[:3]
    frame.to_csv(task.train_path, index=False)
    with pytest.raises(ValueError, match="stratified"):
        load_data(task)


def test_unseen_validation_class_rejected(tmp_path):
    task = make_task(tmp_path, validation_path=tmp_path / "valid.csv")
    pd.DataFrame({"id": ["99", "98"], "number": [5, 6], "category": ["a", "a"], "label": [0, 2]}).to_csv(task.validation_path, index=False)
    with pytest.raises(ValueError, match="absent from training"):
        load_data(task)
