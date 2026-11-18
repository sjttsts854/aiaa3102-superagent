import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from superagent.data import TaskSpec
from superagent.evaluation import evaluate_model
from superagent.models import build_pipeline
from superagent.utils import file_hash


def _bundle(tmp_path, data_format="csv"):
    run = tmp_path / "run"
    run.mkdir()
    feature_frame = pd.DataFrame({"number": [0, 1] * 12, "category": ["a", "b"] * 12})
    y = np.array([0, 1] * 12)
    train = tmp_path / f"train.{data_format}"
    valid = tmp_path / f"valid.{data_format}"
    options = {}
    if data_format == "npy":
        np.save(train, feature_frame.to_numpy())
        np.save(valid, feature_frame.to_numpy()[:4])
        train_labels = tmp_path / "train_labels.npy"
        valid_labels = tmp_path / "valid_labels.npy"
        np.save(train_labels, y)
        np.save(valid_labels, y[:4])
        options = {"data_format": "npy", "train_labels_path": train_labels,
                   "validation_labels_path": valid_labels,
                   "feature_names": ["number", "category"],
                   "categorical_columns": ["number", "category"]}
        feature_frame = feature_frame.astype(object)
    else:
        training_frame = feature_frame.assign(label=y, id=[f"{i:03}" for i in range(len(y))])
        training_frame.to_csv(train, index=False)
        training_frame.iloc[:4].to_csv(valid, index=False)
        options["id_column"] = "id"
    task = TaskSpec(data_type="tabular", train_path=train, validation_path=valid,
                    target_column="label", **options)
    model = build_pipeline(task, {"algorithm": "logistic_regression"}, feature_frame).fit(feature_frame, y)
    selection = {"trial": 1, "candidate": {"algorithm": "logistic_regression", "params": {}},
                 "status": "success", "score": 1.0}
    (run / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    model_path = run / "model.joblib"
    joblib.dump({"pipeline": model, "task": task, "feature_columns": list(feature_frame),
                 "selection": selection, "source_sha256": "test-training-source"}, model_path)
    test = tmp_path / f"test.{data_format}"
    labels_path = None
    if data_format == "npy":
        np.save(test, feature_frame.iloc[:4].to_numpy())
        labels_path = tmp_path / "test_labels.npy"
        # Object scalar labels also work, matching the general data adapter.
        np.save(labels_path, y[:4].astype(object))
    else:
        feature_frame.iloc[:4].assign(label=y[:4], id=["001", "002", "003", "004"]).to_csv(test, index=False)
    return model_path, test, labels_path, task


@pytest.mark.parametrize("data_format", ["csv", "npy"])
def test_final_evaluation_scores_existing_model_without_fit(tmp_path, monkeypatch, data_format):
    model_path, test, labels, task = _bundle(tmp_path, data_format)
    selection_before = (model_path.parent / "selection.json").read_bytes()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Final evaluation must never fit")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    original_predict = Pipeline.predict

    def assert_features_only(self, X, *args, **kwargs):
        assert list(X) == ["number", "category"]
        assert "label" not in X
        return original_predict(self, X, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "predict", assert_features_only)
    output = tmp_path / "evaluation"
    result = evaluate_model(model_path, test, output, labels)
    assert result["status"] == "completed"
    assert result["metric"] == "accuracy" and result["test_accuracy"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["class_support"] == [{"label": 0, "rows": 2}, {"label": 1, "rows": 2}]
    assert result["rows"] == 4 and result["training_performed"] is False
    assert result["planner_calls"] == 0
    assert result["model_sha256"] == file_hash(model_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_source_sha256"] == "test-training-source"
    assert len(manifest["evaluation_source"]["source_sha256"]) == 64
    assert manifest["data_sha256"]["features"] == file_hash(test)
    assert manifest["run_selection_file_sha256"] == file_hash(model_path.parent / "selection.json")
    assert (model_path.parent / "selection.json").read_bytes() == selection_before
    assert not (model_path.parent / "summary.json").exists()
    predictions = pd.read_csv(result["predictions"], dtype={"id": str})
    assert predictions[task.target_column].tolist() == [0, 1, 0, 1]
    if data_format == "csv":
        assert predictions.id.tolist() == ["001", "002", "003", "004"]
    probability = pd.read_csv(result["probabilities"])
    np.testing.assert_allclose(probability[["probability__0", "probability__1"]].sum(axis=1), 1)
    with pytest.raises(ValueError, match="already exists"):
        evaluate_model(model_path, test, output, labels)


def test_csv_test_label_changes_cannot_change_predictions(tmp_path):
    model, test, _, _ = _bundle(tmp_path)
    first = evaluate_model(model, test, tmp_path / "score1")
    frame = pd.read_csv(test, dtype={"id": str})
    frame["label"] = 1 - frame["label"]
    changed = tmp_path / "changed_labels.csv"
    frame.to_csv(changed, index=False)
    second = evaluate_model(model, changed, tmp_path / "score2")
    assert first["test_accuracy"] == 1.0 and second["test_accuracy"] == 0.0
    assert (tmp_path / "score1" / "predictions.csv").read_bytes() == (tmp_path / "score2" / "predictions.csv").read_bytes()


@pytest.mark.parametrize("key", ["train_path", "validation_path", "train_labels_path", "validation_labels_path"])
def test_evaluation_rejects_development_file_overlap(tmp_path, key):
    model, test, labels, task = _bundle(tmp_path, "npy")
    forbidden_path = getattr(task, key)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_model(model, forbidden_path, tmp_path / "eval", labels)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_model(model, test, tmp_path / "eval", forbidden_path)
    assert not (tmp_path / "eval").exists()


@pytest.mark.parametrize("problem", ["missing", "unknown", "absent"])
def test_csv_invalid_labels_fail_without_publishing_predictions(tmp_path, problem):
    model, test, _, _ = _bundle(tmp_path)
    frame = pd.read_csv(test, dtype={"id": str})
    if problem == "absent":
        frame = frame.drop(columns="label")
    else:
        frame.loc[0, "label"] = np.nan if problem == "missing" else 99
    frame.to_csv(test, index=False)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        evaluate_model(model, test, output)
    assert not (output / "predictions.csv").exists()
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("values", [np.array([0, 1]), np.array([[0], [1], [0], [1]])])
def test_npy_labels_require_matching_rows_and_one_dimension(tmp_path, values):
    model, test, labels, _ = _bundle(tmp_path, "npy")
    np.save(labels, values)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="rows|one-dimensional"):
        evaluate_model(model, test, output, labels)
    assert not (output / "predictions.csv").exists()


def test_missing_npy_labels_and_unneeded_csv_labels_rejected(tmp_path):
    npy_dir = tmp_path / "npy"
    npy_dir.mkdir()
    model, test, labels, _ = _bundle(npy_dir, "npy")
    with pytest.raises(ValueError, match="requires"):
        evaluate_model(model, test, npy_dir / "evaluation")
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv_model, csv_test, _, _ = _bundle(csv_dir)
    with pytest.raises(ValueError, match="do not pass"):
        evaluate_model(csv_model, csv_test, csv_dir / "evaluation", labels)


def test_existing_empty_evaluation_directory_is_not_reused(tmp_path):
    model, test, _, _ = _bundle(tmp_path)
    output = tmp_path / "evaluation"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        evaluate_model(model, test, output)
