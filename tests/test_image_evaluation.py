"""Final image scoring uses synthetic fixtures only, never official test data."""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.pipeline import Pipeline

from superagent.data import TaskSpec, image_source_hashes, load_data
from superagent.evaluation import evaluate_model
from superagent.models import build_pipeline
from superagent.utils import file_hash, object_hash


def _image_bundle(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(30):
        label = index % 3
        pixels = np.full((12, 12, 3), index, dtype=np.uint8)
        pixels[:, :, label] = 230
        relative_path = f"images/picture_{index:03}.png"
        Image.fromarray(pixels).save(tmp_path / relative_path)
        rows.append({"id": f"{index:03}", "picture": relative_path, "label": label})
    train, valid, test = (tmp_path / name for name in ("train.csv", "valid.csv", "test.csv"))
    pd.DataFrame(rows[:18]).to_csv(train, index=False)
    pd.DataFrame(rows[18:24]).to_csv(valid, index=False)
    pd.DataFrame(rows[24:]).to_csv(test, index=False)
    task = TaskSpec(data_type="image", train_path=train, validation_path=valid,
                    target_column="label", id_column="id", image_column="picture", image_size=(12, 12))
    data = load_data(task)
    candidate = {"algorithm": "image_logistic", "params": {"grid_size": 4}}
    pipeline = build_pipeline(task, candidate, data.X_train).fit(data.X_train, data.y_train)
    selection = {"trial": 1, "candidate": candidate, "status": "success", "score": 1.0}
    model = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "task": task, "feature_columns": data.feature_columns,
                 "selection": selection, "source_sha256": "synthetic-training-source"}, model)
    return model, test, task


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_final_image_evaluation_rejects_shared_assets_before_creating_output(tmp_path, split, hardlink):
    model, test, task = _image_bundle(tmp_path)
    development = pd.read_csv(getattr(task, f"{split}_path"))
    relative_path = development.loc[0, "picture"]
    original = tmp_path / relative_path
    if hardlink:
        linked = tmp_path / "images" / "test_hardlink.png"
        try:
            os.link(original, linked)
        except OSError as exc:
            pytest.skip(f"Hardlinks are unavailable on this filesystem: {exc}")
        assert original.resolve() != linked.resolve()
        assert original.samefile(linked)
        relative_path = linked.relative_to(tmp_path).as_posix()
    frame = pd.read_csv(test, dtype={"id": str})
    frame.loc[0, "picture"] = relative_path
    frame.to_csv(test, index=False)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="test image assets must not overlap"):
        evaluate_model(model, test, output)
    assert not output.exists()


def test_final_image_evaluation_records_asset_fingerprints_without_training(tmp_path, monkeypatch):
    model, test, task = _image_bundle(tmp_path)
    before = image_source_hashes(task, test)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Final image evaluation must not train")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    original_predict = Pipeline.predict

    def assert_pixels_only(self, X, *args, **kwargs):
        assert list(X) == ["picture"]
        assert all(isinstance(value, np.ndarray) and value.shape == (12, 12, 3)
                   and value.dtype == np.uint8 for value in X["picture"])
        return original_predict(self, X, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "predict", assert_pixels_only)
    output = tmp_path / "evaluation"
    result = evaluate_model(model, test, output)
    assert result["status"] == "completed" and result["test_accuracy"] == 1.0
    assert result["training_performed"] is False and result["planner_calls"] == 0
    assert result["data_sha256"] == {"features": file_hash(test), "image_assets": object_hash(before)}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_assets_sha256"] == before
    assert manifest["data_sha256"] == result["data_sha256"]
    assert image_source_hashes(task, test) == before
    predictions = pd.read_csv(result["predictions"], dtype={"id": str})
    assert predictions["id"].tolist() == ["024", "025", "026", "027", "028", "029"]
    assert predictions["label"].tolist() == [0, 1, 2, 0, 1, 2]


def test_final_image_evaluation_requires_development_assets_for_overlap_check(tmp_path):
    model, test, task = _image_bundle(tmp_path)
    task.validation_path.unlink()
    output = tmp_path / "evaluation"
    with pytest.raises(FileNotFoundError):
        evaluate_model(model, test, output)
    assert not output.exists()
