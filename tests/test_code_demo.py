"""Synthetic code-generation fixtures exercise the shared data protocol only."""

import json

import numpy as np
import pandas as pd
import pytest

from superagent.code_demo import make_code_demo
from superagent.data import image_source_paths, load_data, load_task
from superagent.utils import file_hash


@pytest.fixture
def code_demo(tmp_path):
    directory = tmp_path / "code_demo"
    paths = make_code_demo(directory)
    return directory, {path.parent.name: path for path in paths}


@pytest.mark.parametrize("kind,class_count", [("tabular", 2), ("text", 3), ("image", 3)])
def test_code_demo_tasks_follow_shared_data_protocol(code_demo, kind, class_count):
    directory, paths = code_demo
    assert set(paths) == {"tabular", "text", "image"}
    task_path = paths[kind]
    configuration = json.loads(task_path.read_text(encoding="utf-8"))
    assert "predict_path" not in configuration
    task = load_task(task_path)
    assert task.data_type == kind and task.predict_path is None
    data = load_data(task)
    assert len(data.X_train) == len(data.y_train) == len(data.X_refit) == 180
    assert len(data.X_valid) == len(data.y_valid) == 60
    assert set(data.y_train) == set(data.y_valid) == set(range(class_count))
    assert data.profile["class_count"] == class_count
    assert data.split_manifest["refit_uses"] == "training_file_only"
    assert task.target_column not in data.feature_columns
    assert list(data.X_train) == list(data.X_valid) == data.feature_columns
    prediction_example = pd.read_csv(task_path.parent / "predict.csv")
    assert len(prediction_example) == 15 and task.target_column not in prediction_example

    if kind == "tabular":
        assert len(data.feature_columns) == 2
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.X_train.dtypes)
    elif kind == "text":
        assert data.feature_columns == [task.text_column]
        assert all(isinstance(value, str) and value for value in data.X_train[task.text_column])
    else:
        train_paths = image_source_paths(task, task.train_path)
        valid_paths = image_source_paths(task, task.validation_path)
        assert len(set(train_paths)) == 180 and len(set(valid_paths)) == 60
        assert set(train_paths).isdisjoint(valid_paths)
        train_identities = {(path.stat().st_dev, path.stat().st_ino) for path in train_paths}
        valid_identities = {(path.stat().st_dev, path.stat().st_ino) for path in valid_paths}
        assert train_identities.isdisjoint(valid_identities)
        for frame in (data.X_train, data.X_valid):
            assert list(frame) == [task.image_column]
            assert all(isinstance(value, np.ndarray) and value.shape == (28, 28, 3)
                       and value.dtype == np.uint8 for value in frame[task.image_column])
        assert data.profile["image_shape"] == [28, 28, 3]

    purpose = json.loads((directory / "purpose.json").read_text(encoding="utf-8"))
    assert purpose["official_course_data_used"] is False
    assert purpose["scores_are_course_results"] is False
    assert set(purpose["tasks"]) == {str(path.relative_to(directory)) for path in paths.values()}


def test_code_demo_does_not_replace_an_existing_demo(code_demo):
    directory, _ = code_demo
    before = {path.relative_to(directory): file_hash(path) for path in directory.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="new or empty"):
        make_code_demo(directory)
    after = {path.relative_to(directory): file_hash(path) for path in directory.rglob("*") if path.is_file()}
    assert after == before


def test_code_demo_does_not_overwrite_unrelated_directory_contents(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("existing work", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        make_code_demo(directory)
    assert list(directory.iterdir()) == [marker]
    assert marker.read_text(encoding="utf-8") == "existing work"


def test_code_demo_accepts_an_existing_empty_directory(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    assert len(make_code_demo(directory)) == 3
