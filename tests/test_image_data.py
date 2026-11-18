import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.base import clone

from superagent.data import TaskSpec, image_source_hashes, image_source_paths, load_data, load_prediction_data, load_task
from superagent.image_features import ImageGridFeatures
from superagent.models import build_pipeline, candidate_catalog, score_pipeline


def make_image_task(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(27)
    rows = []
    for i in range(36):
        label = i % 3
        pixels = rng.integers(0, 20, (17 + i % 3, 19, 3), dtype=np.uint8)
        pixels[:, :, label] = rng.integers(210, 255, pixels.shape[:2], dtype=np.uint8)
        name = f"images/frame_{i:03}.png"
        Image.fromarray(pixels).save(tmp_path / name)
        rows.append({"id": f"{i:03}", "picture": name, "label": label})
    pd.DataFrame(rows[:24]).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(rows[24:30]).to_csv(tmp_path / "valid.csv", index=False)
    # Only later prediction should read these; its labels are intentionally bogus.
    pd.DataFrame(rows[30:]).assign(label="ignored").to_csv(tmp_path / "predict.csv", index=False)
    return TaskSpec(data_type="image", train_path=tmp_path / "train.csv",
                    validation_path=tmp_path / "valid.csv", predict_path=tmp_path / "predict.csv",
                    target_column="label", id_column="id", image_column="picture", image_size=[16, 24])


def test_image_multiclass_pipeline_reloads_and_predicts_without_labels(tmp_path):
    task = make_image_task(tmp_path)
    data = load_data(task)
    assert len(data.X_train) == len(data.X_refit) == 24
    assert data.feature_columns == ["picture"]
    assert data.X_train.iloc[0, 0].shape == (16, 24, 3)
    assert data.X_train.iloc[0, 0].dtype == np.uint8
    assert set(data.y_train) == {0, 1, 2}
    assert data.profile["image_shape"] == [16, 24, 3]
    assert data.profile["n_channels"] == 3
    assert data.profile["class_count"] == 3
    assert data.profile["categorical_feature_count"] == 0
    assert data.profile["missing_fraction"] == 0
    profile = json.dumps(data.profile)
    assert "picture" not in profile and "frame_" not in profile and str(tmp_path) not in profile
    assert set(data.split_manifest["image_assets_sha256"]) == {"train", "validation"}
    assert len(data.split_manifest["image_assets_sha256"]["train"]) == 24
    assert "image_logistic" in candidate_catalog("image")
    model = build_pipeline(task, {"algorithm": "image_logistic", "params": {}}, data.X_train)
    model.fit(data.X_train, data.y_train)
    assert score_pipeline(model, data.X_valid, data.y_valid, "accuracy") == 1.0
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    X, ids = load_prediction_data(task, data.feature_columns)
    assert ids.tolist() == ["030", "031", "032", "033", "034", "035"]
    assert list(X.columns) == ["picture"]
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    np.testing.assert_array_equal(loaded.predict(X), [0, 1, 2, 0, 1, 2])
    np.testing.assert_allclose(loaded.predict_proba(X).sum(axis=1), 1)
    assert clone(model).get_params()["image_features__grid_size"] == 8


def test_image_prediction_is_deferred_and_split_manifest_is_explicit(tmp_path):
    task = make_image_task(tmp_path)
    task = replace(task, predict_path=tmp_path / "does_not_exist.csv")
    data = load_data(task)
    assert data.split_manifest["refit_uses"] == "training_file_only"
    assert data.profile["training_rows"] == 24
    with pytest.raises(FileNotFoundError):
        load_prediction_data(task, data.feature_columns)
    internal = load_data(replace(task, validation_path=None))
    assert internal.split_manifest["strategy"] == "stratified_holdout"
    assert len(internal.X_refit) == 24
    assert set(internal.split_manifest["train_indices"]).isdisjoint(internal.split_manifest["validation_indices"])


def test_image_fingerprints_track_asset_bytes_and_not_unrequested_splits(tmp_path):
    task = make_image_task(tmp_path)
    before = image_source_hashes(task, task.train_path)
    assert len(before) == 24 and all(key.startswith("images/") for key in before)
    changed = tmp_path / "images/frame_000.png"
    Image.new("RGB", (20, 20), (123, 45, 67)).save(changed)
    after = image_source_hashes(task, task.train_path)
    assert [key for key in before if before[key] != after[key]] == ["images/frame_000.png"]
    assert image_source_paths(task, task.train_path)[0] == changed.resolve()
    (tmp_path / "images/frame_035.png").unlink()
    assert image_source_hashes(task, task.train_path) == after
    assert len(load_data(task).X_train) == 24
    with pytest.raises(ValueError, match="regular file"):
        image_source_hashes(task, task.predict_path)


@pytest.mark.parametrize("value", ["../outside.png", "images/../../outside.png", "C:/outside.png",
                                  "\\\\server\\share\\outside.png", "https://example.com/image.png",
                                  "images/missing.png", ""])
def test_image_manifest_rejects_outside_missing_and_remote_paths(tmp_path, value):
    task = make_image_task(tmp_path)
    frame = pd.read_csv(task.train_path)
    frame.loc[0, "picture"] = value
    frame.to_csv(task.train_path, index=False)
    with pytest.raises(ValueError, match="Image row"):
        load_data(task)


def test_image_symlink_cannot_escape_manifest_directory(tmp_path):
    task = make_image_task(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside.png"
    Image.new("RGB", (8, 8)).save(outside)
    link = tmp_path / "images/link.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    frame = pd.read_csv(task.train_path)
    frame.loc[0, "picture"] = "images/link.png"
    frame.to_csv(task.train_path, index=False)
    with pytest.raises(ValueError, match="outside"):
        load_data(task)


def test_corrupt_and_multiframe_images_are_rejected(tmp_path):
    task = make_image_task(tmp_path)
    asset = tmp_path / "images/frame_000.png"
    asset.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be decoded"):
        load_data(task)
    Image.new("RGB", (8, 8), "red").save(asset, format="GIF", save_all=True,
                                         append_images=[Image.new("RGB", (8, 8), "blue")])
    with pytest.raises(ValueError, match="cannot be decoded"):
        load_data(task)


def test_large_image_file_is_rejected_before_decoding(tmp_path):
    task = make_image_task(tmp_path)
    with (tmp_path / "images/frame_000.png").open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        load_data(task)


def test_large_resized_split_is_rejected_before_allocating_pixels(tmp_path):
    task = make_image_task(tmp_path)
    frame = pd.read_csv(task.train_path)
    pd.concat([frame] * 125, ignore_index=True).to_csv(task.train_path, index=False)
    with pytest.raises(ValueError, match="512 MiB"):
        load_data(replace(task, image_size=[256, 256]))


def test_same_asset_cannot_appear_in_train_and_validation(tmp_path):
    task = make_image_task(tmp_path)
    valid = pd.read_csv(task.validation_path)
    valid.loc[0, "picture"] = "images/frame_000.png"
    valid.to_csv(task.validation_path, index=False)
    with pytest.raises(ValueError, match="same image asset"):
        load_data(task)


@pytest.mark.parametrize("updates", [
    {"image_column": None}, {"image_column": "label"}, {"image_column": "id"},
    {"image_size": [0, 32]}, {"image_size": [32, True]}, {"image_size": [32]},
    {"image_size": [32, 1000]}, {"data_format": "npy"},
])
def test_image_task_schema_rejects_invalid_contract(tmp_path, updates):
    task = make_image_task(tmp_path)
    with pytest.raises(ValueError):
        replace(task, **updates)


def test_manifest_roundtrip_and_image_features_validate_shape(tmp_path):
    task = make_image_task(tmp_path)
    config = {"data_type": "image", "train_path": "train.csv", "validation_path": "valid.csv",
              "target_column": "label", "image_column": "picture", "image_size": [16, 24]}
    path = tmp_path / "task.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_task(path)
    assert loaded.image_size == (16, 24)
    data = load_data(loaded)
    transformer = ImageGridFeatures(grid_size=4).fit(data.X_train)
    transformed = transformer.transform(data.X_valid)
    assert transformed.shape == (6, 4 * 4 * 3 + 6)
    assert np.isfinite(transformed).all()
    broken = data.X_valid.copy()
    broken.iloc[0, 0] = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape|dimensions"):
        transformer.transform(broken)
    with pytest.raises(ValueError, match="grid_size"):
        ImageGridFeatures(grid_size=17).fit(data.X_train)


def test_nonimage_task_fingerprints_and_legacy_defaults(tmp_path):
    task = TaskSpec(data_type="tabular", train_path=tmp_path / "missing.csv", target_column="label")
    del task.image_column
    del task.image_size
    assert task.image_column is None and task.image_size == (32, 32)
    assert image_source_hashes(task, task.train_path) == {}
