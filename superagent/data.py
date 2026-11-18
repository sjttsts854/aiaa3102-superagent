"""Task manifests and train-only data inspection for the first SuperAgent backend."""

from __future__ import annotations

import json
import math
import io
import warnings
from dataclasses import dataclass, fields
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from .utils import file_hash


@dataclass
class TaskSpec:
    data_type: str
    train_path: Path
    target_column: str
    predict_path: Path | None = None
    validation_path: Path | None = None
    id_column: str | None = None
    text_column: str | None = None
    metric: str = "accuracy"
    seed: int = 42
    validation_fraction: float = 0.2
    max_experiments: int = 4
    name: str = "classification"
    data_format: str = "csv"
    train_labels_path: Path | None = None
    validation_labels_path: Path | None = None
    feature_names: list[str] | None = None
    categorical_columns: list[str] | None = None
    text_columns: list[str] | None = None
    image_column: str | None = None
    image_size: tuple[int, int] | list[int] = (32, 32)

    def __post_init__(self) -> None:
        if self.data_type not in ("tabular", "text", "paired_text", "image"):
            raise ValueError("data_type must be tabular, text, paired_text, or image")
        if self.data_format not in ("csv", "npy"):
            raise ValueError("data_format must be csv or npy")
        for key in ("target_column", "name"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        for key in ("id_column", "text_column"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{key} must be null or a nonempty string")
        if self.id_column == self.target_column or self.text_column == self.target_column:
            raise ValueError("ID/text columns must differ from the target column")
        if self.data_type == "text" and (not self.text_column or self.text_column == self.id_column):
            raise ValueError("text tasks need a text_column distinct from the ID column")
        if self.data_type != "text" and self.text_column is not None:
            raise ValueError("text_column is only supported for text tasks")
        if (not isinstance(self.image_size, (list, tuple)) or len(self.image_size) != 2
                or any(type(v) is not int or not 8 <= v <= 256 for v in self.image_size)):
            raise ValueError("image_size must be [height, width] with integers between 8 and 256")
        self.image_size = tuple(self.image_size)
        if self.data_type == "image":
            if (not isinstance(self.image_column, str) or not self.image_column.strip()
                    or self.image_column in (self.target_column, self.id_column)):
                raise ValueError("image tasks require image_column distinct from target and ID")
            if self.data_format != "csv":
                raise ValueError("image tasks currently require a CSV image manifest")
        elif self.image_column is not None or self.image_size != (32, 32):
            raise ValueError("image_column and nondefault image_size require an image task")
        for key in ("feature_names", "categorical_columns", "text_columns"):
            value = getattr(self, key)
            if value is not None:
                if (not isinstance(value, list) or not value or
                        any(not isinstance(c, str) or not c.strip() for c in value) or
                        len(set(value)) != len(value)):
                    raise ValueError(f"{key} must be a nonempty list of distinct column names")
                if self.target_column in value:
                    raise ValueError(f"{key} must exclude the target column")
        if self.data_type == "paired_text":
            if self.text_columns is None or len(self.text_columns) != 2 or self.id_column in self.text_columns:
                raise ValueError("paired_text tasks need two ordered text_columns distinct from the ID column")
        elif self.text_columns is not None:
            raise ValueError("text_columns is only supported for paired_text tasks")
        if self.categorical_columns is not None:
            if self.data_type != "tabular" or self.id_column in self.categorical_columns:
                raise ValueError("categorical_columns must be tabular features distinct from the ID column")
        if self.metric not in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc"):
            raise ValueError("Unsupported metric; use accuracy, balanced_accuracy, f1_macro, or roc_auc")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        if type(self.max_experiments) is not int or not 1 <= self.max_experiments <= 100:
            raise ValueError("max_experiments must be an integer between 1 and 100")
        if type(self.validation_fraction) not in (int, float) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be a number strictly between 0 and 1")
        for key in ("train_path", "predict_path", "validation_path", "train_labels_path", "validation_labels_path"):
            value = getattr(self, key)
            if value is None and key != "train_path":
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{key} must be a nonempty file path")
            setattr(self, key, Path(value).resolve())
        if self.validation_path == self.train_path:
            raise ValueError("validation_path must differ from train_path")
        if self.predict_path is not None and self.predict_path in (self.train_path, self.validation_path):
            raise ValueError("predict_path must differ from train_path and validation_path")
        if self.data_format == "npy":
            if self.feature_names is None or self.train_labels_path is None:
                raise ValueError("npy tasks require feature_names and train_labels_path")
            if (self.validation_path is None) != (self.validation_labels_path is None):
                raise ValueError("npy validation features and labels must be provided together")
            if self.id_column is not None and self.id_column not in self.feature_names:
                raise ValueError("npy feature_names must include the declared ID column")
            feature_sources = {self.train_path, self.validation_path, self.predict_path} - {None}
            label_sources = [p for p in (self.train_labels_path, self.validation_labels_path) if p is not None]
            if len(set(label_sources)) != len(label_sources) or feature_sources.intersection(label_sources):
                raise ValueError("Feature and label files for each split must have distinct paths")
        elif any(v is not None for v in (self.train_labels_path, self.validation_labels_path, self.feature_names)):
            raise ValueError("Separate label paths and feature_names are only supported for npy tasks")


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_refit: pd.DataFrame
    y_refit: pd.Series
    feature_columns: list[str]
    profile: dict[str, Any]
    split_manifest: dict[str, Any]


def load_task(path: str | Path) -> TaskSpec:
    """Load strict JSON; all relative data paths are relative to its manifest."""
    manifest = Path(path).resolve()
    with manifest.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Task manifest must be a JSON object")
    extra = set(config) - {field.name for field in fields(TaskSpec)}
    if extra:
        raise ValueError(f"Unknown task fields: {sorted(extra)}")
    for key in ("data_type", "train_path", "target_column"):
        if key not in config:
            raise ValueError(f"Missing task field: {key}")
    for key in ("train_path", "predict_path", "validation_path", "train_labels_path", "validation_labels_path"):
        if key in config and config[key] is not None:
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f"{key} must be a nonempty string path")
            config[key] = (manifest.parent / config[key]).resolve()
    return TaskSpec(**config)


def _read_csv(path: Path, task: TaskSpec) -> pd.DataFrame:
    # Preserve leading zeroes in identifiers used by prediction result files.
    dtype = {task.id_column: "string"} if task.id_column else {}
    if task.data_type == "image":
        dtype[task.image_column] = "string"
    data = pd.read_csv(path, dtype=dtype)
    if data.empty:
        raise ValueError(f"CSV contains no data rows: {path}")
    if task.id_column and task.id_column not in data:
        raise ValueError(f"Missing ID column: {task.id_column}")
    return data


def _image_paths(frame: pd.DataFrame, task: TaskSpec, manifest_path: Path) -> list[Path]:
    """Resolve only files explicitly listed beneath this CSV's own directory."""
    if task.image_column not in frame:
        raise ValueError("Image manifest is missing image_column")
    base = manifest_path.resolve().parent
    paths = []
    for index, value in enumerate(frame[task.image_column]):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Image row {index} needs a nonempty relative file path")
        relative = Path(value)
        windows = PureWindowsPath(value)
        if (relative.is_absolute() or windows.is_absolute() or windows.drive
                or ":" in value or ".." in relative.parts or ".." in windows.parts):
            raise ValueError(f"Image row {index} must remain beneath its CSV directory")
        resolved = (base / relative).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"Image row {index} resolves outside its CSV directory")
        if not resolved.is_file():
            raise ValueError(f"Image row {index} does not identify a regular file")
        if resolved.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f"Image row {index} exceeds the 32 MiB file limit")
        paths.append(resolved)
    return paths


def image_source_paths(task: TaskSpec, input_path: Path) -> list[Path]:
    """Return confined asset identities for one explicit split, without decoding."""
    if task.data_type != "image":
        return []
    return _image_paths(_read_csv(Path(input_path), task), task, Path(input_path))


def image_source_hashes(task: TaskSpec, input_path: Path) -> dict[str, str]:
    """Hash explicitly referenced image files for source/evaluation manifests."""
    path = Path(input_path).resolve()
    return {item.relative_to(path.parent).as_posix(): file_hash(item)
            for item in image_source_paths(task, path)}


def _decode_images(frame: pd.DataFrame, task: TaskSpec, path: Path) -> pd.DataFrame:
    from PIL import Image, ImageOps, UnidentifiedImageError

    pixels = []
    height, width = task.image_size
    if len(frame) * height * width * 3 > 512 * 1024 * 1024:
        raise ValueError("Resized image split exceeds 512 MiB; reduce image_size or split size")
    for index, asset in enumerate(_image_paths(frame, task, path)):
        # Decode a private byte buffer; image codecs never receive an external URI.
        raw = asset.read_bytes()
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError(f"Image row {index} exceeds the 32 MiB file limit")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > 20_000_000:
                        raise ValueError("Decoded image exceeds the 20 million pixel limit")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Animated or multi-frame images are not supported")
                    converted = ImageOps.exif_transpose(image).convert("RGB")
                    resized = converted.resize((width, height), Image.Resampling.BILINEAR)
                    pixels.append(np.array(resized, dtype=np.uint8, copy=True))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
                Image.DecompressionBombError, ValueError) as exc:
            raise ValueError(f"Image row {index} cannot be decoded as a bounded static image: {type(exc).__name__}") from exc
    result = frame.copy()
    values = np.empty(len(pixels), dtype=object)
    values[:] = pixels
    result[task.image_column] = values
    return result


def _read_frame(path: Path, task: TaskSpec, labels_path: Path | None = None) -> pd.DataFrame:
    """Read one explicitly requested split; never infer paths for other splits.

    Object NPY arrays require pickle support. Only use manifests pointing to
    trusted local datasets, such as the course-provided feature arrays.
    """
    if task.data_format == "csv":
        frame = _read_csv(path, task)
        if task.data_type == "image":
            frame = _decode_images(frame, task, path)
    else:
        values = np.load(path, allow_pickle=True)
        if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("NPY features must be a nonempty two-dimensional array")
        if values.shape[1] != len(task.feature_names):
            raise ValueError("NPY feature width must match feature_names")
        frame = pd.DataFrame(values, columns=task.feature_names)
        if task.id_column:
            frame[task.id_column] = frame[task.id_column].astype("string")
        if labels_path is not None:
            labels = np.load(labels_path, allow_pickle=True)
            if not isinstance(labels, np.ndarray) or labels.ndim != 1:
                raise ValueError("NPY labels must be a one-dimensional array")
            if len(labels) != len(frame):
                raise ValueError("NPY feature and label row counts must match")
            # Infer scalar numeric label types even when the array uses object.
            frame[task.target_column] = pd.Series(labels.tolist())
    if task.categorical_columns:
        missing = set(task.categorical_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing categorical columns: {sorted(missing)}")
        for column in task.categorical_columns:
            frame[column] = frame[column].astype(object)
    return frame


def _target(frame: pd.DataFrame, task: TaskSpec) -> pd.Series:
    if task.target_column not in frame:
        raise ValueError(f"Missing target column: {task.target_column}")
    target = frame[task.target_column].copy()
    if target.isna().any():
        raise ValueError("Target labels must not be missing")
    if type_of_target(target) not in ("binary", "multiclass"):
        raise ValueError("Target must contain binary or multiclass classification labels")
    return target


def _features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return frame.loc[:, columns].copy()


def _profile(task: TaskSpec, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    # Do not expose dataset names, paths, column names, label identities, or rows.
    numeric_count = sum(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
    result: dict[str, Any] = {
        "data_type": task.data_type,
        "metric": task.metric,
        "training_rows": int(len(X)),
        "feature_count": int(len(X.columns)),
        "numeric_feature_count": int(numeric_count),
        "categorical_feature_count": int(len(X.columns) - numeric_count),
        "missing_fraction": float(X.isna().to_numpy().mean()),
        "class_count": int(y.nunique()),
        "class_counts_descending": sorted([int(n) for n in y.value_counts()], reverse=True),
        "class_imbalance_ratio": float(y.value_counts().max() / y.value_counts().min()),
    }
    if task.data_type == "text":
        text = X[task.text_column].fillna("").astype(str)
        result["mean_text_length"] = float(text.str.len().mean())
        result["empty_text_fraction"] = float(text.str.strip().eq("").mean())
    elif task.data_type == "paired_text":
        texts = [X[column].fillna("").astype(str) for column in task.text_columns]
        result["mean_text_lengths"] = [float(text.str.len().mean()) for text in texts]
        result["empty_text_fractions"] = [float(text.str.strip().eq("").mean()) for text in texts]
    elif task.data_type == "image":
        result.update(categorical_feature_count=0, image_feature_count=1,
                      image_shape=[*task.image_size, 3], n_channels=3,
                      image_dtype="uint8", image_pixel_range=[0, 255])
    return result


def load_data(task: TaskSpec) -> DataBundle:
    """Read training/validation only. Prediction data is intentionally deferred."""
    frame = _read_frame(task.train_path, task, task.train_labels_path)
    y = _target(frame, task)
    if y.nunique() < 2:
        raise ValueError("Training data must contain at least two classes")
    if task.metric == "roc_auc" and y.nunique() != 2:
        raise ValueError("The initial roc_auc implementation supports binary classification only")
    if task.data_type == "paired_text" and y.nunique() != 2:
        raise ValueError("paired_text requires exactly two target classes")
    if task.data_type == "text":
        columns = [task.text_column]
    elif task.data_type == "paired_text":
        columns = list(task.text_columns)
    elif task.data_type == "image":
        columns = [task.image_column]
    else:
        columns = [c for c in frame if c not in (task.target_column, task.id_column)]
    if not columns:
        raise ValueError("No feature columns remain after excluding the label and ID")
    X = _features(frame, columns)
    if task.validation_path is not None:
        valid = _read_frame(task.validation_path, task, task.validation_labels_path)
        y_valid = _target(valid, task)
        unseen = set(y_valid.unique()) - set(y.unique())
        if unseen:
            raise ValueError("Validation labels contain classes absent from training")
        if set(y_valid.unique()) != set(y.unique()):
            raise ValueError("Validation data must contain every training class for comparable metrics")
        X_train, y_train = X.copy(), y.copy()
        X_valid = _features(valid, columns)
        manifest = {
            "strategy": "provided_validation",
            "train_source": str(task.train_path),
            "validation_source": str(task.validation_path),
            "train_indices": list(range(len(X))),
            "validation_indices": list(range(len(X_valid))),
            "refit_uses": "training_file_only",
        }
    else:
        classes = y.nunique()
        n_valid = math.ceil(len(y) * task.validation_fraction)
        if y.value_counts().min() < 2 or min(n_valid, len(y) - n_valid) < classes:
            raise ValueError("Not enough examples per class for a stratified split; provide more data or a validation file")
        train_idx, valid_idx = train_test_split(
            np.arange(len(X)), test_size=task.validation_fraction, stratify=y,
            random_state=task.seed,
        )
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()
        if set(y_train.unique()) != set(y_valid.unique()):
            raise ValueError("Stratified split omitted a class; adjust the fraction or provide a validation file")
        manifest = {
            "strategy": "stratified_holdout",
            "seed": task.seed,
            "train_source": str(task.train_path),
            "validation_source": str(task.train_path),
            "train_indices": train_idx.tolist(),
            "validation_indices": valid_idx.tolist(),
            "refit_uses": "full_training_file_after_selection",
        }
    if task.data_format == "npy":
        manifest["train_labels_source"] = str(task.train_labels_path)
        manifest["validation_labels_source"] = str(task.validation_labels_path or task.train_labels_path)
    if task.data_type == "image":
        assets = image_source_paths(task, task.train_path)
        if task.validation_path is not None:
            train_assets, valid_assets = set(assets), set(image_source_paths(task, task.validation_path))
        else:
            train_assets = {assets[index] for index in manifest["train_indices"]}
            valid_assets = {assets[index] for index in manifest["validation_indices"]}
        if train_assets & valid_assets:
            raise ValueError("Training and validation splits must not reference the same image asset")
        manifest["image_assets_sha256"] = {"train": image_source_hashes(task, task.train_path)}
        if task.validation_path is not None:
            manifest["image_assets_sha256"]["validation"] = image_source_hashes(task, task.validation_path)
    return DataBundle(
        X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
        X_refit=X, y_refit=y, feature_columns=columns,
        profile=_profile(task, X_train, y_train), split_manifest=manifest,
    )


def load_prediction_data(task: TaskSpec, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Call only after final model selection. Any target column is ignored."""
    if task.predict_path is None:
        raise ValueError("Task has no predict_path")
    frame = _read_frame(task.predict_path, task)
    ids = frame[task.id_column].copy() if task.id_column else None
    return _features(frame, feature_columns), ids
