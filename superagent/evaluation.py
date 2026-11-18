"""Isolated final scoring of an already selected, fitted prediction bundle.

This module never trains a model, calls a planner, or changes development logs.
Its output belongs to a separate, new directory. Test scores must not be used to
choose another candidate or continue tuning the evaluated solution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import type_of_target

from .data import TaskSpec, image_source_hashes, image_source_paths, load_prediction_data
from .utils import file_hash, now, object_hash, write_json


def _same_file(left: Path, right: Path) -> bool:
    return left == right or (left.exists() and right.exists() and left.samefile(right))


def _reject_image_asset_overlap(task: TaskSpec, input_path: Path) -> None:
    """Check file identity, including hardlinks, before publishing any results."""
    if task.data_type != "image":
        return
    development_paths = set()
    for manifest in (task.train_path, task.validation_path):
        if manifest is not None:
            development_paths.update(image_source_paths(task, manifest))
    development_ids = set()
    unknown_identity = []
    for path in development_paths:
        info = path.stat()
        if info.st_ino:
            development_ids.add((info.st_dev, info.st_ino))
        else:
            unknown_identity.append(path)
    for path in set(image_source_paths(task, input_path)):
        info = path.stat()
        identity = (info.st_dev, info.st_ino) if info.st_ino else None
        # Stat identities avoid a quadratic comparison on large image manifests.
        # Use samefile where a filesystem does not report a usable inode.
        fallback = unknown_identity if identity is not None else development_paths
        if (path in development_paths or identity in development_ids
                or any(_same_file(path, other) for other in fallback)):
            raise ValueError("Final test image assets must not overlap training or validation image assets")


def _runtime_source() -> dict:
    project = Path(__file__).resolve().parent.parent
    paths = sorted((project / "superagent").glob("*.py")) + [project / "requirements.txt"]
    hashes = {path.relative_to(project).as_posix(): file_hash(path) for path in paths if path.is_file()}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"source_sha256": fingerprint, "files": hashes}


def _read_labels(task: TaskSpec, input_path: Path, labels_path: Path | None) -> pd.Series:
    if getattr(task, "data_format", "csv") == "npy":
        if labels_path is None:
            raise ValueError("NPY final evaluation requires a separate labels_path")
        values = np.load(labels_path, allow_pickle=True)
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError("Test labels must be a one-dimensional array")
        labels = pd.Series(values.tolist(), name=task.target_column)
    else:
        if labels_path is not None:
            raise ValueError("CSV final evaluation reads labels from target_column; do not pass labels_path")
        try:
            labels = pd.read_csv(input_path, usecols=[task.target_column])[task.target_column]
        except ValueError as exc:
            raise ValueError("Evaluation CSV must contain the target column") from exc
    if labels.empty or labels.isna().any():
        raise ValueError("Test labels must be nonempty and must not be missing")
    if type_of_target(labels) not in ("binary", "multiclass"):
        raise ValueError("Test labels must be binary or multiclass classification labels")
    return labels


def evaluate_model(
    model_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    labels_path: str | Path | None = None,
) -> dict:
    """Predict and score once into a new output directory, with full provenance.

    Only locally generated, trusted joblib bundles and NPY arrays should be loaded.
    NPY labels must be a one-dimensional array in feature-row order. CSV labels
    are read from the bundle's target column and are never supplied to the model.
    The function records both training-source and evaluation-source fingerprints;
    callers must freeze a single final version before producing reported results.
    Image evaluation also needs the original development CSVs and assets available
    to reject file-identity overlap before creating the result directory.
    """
    model_path = Path(model_path).resolve()
    input_path = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    labels_path = Path(labels_path).resolve() if labels_path is not None else None
    if output.exists():
        raise ValueError("Evaluation output directory already exists; choose a new directory")
    for path in (model_path, input_path, labels_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if _same_file(model_path, input_path) or (labels_path is not None and any(
            _same_file(labels_path, path) for path in (input_path, model_path))):
        raise ValueError("Model, test feature, and separate test label files must differ")

    model_sha256 = file_hash(model_path)
    bundle = joblib.load(model_path)
    required = {"pipeline", "task", "feature_columns", "selection", "source_sha256"}
    if not isinstance(bundle, dict) or not required.issubset(bundle) or not isinstance(bundle["task"], TaskSpec):
        raise ValueError("Expected a selected SuperAgent model bundle")
    if not isinstance(bundle["selection"], dict) or not bundle["selection"]:
        raise ValueError("Model bundle has no recorded final selection")
    task = bundle["task"]
    forbidden = [Path(path).resolve() for key in (
        "train_path", "validation_path", "train_labels_path", "validation_labels_path"
    ) if (path := getattr(task, key, None)) is not None]
    for path in (input_path, labels_path):
        if path is not None and any(_same_file(path, other) for other in forbidden):
            raise ValueError("Final test files must not overlap training or validation files")
    data_format = getattr(task, "data_format", "csv")
    if data_format == "npy" and labels_path is None:
        raise ValueError("NPY final evaluation requires a separate labels_path")
    if data_format == "csv" and labels_path is not None:
        raise ValueError("CSV final evaluation reads labels from target_column; do not pass labels_path")
    selected_task = replace(task, predict_path=input_path)
    model = bundle["pipeline"]
    if not hasattr(model, "predict") or not hasattr(model, "classes_"):
        raise ValueError("Model bundle must contain a fitted classifier")
    _reject_image_asset_overlap(task, input_path)

    output.mkdir(parents=True, exist_ok=False)
    data_hashes = {"features": file_hash(input_path)}
    if labels_path is not None:
        data_hashes["labels"] = file_hash(labels_path)
    image_assets = None
    if task.data_type == "image":
        image_assets = image_source_hashes(task, input_path)
        data_hashes["image_assets"] = object_hash(image_assets)
    selection_path = model_path.parent / "selection.json"
    selection_record = {
        "selection": bundle["selection"],
        "selection_sha256": hashlib.sha256(json.dumps(bundle["selection"], sort_keys=True, default=str).encode()).hexdigest(),
        "run_selection_path": str(selection_path) if selection_path.is_file() else None,
        "run_selection_file_sha256": file_hash(selection_path) if selection_path.is_file() else None,
    }
    manifest = {
        "created_at": now(),
        "model": str(model_path), "model_sha256": model_sha256,
        "input": str(input_path), "labels": str(labels_path) if labels_path else None,
        "data_sha256": data_hashes,
        "image_assets_sha256": image_assets,
        "training_source_sha256": bundle["source_sha256"],
        "evaluation_source": _runtime_source(),
        **selection_record,
        "policy": "Final evaluation only. No training, planner calls, or test-based selection. Do not tune after viewing these results.",
    }
    write_json(output / "manifest.json", manifest)
    try:
        # All preprocessing is already fitted. No labels are passed to predict.
        X, ids = load_prediction_data(selected_task, bundle["feature_columns"])
        prediction = np.asarray(model.predict(X))
        classes = np.asarray(model.classes_)
        if prediction.ndim != 1 or len(prediction) != len(X):
            raise ValueError("Model returned an invalid prediction shape")
        if not set(pd.unique(prediction)).issubset(set(classes)):
            raise ValueError("Model returned unknown class labels")
        labels = _read_labels(selected_task, input_path, labels_path)
        if len(labels) != len(prediction):
            raise ValueError("Test features and labels must have the same number of rows")
        if not set(pd.unique(labels)).issubset(set(classes)):
            raise ValueError("Test labels contain classes absent from the fitted model")

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X))
            if (probabilities.shape != (len(X), len(classes)) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(axis=1), 1)):
                raise ValueError("Model returned invalid class probabilities")
        # Detect files changed during scoring instead of silently mixing versions.
        if file_hash(model_path) != model_sha256 or file_hash(input_path) != data_hashes["features"]:
            raise ValueError("Model or test data changed during evaluation")
        if labels_path is not None and file_hash(labels_path) != data_hashes["labels"]:
            raise ValueError("Test labels changed during evaluation")
        if image_assets is not None and image_source_hashes(task, input_path) != image_assets:
            raise ValueError("Image assets changed during evaluation")

        prediction_path = output / "predictions.csv"
        frame = pd.DataFrame({task.target_column: prediction})
        if task.id_column and ids is not None:
            frame.insert(0, task.id_column, ids.to_numpy())
        frame.to_csv(prediction_path, index=False)
        probability_path = None
        if probabilities is not None:
            probability_path = output / "predictions_probabilities.csv"
            frame = pd.DataFrame(probabilities, columns=[f"probability__{label}" for label in classes])
            if task.id_column and ids is not None:
                frame.insert(0, task.id_column, ids.to_numpy())
            frame.to_csv(probability_path, index=False)
        result = {
            "status": "completed", "purpose": "final_evaluation",
            "created_at": now(), "metric": "accuracy",
            "test_accuracy": float(accuracy_score(labels, prediction)),
            "rows": len(prediction), "classes": classes.tolist(),
            "class_support": [{"label": label, "rows": int(np.sum(labels.to_numpy() == label))} for label in classes.tolist()],
            "confusion_matrix": confusion_matrix(labels, prediction, labels=classes).tolist(),
            "confusion_matrix_axes": "Rows=true class, columns=predicted class, both in classes order",
            "model": str(model_path), "model_sha256": model_sha256,
            "source_sha256": bundle["source_sha256"],
            "evaluation_source_sha256": manifest["evaluation_source"]["source_sha256"],
            "training_and_evaluation_source_match": bundle["source_sha256"] == manifest["evaluation_source"]["source_sha256"],
            "data_sha256": data_hashes,
            "predictions": str(prediction_path),
            "probabilities": str(probability_path) if probability_path else None,
            "selection": bundle["selection"],
            "training_performed": False, "planner_calls": 0,
            "note": "Held-out test score; this result must not enter model selection or further tuning.",
        }
        write_json(output / "summary.json", result)
        return result
    except Exception as exc:
        write_json(output / "summary.json", {"status": "failed", "purpose": "final_evaluation", "error_type": type(exc).__name__, "error": str(exc)})
        raise
