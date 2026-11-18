"""Generated estimator execution and portable, process-isolated model handles.

Python AST checks/audit hooks are defense in depth, NOT an adversarial OS sandbox.
Use only locally generated/reviewed artifacts. For hostile code use an external
OS container with network, filesystem and memory restrictions. Cloudpickle
state requires matching Python/dependency versions and is kept opaque here.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.utils.metaestimators import available_if

from .data import load_data, load_task
from .utils import write_json

MAX_SOURCE_BYTES = 10000
MAX_STATE_BYTES = 256 * 1024 * 1024
ALLOWED_IMPORTS = {"numpy", "pandas", "scipy", "sklearn", "math", "re",
                   "collections", "itertools", "functools", "statistics", "typing", "catboost", "PIL"}
FORBIDDEN_MODULES = ("sklearn.datasets", "sklearn.externals", "scipy.io",
                     "numpy.ctypeslib", "numpy.f2py", "numpy.testing")
FORBIDDEN_SPECIAL_METHODS = {"__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
    "__getnewargs__", "__getnewargs_ex__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "dir",
                   "getattr", "setattr", "delattr", "input", "help", "breakpoint", "exit", "quit"}
FORBIDDEN_ATTRIBUTES = FORBIDDEN_NAMES | {"load", "loads", "save", "savez", "savez_compressed", "dump", "dumps",
    "fromfile", "tofile", "memmap", "loadtxt", "savetxt", "genfromtxt", "DataSource", "ctypes", "ctypeslib",
    "f2py", "query", "to_pickle", "to_csv", "to_json", "to_excel", "to_sql", "to_hdf", "to_feather",
    "to_parquet", "to_stata", "to_clipboard", "to_html", "to_xml", "to_markdown", "get_data", "get_source"}


class CodePolicyError(ValueError):
    """Candidate source violates the documented local execution contract."""


def permitted_import(module: str) -> bool:
    return (isinstance(module, str) and module.split(".")[0] in ALLOWED_IMPORTS
            and not any(part.startswith("_") for part in module.split("."))
            and not any(module == blocked or module.startswith(blocked + ".") for blocked in FORBIDDEN_MODULES))


def validate_source(source: str) -> ast.Module:
    if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise CodePolicyError(f"source must be nonempty Python, at most {MAX_SOURCE_BYTES} UTF-8 bytes")
    tree = ast.parse(source, filename="<generated_candidate>")
    builders = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model"]
    if len(builders) != 1:
        raise CodePolicyError("Define exactly one top-level build_model(X_train, y_train, seed)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if (getattr(node, "level", 0) or any(not permitted_import(name) for name in modules)
                    or any(alias.name == "*" or alias.name.startswith("_") for alias in node.names)):
                raise CodePolicyError(f"Import is outside the scientific-library contract (line {node.lineno})")
            if isinstance(node, ast.ImportFrom) and any(not permitted_import(node.module + "." + alias.name)
                                                       or alias.name in FORBIDDEN_ATTRIBUTES for alias in node.names):
                raise CodePolicyError(f"Imported symbol is not permitted (line {node.lineno})")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CodePolicyError(f"Name {node.id!r} is not permitted (line {node.lineno})")
        if ((isinstance(node, ast.Name) and node.id in FORBIDDEN_SPECIAL_METHODS)
                or (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_SPECIAL_METHODS)):
            raise CodePolicyError("Custom serialization/introspection hooks are not permitted")
        if isinstance(node, ast.Attribute):
            private = node.attr.startswith("_") and not (isinstance(node.value, ast.Name) and node.value.id == "self" and not node.attr.startswith("__"))
            if private or node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith(("read_", "fetch_")):
                raise CodePolicyError(f"Attribute {node.attr!r} is not permitted (line {node.lineno})")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("__") and node.name not in {"__init__", "__call__"}:
            raise CodePolicyError(f"Special method {node.name!r} is not permitted")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise CodePolicyError("Global/nonlocal mutation and asynchronous execution are not supported")
    return tree


def safe_diagnostic(exc: BaseException) -> str:
    """Bound feedback; omit absolute paths and source data from tracebacks."""
    if isinstance(exc, SyntaxError):
        message = f"SyntaxError at generated line {exc.lineno}: {exc.msg}"
    else:
        message = f"{type(exc).__name__}: {str(exc)}"
    message = re.sub(r"[A-Za-z]:[\\/][^\s\"'<>]*", "<path>", message)
    message = re.sub(r"(?<!\w)/(?:[^\s/:\"'<>]+/)+[^\s\"'<>]*", "<path>", message)
    return message[:800]


def _aliases(frame: pd.DataFrame, expected: list[str] | None = None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Generated models accept a pandas DataFrame")
    if expected is not None and list(frame.columns) != expected:
        raise ValueError("Prediction features differ from the fitted schema")
    result = frame.copy(deep=True)
    result.columns = [f"column_{i}" for i in range(len(result.columns))]
    result.attrs = {}
    return result


def _child(payload: dict, folder: Path, timeout: float, runtime_root: Path | None = None) -> tuple[dict, bytes | None]:
    folder.mkdir(parents=True, exist_ok=True)
    payload_path, response_path = folder / "input.joblib", folder / "response.json"
    joblib.dump(payload, payload_path)
    root = runtime_root or Path(__file__).resolve().parent.parent
    bootstrap = "import sys; sys.path.insert(0, sys.argv[1]); from superagent.code_worker import main; raise SystemExit(main(sys.argv[2:]))"
    environment = {key: value for key, value in os.environ.items() if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH"}}
    environment.update(TEMP=str(folder), TMP=str(folder), PYTHONDONTWRITEBYTECODE="1", LOKY_MAX_CPU_COUNT="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    with (folder / "stdout.log").open("w", encoding="utf-8") as stdout, (folder / "stderr.log").open("w", encoding="utf-8") as stderr:
        subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap, str(root), str(payload_path), str(response_path)],
                       cwd=folder, env=environment, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
    if not response_path.is_file() or response_path.stat().st_size > 128 * 1024 * 1024:
        raise RuntimeError("Generated worker did not return a bounded response; see worker logs")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(response, dict) or response.get("status") not in {"success", "failed"}:
        raise ValueError("Generated worker returned invalid response metadata")
    state = None
    if payload["mode"] == "fit" and response["status"] == "success":
        state_path = folder / "estimator.cloudpickle"
        if not state_path.is_file() or not 0 < state_path.stat().st_size <= MAX_STATE_BYTES:
            raise ValueError("Generated estimator state is missing or exceeds 256 MiB")
        state = state_path.read_bytes()  # Opaque bytes; never deserialize candidate objects here.
    return response, state


def _labels(value, rows: int, classes=None) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows or any(type(x) not in (int, float, str, bool) for x in value):
        raise ValueError("Generated model returned invalid prediction labels/shape")
    array = np.asarray(value)
    if array.ndim != 1 or (array.dtype.kind in "f" and not np.isfinite(array).all()):
        raise ValueError("Generated model returned nonfinite or multidimensional labels")
    if classes is not None and not set(value).issubset(set(classes.tolist())):
        raise ValueError("Generated model returned unknown class labels")
    return array


def _probabilities(value, rows: int, classes: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (result.shape != (rows, classes) or not np.isfinite(result).all() or (result < 0).any()
            or (result > 1).any() or not np.allclose(result.sum(axis=1), 1, atol=1e-7)):
        raise ValueError("Generated model returned invalid class probabilities")
    return result


def _score(y, prediction, probability, classes, metric):
    if metric == "accuracy":
        result = accuracy_score(y, prediction)
    elif metric == "balanced_accuracy":
        result = balanced_accuracy_score(y, prediction)
    elif metric == "f1_macro":
        result = f1_score(y, prediction, average="macro", zero_division=0)
    elif metric == "roc_auc":
        if probability is None or len(classes) != 2:
            raise ValueError("roc_auc requires a binary classifier with predict_proba")
        result = roc_auc_score(np.asarray(y) == classes[1], probability[:, 1])
    else:
        raise ValueError("Unsupported evaluation metric")
    if not math.isfinite(result):
        raise ValueError("Generated model metric is nonfinite")
    return float(result)


class GeneratedModel:
    """Serializable handle; candidate code executes only in a guarded child."""
    def __init__(self, source: str, state: bytes, classes, feature_columns: list[str], has_proba: bool, timeout: float = 120):
        self.source = source
        self.source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.state = state
        self.state_sha256 = hashlib.sha256(state).hexdigest()
        self.python_version = tuple(sys.version_info[:3])
        self.package_versions = {name: importlib.metadata.version(name) for name in
                                 ("cloudpickle", "numpy", "pandas", "scipy", "scikit-learn")}
        for package in ("catboost", "Pillow"):
            try:
                self.package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        self.classes_ = np.asarray(classes)
        self.feature_columns = feature_columns
        self.has_predict_proba_ = has_proba
        self.timeout = timeout

    def _infer(self, X, method):
        validate_source(self.source)
        if tuple(sys.version_info[:3]) != self.python_version or any(
                importlib.metadata.version(name) != version for name, version in self.package_versions.items()):
            raise ValueError("Generated model requires its recorded Python and dependency versions")
        if hashlib.sha256(self.source.encode("utf-8")).hexdigest() != self.source_sha256:
            raise ValueError("Generated source fingerprint changed")
        if hashlib.sha256(self.state).hexdigest() != self.state_sha256:
            raise ValueError("Generated estimator state fingerprint changed")
        payload = {"mode": "predict", "source": self.source, "state": self.state,
                   "X": _aliases(X, self.feature_columns), "method": method}
        with tempfile.TemporaryDirectory(prefix="superagent_inference_") as tmp:
            response, _ = _child(payload, Path(tmp), self.timeout)
        if response["status"] != "success":
            raise RuntimeError(response.get("error_feedback", "Generated inference failed"))
        if method == "predict":
            return _labels(response["prediction"], len(X), self.classes_)
        return _probabilities(response["prediction"], len(X), len(self.classes_))

    def predict(self, X):
        return self._infer(X, "predict")

    @available_if(lambda self: self.has_predict_proba_)
    def predict_proba(self, X):
        return self._infer(X, "predict_proba")


def run_generated_candidate(task_path: Path, candidate: dict, folder: Path, timeout: float, refit: bool = False) -> dict:
    """Evaluate generated code without exposing validation labels to that code."""
    started = time.monotonic()
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "job.json", {"candidate": candidate, "refit": refit, "protocol": "generated-estimator-v1"})
    try:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        if set(candidate) != {"algorithm", "name", "source"} or candidate["algorithm"] != "generated_python":
            raise ValueError("Generated candidate requires algorithm, name and source")
        if not isinstance(candidate["name"], str) or not 1 <= len(candidate["name"].strip()) <= 80:
            raise ValueError("Generated candidate name must be a short nonempty string")
        source = candidate["source"]
        if isinstance(source, str):
            (folder / "candidate.py").write_text(source, encoding="utf-8")
        validate_source(source)
        task = load_task(task_path)
        data = load_data(task)
        X, y = (data.X_refit, data.y_refit) if refit else (data.X_train, data.y_train)
        payload = {"mode": "fit", "source": source, "seed": task.seed,
                   "X_train": _aliases(X), "y_train": y.copy(),
                   "prediction_features": {} if refit else {"train": _aliases(data.X_train), "validation": _aliases(data.X_valid)}}
        runtime = Path(task_path).resolve().parent / "source_snapshot"
        if not (runtime / "superagent" / "code_worker.py").is_file():
            raise ValueError("Generated worker must execute from a captured source snapshot")
        response, state = _child(payload, folder / "worker", timeout, runtime)
        if response["status"] == "failed":
            result = {k: response[k] for k in ("status", "error_type", "error_feedback", "generated_line") if k in response}
            result["score"] = None
        else:
            expected_serialization = {"python": list(sys.version_info[:3]), "cloudpickle": importlib.metadata.version("cloudpickle")}
            if response.get("serialization") != expected_serialization:
                raise ValueError("Generated worker serialization versions differ from the parent")
            classes = _labels(response.get("classes"), len(pd.unique(y)))
            if set(classes.tolist()) != set(pd.unique(y).tolist()) or len(set(classes.tolist())) != len(classes):
                raise ValueError("Generated model classes differ from training labels")
            has_proba = response.get("has_predict_proba")
            if type(has_proba) is not bool:
                raise ValueError("Missing generated probability capability metadata")
            scores = {}
            if not refit:
                for split, targets in (("train", data.y_train), ("validation", data.y_valid)):
                    output = response["predictions"][split]
                    prediction = _labels(output["labels"], len(targets), classes)
                    probabilities = _probabilities(output["probabilities"], len(targets), len(classes)) if has_proba else None
                    scores[split] = _score(targets, prediction, probabilities, classes, task.metric)
            model = GeneratedModel(source, state, classes, list(X.columns), has_proba, timeout)
            joblib.dump(model, folder / "pipeline.joblib")
            result = {"status": "success", "score": scores.get("validation"), "training_score": scores.get("train"),
                      "generalization_gap": scores["train"] - scores["validation"] if scores else None,
                      "training_rows": len(y), "generated_source_sha256": model.source_sha256,
                      "validation_labels_passed_to_generated_code": False,
                      "runtime_is_os_security_sandbox": False}
    except subprocess.TimeoutExpired:
        result = {"status": "failed", "score": None, "error_type": "TimeoutError",
                  "error_feedback": f"Generated code exceeded {timeout:g} seconds; simplify the algorithm or reduce its workload."}
    except Exception as exc:
        result = {"status": "failed", "score": None, "error_type": type(exc).__name__, "error_feedback": safe_diagnostic(exc)}
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    write_json(folder / "result.json", result)
    return result
