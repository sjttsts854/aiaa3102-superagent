"""Child-only generated source execution; guards are not an OS sandbox.

CPython explicitly documents that Python audit hooks may be bypassed by hostile
code: https://docs.python.org/3/library/sys.html#sys.addaudithook
State uses cloudpickle and must be treated as trusted-local, version-specific:
https://github.com/cloudpipe/cloudpickle
"""
from __future__ import annotations

import builtins
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import cloudpickle
import joblib
import numpy as np
import pandas as pd

from .generated_code import ALLOWED_IMPORTS, CodePolicyError, FORBIDDEN_NAMES, permitted_import, safe_diagnostic, validate_source


def _guarded_builtins():
    namespace = {name: value for name, value in vars(builtins).items() if name not in FORBIDDEN_NAMES}
    original_import = builtins.__import__
    printed = [0]

    def scientific_import(name, globals=None, locals=None, fromlist=(), level=0):
        # NumPy ndarray reductions invoke the caller's import hook for this
        # exact internal module. Candidate private imports remain AST-blocked.
        numpy_reduction_import = name == "numpy._core._methods" and level == 0
        if level or (not permitted_import(name) and not numpy_reduction_import):
            raise CodePolicyError("Runtime import is outside the scientific-library contract")
        return original_import(name, globals, locals, fromlist, level)

    def bounded_print(*args, **kwargs):
        text = " ".join(str(value) for value in args)
        printed[0] += len(text)
        if printed[0] > 65536:
            raise CodePolicyError("Generated output exceeded 64 KiB")
        builtins.print(*args, **kwargs)

    namespace["__import__"] = scientific_import
    namespace["print"] = bounded_print
    return namespace


def _install_audit_guard():
    active = [False]
    roots = [Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(), Path(__file__).resolve().parent]

    def library_path(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        try:
            path = Path(os.fsdecode(value)).resolve()
            return any(path.is_relative_to(root) for root in roots)
        except (ValueError, OSError):
            return False

    def audit(event, args):
        if not active[0]:
            return
        if event == "open":
            path, mode, flags = args
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing or not library_path(path):
                raise CodePolicyError("Generated code cannot access user files or write files")
        elif event in {"os.listdir", "os.scandir"}:
            if not library_path(args[0]):
                raise CodePolicyError("Generated code cannot enumerate user directories")
        elif (event.startswith(("socket.", "subprocess.", "shutil.", "winreg."))
              or event in {"os.system", "os.startfile", "os.startfile/2", "os.fork", "os.forkpty", "os.posix_spawn",
                           "os.exec", "os.spawn", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod",
                           "os.chdir", "os.link", "os.symlink", "os.truncate", "os.putenv", "os.unsetenv", "sys.addaudithook"}):
            raise CodePolicyError("Generated code cannot use network, processes or filesystem mutation")
    sys.addaudithook(audit)
    return active


def _plain_labels(values):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Prediction labels/classes must be one-dimensional")
    result = array.tolist()
    if any(type(item) not in (int, float, str, bool) for item in result):
        raise ValueError("Prediction labels/classes must be scalar JSON values")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("Prediction labels/classes must be finite")
    return result


def _prediction(model, X):
    output = {"labels": _plain_labels(model.predict(X)), "probabilities": None}
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Predicted probabilities must be finite")
        output["probabilities"] = probabilities.tolist()
    return output


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    input_path, response_path = map(Path, args)
    payload = joblib.load(input_path)  # Trusted parent-created input, before candidate code.
    source = payload["source"]
    guard = None
    try:
        tree = validate_source(source)
        compiled = compile(tree, "<generated_candidate>", "exec")
        # Load common dependency initialization before denying writes/processes.
        for name in ("scipy", "sklearn", "sklearn.pipeline", "sklearn.compose", "sklearn.preprocessing",
                     "sklearn.impute", "sklearn.linear_model", "sklearn.svm", "sklearn.ensemble",
                     "sklearn.feature_extraction.text", "sklearn.base", "threadpoolctl"):
            importlib.import_module(name)
        guard = _install_audit_guard()
        namespace = {"__name__": "__generated_candidate__", "__builtins__": _guarded_builtins()}
        guard[0] = True
        exec(compiled, namespace)
        if payload["mode"] == "fit":
            model = namespace["build_model"](payload["X_train"].copy(deep=True), payload["y_train"].copy(), payload["seed"])
            if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
                raise ValueError("build_model must return an estimator with fit and predict")
            model.fit(payload["X_train"], payload["y_train"])
            classes = _plain_labels(model.classes_)
            predictions = {name: _prediction(model, X) for name, X in payload["prediction_features"].items()}
            response = {"status": "success", "classes": classes,
                        "has_predict_proba": hasattr(model, "predict_proba"), "predictions": predictions,
                        "serialization": {"python": list(sys.version_info[:3]), "cloudpickle": cloudpickle.__version__}}
            state = cloudpickle.dumps(model)
        else:
            model = cloudpickle.loads(payload["state"])
            value = getattr(model, payload["method"])(payload["X"])
            if payload["method"] == "predict":
                value = _plain_labels(value)
            else:
                value = np.asarray(value, dtype=float)
                if not np.isfinite(value).all():
                    raise ValueError("Predicted probabilities must be finite")
                value = value.tolist()
            response = {"status": "success", "prediction": value}
            state = None
        guard[0] = False
        if state is not None:
            (response_path.parent / "estimator.cloudpickle").write_bytes(state)
    except Exception as exc:
        if guard is not None:
            guard[0] = False
        frames = traceback.extract_tb(exc.__traceback__)
        generated = [frame.lineno for frame in frames if frame.filename == "<generated_candidate>"]
        line = generated[-1] if generated else getattr(exc, "lineno", None)
        feedback = safe_diagnostic(exc)
        if line is not None:
            feedback = f"Generated line {line}: {feedback}"[:800]
        response = {"status": "failed", "error_type": type(exc).__name__, "error_feedback": feedback,
                    "generated_line": line}
    response_path.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")
    return 0 if response["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
