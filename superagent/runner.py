from __future__ import annotations

import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import pandas as pd

from .data import load_data, load_prediction_data, load_task
from .models import candidate_catalog
from .planner import InvalidPlanError, LLMPlanner, OfflinePlanner
from .usage import BudgetExhausted
from .utils import append_jsonl, file_hash, now, object_hash, write_json


def snapshot_source(output: Path) -> dict:
    project = Path(__file__).resolve().parent.parent
    files = sorted((project / "superagent").glob("*.py")) + [project / "requirements.txt"]
    hashes = {}
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(project)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = file_hash(path)
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "httpx")}
    for package in ("catboost", "cloudpickle", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return {"source_sha256": fingerprint, "files": hashes, "python": sys.version, "packages": versions}


def run_worker(task_path: Path, candidate: dict, folder: Path, timeout: float, refit: bool = False) -> dict:
    if candidate.get("algorithm") == "generated_python":
        from .generated_code import run_generated_candidate
        return run_generated_candidate(task_path, candidate, folder, timeout, refit=refit)
    folder.mkdir(parents=True, exist_ok=False)
    job_path = folder / "job.json"
    write_json(job_path, {"task_path": str(task_path), "candidate": candidate, "output_dir": str(folder), "refit": refit})
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    started = time.monotonic()
    # Training runs from the captured code snapshot, including after local edits.
    project = task_path.parent / "source_snapshot"
    result_path = folder / "result.json"
    try:
        with (folder / "stdout.log").open("w", encoding="utf-8") as stdout, (folder / "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run([sys.executable, "-m", "superagent.worker", "--job", str(job_path)], cwd=project, env=environment, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {"status": "failed", "score": None, "error_type": "WorkerError", "error": f"Worker exited {completed.returncode}; see stderr.log"}
    except subprocess.TimeoutExpired:
        result = {"status": "failed", "score": None, "error_type": "TimeoutError", "error": f"Training exceeded {timeout} seconds"}
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    write_json(result_path, result)
    return result


def export_predictions(bundle: dict, input_path: Path, output_path: Path) -> dict:
    task = replace(bundle["task"], predict_path=input_path.resolve())
    X, ids = load_prediction_data(task, bundle["feature_columns"])
    model = bundle["pipeline"]
    predictions = model.predict(X)
    result = pd.DataFrame({task.target_column: predictions})
    if task.id_column and ids is not None:
        result.insert(0, task.id_column, ids.to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    probability_path = None
    if hasattr(model, "predict_proba"):
        probabilities = pd.DataFrame(model.predict_proba(X), columns=[f"probability__{label}" for label in model.classes_])
        if task.id_column and ids is not None:
            probabilities.insert(0, task.id_column, ids.to_numpy())
        probability_path = output_path.with_name(output_path.stem + "_probabilities.csv")
        probabilities.to_csv(probability_path, index=False)
    return {"rows": len(result), "predictions": str(output_path), "probabilities": str(probability_path) if probability_path else None, "classes_in_probability_order": model.classes_.tolist()}


def run(task_path: Path, output: Path, planner_mode: str = "offline", llm_config: Path | None = None, experiment_timeout: float = 120, refit_timeout: float = 240, budget_limit: float | None = None) -> dict:
    if not math.isfinite(experiment_timeout) or experiment_timeout <= 0 or not math.isfinite(refit_timeout) or refit_timeout <= 0:
        raise ValueError("Timeouts must be positive finite numbers")
    task = load_task(task_path)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new directory for every run")
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"

    def event(kind: str, **fields):
        append_jsonl(events, {"time": now(), "event": kind, **fields})

    event("run_started", planner=planner_mode, purpose="development_demo" if planner_mode == "offline" else "llm_enabled_development_run")
    try:
        metadata = snapshot_source(output)
        metadata.update({"created_at": now(), "planner": planner_mode, "experiment_timeout": experiment_timeout, "refit_timeout": refit_timeout, "data_sha256": {"train": file_hash(task.train_path)}})
        if task.validation_path:
            metadata["data_sha256"]["validation"] = file_hash(task.validation_path)
        for split in ("train", "validation"):
            labels_path = getattr(task, f"{split}_labels_path", None)
            if labels_path is not None:
                metadata["data_sha256"][f"{split}_labels"] = file_hash(labels_path)
        image_assets = {}
        if task.data_type == "image":
            from .data import image_source_hashes
            for split in ("train", "validation"):
                image_csv = getattr(task, f"{split}_path", None)
                if image_csv is not None:
                    image_assets[split] = image_source_hashes(task, image_csv)
                    metadata["data_sha256"][f"{split}_images"] = object_hash(image_assets[split])
            metadata["image_assets_sha256"] = image_assets
        write_json(output / "manifest.json", metadata)
        frozen_task = output / "task_snapshot.json"
        write_json(frozen_task, asdict(task))
        if planner_mode == "offline":
            planner = OfflinePlanner()
        elif planner_mode in {"llm", "llm-code"}:
            if llm_config is None:
                raise ValueError("LLM mode requires --llm-config")
            if planner_mode == "llm-code":
                from .coding_planner import CodingPlanner
                planner = CodingPlanner(llm_config, output, budget_limit=budget_limit)
            else:
                planner = LLMPlanner(llm_config, output, budget_limit=budget_limit)
        else:
            raise ValueError("planner must be offline, llm, or llm-code")
        data = load_data(task)
        profile = dict(data.profile)
        if planner_mode == "llm-code":
            profile["feature_schema"] = []
            for index, column in enumerate(data.X_train):
                values = data.X_train[column]
                kind = ("image" if task.data_type == "image" else "text" if task.data_type in {"text", "paired_text"}
                        else "numeric" if pd.api.types.is_numeric_dtype(values.dtype) else "categorical")
                item = {"name": f"column_{index}", "kind": kind, "dtype": str(values.dtype)}
                if kind != "image":
                    item.update(missing_fraction=float(values.isna().mean()), unique_count=int(values.nunique(dropna=True)))
                profile["feature_schema"].append(item)
        write_json(output / "data_profile.json", profile)
        write_json(output / "split_manifest.json", data.split_manifest)
        catalog = candidate_catalog(task.data_type)
        history = []
        attempts = 0
        planning_attempts = 0
        valid_llm_decisions = 0
        invalid_plan_count = 0
        consecutive_invalid_plans = 0
        planner_stop_reason = None
        warnings = []
        seen = set()

        def invalid_plan(reason: str) -> bool:
            nonlocal invalid_plan_count, consecutive_invalid_plans, planner_stop_reason
            invalid_plan_count += 1
            consecutive_invalid_plans += 1
            event("invalid_plan", reason=reason, invalid_plan_count=invalid_plan_count,
                  consecutive_invalid_plans=consecutive_invalid_plans)
            if consecutive_invalid_plans < 3:
                return False
            planner_stop_reason = "consecutive_invalid_plan_limit"
            warnings.append("Planner stopped after 3 consecutive invalid plans; retained successful experiments")
            event("planner_stopped", reason=planner_stop_reason, planning_attempts=planning_attempts,
                  consecutive_invalid_plans=consecutive_invalid_plans)
            return True

        # A universal inexpensive baseline guarantees a useful first result.
        baseline = {"tabular": "logistic_regression", "text": "text_logistic", "paired_text": "paired_text_logistic", "image": "image_logistic"}[task.data_type]
        decision = {"action": "experiment", "candidate": {"algorithm": baseline, "params": {}}, "reason": "Automatic generic baseline before feedback-driven improvement."}
        while len(history) < task.max_experiments and attempts < task.max_experiments * 3:
            attempts += 1
            if attempts > 1:
                # Raw failure paths/messages and all task identifiers stay out of the LLM prompt.
                feedback = [{key: record.get(key) for key in ("candidate", "status", "score", "training_score", "generalization_gap", "duration_seconds", "error_type", "error_feedback", "generated_line")} for record in history]
                planning_attempts += 1
                try:
                    decision = planner.propose(profile, feedback, catalog)
                except BudgetExhausted:
                    planner_stop_reason = "budget_exhausted"
                    event("planner_stopped", reason=planner_stop_reason, planning_attempts=planning_attempts)
                    break
                except Exception as exc:
                    event("planner_error", error_type=type(exc).__name__, error=str(exc))
                    if isinstance(exc, InvalidPlanError):
                        if invalid_plan(str(exc)):
                            break
                        continue
                    planner_stop_reason = "planner_error"
                    warnings.append(f"Planner stopped: {type(exc).__name__}")
                    break
            event("decision", attempt=attempts, decision=decision)
            if not isinstance(decision, dict):
                event("invalid_decision", reason="Expected a JSON object")
                if invalid_plan("Expected a JSON object"):
                    break
                continue
            if decision.get("action") == "stop":
                if planner_mode in {"llm", "llm-code"} and attempts > 1:
                    valid_llm_decisions += 1
                consecutive_invalid_plans = 0
                planner_stop_reason = "planner_requested_stop"
                break
            candidate = decision.get("candidate")
            if decision.get("action") != "experiment" or not isinstance(candidate, dict):
                event("invalid_decision", reason="Expected experiment candidate or stop")
                if invalid_plan("Expected experiment candidate or stop"):
                    break
                continue
            signature = json.dumps(candidate, sort_keys=True)
            if signature in seen:
                event("duplicate_candidate_skipped", candidate=candidate)
                if invalid_plan("Repeated candidate"):
                    break
                continue
            consecutive_invalid_plans = 0
            if planner_mode in {"llm", "llm-code"} and attempts > 1:
                valid_llm_decisions += 1
            seen.add(signature)
            trial = len(history) + 1
            folder = output / "experiments" / f"trial_{trial:03d}"
            event("experiment_started", trial=trial, candidate=candidate)
            result = run_worker(frozen_task, candidate, folder, experiment_timeout)
            record = {"trial": trial, "candidate": candidate, **result}
            history.append(record)
            append_jsonl(output / "experiments.jsonl", record)
            event("experiment_finished", **record)
            print(f"Trial {trial}: {candidate.get('algorithm')} | {result['status']}, validation score={result.get('score')}", flush=True)
        if planner_stop_reason is None:
            planner_stop_reason = "max_experiments" if len(history) >= task.max_experiments else "planning_attempt_limit"
            if planner_stop_reason == "planning_attempt_limit":
                warnings.append("Planner reached the total planning attempt limit; retained successful experiments")
            event("planner_stopped", reason=planner_stop_reason, planning_attempts=planning_attempts)
        successful = [record for record in history if record["status"] == "success" and record.get("score") is not None and math.isfinite(record["score"])]
        if image_assets:
            for split, before in image_assets.items():
                if image_source_hashes(task, getattr(task, f"{split}_path")) != before:
                    raise ValueError("Image training/validation assets changed during experiments")
        if not successful:
            write_json(output / "usage_summary.json", planner.summary())
            raise RuntimeError("No successful experiment; inspect experiments.jsonl and worker stderr logs")
        best = max(successful, key=lambda record: record["score"])
        event("model_selected", trial=best["trial"], candidate=best["candidate"], validation_score=best["score"], metric=task.metric)
        write_json(output / "selection.json", best)
        # Selection is finished before any prediction/test data is opened.
        refit = run_worker(frozen_task, best["candidate"], output / "refit", refit_timeout, refit=True)
        if image_assets:
            for split, before in image_assets.items():
                if image_source_hashes(task, getattr(task, f"{split}_path")) != before:
                    raise ValueError("Image training/validation assets changed during refit")
        event("refit_finished", **refit)
        if refit["status"] == "success":
            model_path = output / "refit" / "pipeline.joblib"
        else:
            model_path = output / "experiments" / f"trial_{best['trial']:03d}" / "pipeline.joblib"
            warnings.append("Final refit failed; saved the selected validation-stage model as a partial result")
        bundle = {"pipeline": joblib.load(model_path), "task": task, "feature_columns": data.feature_columns, "selection": best, "source_sha256": metadata["source_sha256"]}
        joblib.dump(bundle, output / "model.joblib")
        prediction = None
        if task.predict_path:
            event("prediction_started")
            prediction = export_predictions(bundle, task.predict_path, output / "predictions.csv")
            metadata["data_sha256"]["prediction"] = file_hash(task.predict_path)
            write_json(output / "manifest.json", metadata)
            event("prediction_finished", **prediction)
        usage = planner.summary()
        write_json(output / "usage_summary.json", usage)
        status = "completed_with_warnings" if warnings else "completed"
        if not usage.get("accounting_complete", False):
            status = "partial_unresolved_accounting"
        elif usage.get("blocked", False):
            status = "partial_accounting_limits_violated"
        summary = {
            "status": status,
            "run_mode": planner_mode,
            "development_demo": planner_mode == "offline",
            "llm_requests": usage.get("request_count", 0),
            "planning_attempts": planning_attempts,
            "valid_llm_decisions": valid_llm_decisions,
            "generated_code_experiments": sum(r["candidate"].get("algorithm") == "generated_python" for r in history),
            "successful_generated_code_experiments": sum(r["candidate"].get("algorithm") == "generated_python" for r in successful),
            "selected_generated_code": best["candidate"].get("algorithm") == "generated_python",
            "invalid_plan_count": invalid_plan_count,
            "consecutive_invalid_plans": consecutive_invalid_plans,
            "planner_stop_reason": planner_stop_reason,
            "accounting_complete": usage.get("accounting_complete", False),
            "official_test_evaluated": False,
            "metric": task.metric,
            "best_validation_score": best["score"],
            "selected_trial": best["trial"],
            "selected_candidate": best["candidate"],
            "experiments_completed": len(successful),
            "experiments_attempted": len(history),
            "refit_success": refit["status"] == "success",
            "model": str(output / "model.joblib"),
            "prediction": prediction,
            "usage": usage,
            "warnings": warnings,
            "source_sha256": metadata["source_sha256"],
            "note": "Validation-only model development. Final test evaluation is a separate operation after freezing the solution. Offline mode or zero LLM calls must not be reported as a real LLM experiment. Unresolved accounting prevents use as a compliant formal result.",
        }
        write_json(output / "summary.json", summary)
        event("run_finished", status=summary["status"])
        return summary
    except Exception as exc:
        event("run_failed", error_type=type(exc).__name__, error=str(exc))
        if "planner" in locals():
            write_json(output / "usage_summary.json", planner.summary())
        write_json(output / "summary.json", {"status": "failed", "run_mode": planner_mode, "error_type": type(exc).__name__, "error": str(exc), "partial_artifacts": True})
        raise
