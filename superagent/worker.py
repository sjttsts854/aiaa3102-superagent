"""One bounded training subprocess. This process never opens prediction data."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import joblib

from .data import load_data, load_task
from .models import build_pipeline, score_pipeline
from .utils import write_json


def execute(job_path: Path) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        task = load_task(job["task_path"])
        data = load_data(task)
        final_fit = job.get("refit", False)
        X, y = (data.X_refit, data.y_refit) if final_fit else (data.X_train, data.y_train)
        model = build_pipeline(task, job["candidate"], X)
        model.fit(X, y)
        score = None if final_fit else score_pipeline(model, data.X_valid, data.y_valid, task.metric)
        training_score = None if final_fit else score_pipeline(model, X, y, task.metric)
        joblib.dump(model, output / "pipeline.joblib")
        result = {"status": "success", "score": score, "training_rows": len(y),
                  "training_score": training_score,
                  "generalization_gap": None if final_fit else training_score - score}
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "failed", "score": None, "error_type": type(exc).__name__, "error": str(exc)}
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    write_json(output / "result.json", result)
    return 0 if result["status"] == "success" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return execute(parser.parse_args().job)


if __name__ == "__main__":
    raise SystemExit(main())
