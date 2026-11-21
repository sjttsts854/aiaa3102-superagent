import json
import subprocess
from pathlib import Path

import joblib
import httpx
import pandas as pd
import pytest

from superagent.demo import make_demo
from superagent import __version__
from superagent.utils import file_hash
from superagent.runner import export_predictions, run, run_worker


@pytest.mark.parametrize("kind", ["tabular", "text"])
def test_complete_run_selects_before_prediction_and_reloads(tmp_path, kind):
    make_demo(tmp_path / "data")
    task_path = tmp_path / "data" / f"{kind}_task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["max_experiments"] = 2
    task_path.write_text(json.dumps(task), encoding="utf-8")
    output = tmp_path / "run"
    summary = run(task_path, output)
    assert summary["status"] == "completed"
    assert summary["development_demo"] is True
    assert summary["experiments_completed"] == 2
    assert 0 <= summary["best_validation_score"] <= 1
    assert (output / "source_snapshot" / "superagent" / "worker.py").is_file()
    events = [json.loads(line)["event"] for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events.index("model_selected") < events.index("prediction_started")
    model = joblib.load(output / "model.joblib")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert summary["project_version"] == manifest["project_version"] == model["project_version"] == __version__
    assert len(summary["run_id"]) == 32
    assert summary["run_id"] == manifest["run_id"] == model["run_id"]
    assert summary["source_sha256"] == manifest["source_sha256"] == model["source_sha256"]
    assert summary["model_sha256"] == file_hash(output / "model.joblib")
    reloaded_path = tmp_path / "reload.csv"
    export_predictions(model, tmp_path / "data" / f"{kind}_predict.csv", reloaded_path)
    pd.testing.assert_frame_equal(pd.read_csv(output / "predictions.csv"), pd.read_csv(reloaded_path))
    assert "test_score" not in summary
    profile = (output / "data_profile.json").read_text(encoding="utf-8")
    assert task["name"] not in profile
    assert "train.csv" not in profile
    with pytest.raises(ValueError, match="empty"):
        run(task_path, output)


def test_timeout_becomes_a_logged_failed_experiment(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_worker(tmp_path / "task.json", {"algorithm": "logistic_regression", "params": {}}, tmp_path / "trial", 0.1)
    assert result["status"] == "failed"
    assert result["error_type"] == "TimeoutError"
    assert json.loads((tmp_path / "trial" / "result.json").read_text())["status"] == "failed"


def test_invalid_timeout_has_no_side_effects(tmp_path):
    with pytest.raises(ValueError, match="Timeout"):
        run(Path("missing.json"), tmp_path / "run", experiment_timeout=float("nan"))
    assert not (tmp_path / "run").exists()


def _mock_llm_config(tmp_path, monkeypatch):
    # Test-only configuration and fake transport: no real API request is sent.
    monkeypatch.setenv("SUPERAGENT_TEST_API_KEY", "test-only-not-a-real-key")
    config = {
        "base_url": "https://provider.example/v1", "model": "test-model",
        "api_key_env": "SUPERAGENT_TEST_API_KEY", "model_family": "deepseek",
        "model_release_date": "2026-01-01", "pricing_as_of": "2026-10-21",
        "input_rmb_per_million": 1.0, "output_rmb_per_million": 2.0,
        "price_source": "https://provider.example/pricing",
        "model_source": "https://provider.example/model", "verified": True,
    }
    path = tmp_path / "test_llm.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_real_planner_interface_receives_results_and_records_usage(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    config = _mock_llm_config(tmp_path, monkeypatch)
    contexts = []

    def fake_post(url, **kwargs):
        context = json.loads(kwargs["json"]["messages"][1]["content"])
        contexts.append(context)
        assert context["validation_history"][0]["status"] == "success"
        for record in context["validation_history"]:
            assert 0 <= record["training_score"] <= 1
            assert record["generalization_gap"] == pytest.approx(record["training_score"] - record["score"])
        assert "synthetic" not in json.dumps(context)
        plan = ({"action": "experiment", "candidate": {"algorithm": "random_forest", "params": {}}, "reason": "Try a nonlinear method after the baseline."}
                if len(contexts) == 1 else {"action": "stop", "reason": "Enough validation evidence for this mocked integration test."})
        body = {"model": "test-model", "usage": {"prompt_tokens": 300, "completion_tokens": 80}, "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    summary = run(tmp_path / "data" / "tabular_task.json", tmp_path / "run", "llm", config)
    assert summary["status"] == "completed"
    assert summary["llm_requests"] == 2
    assert summary["valid_llm_decisions"] == 2
    assert summary["experiments_completed"] == 2
    assert len(contexts[1]["validation_history"]) == 2
    assert summary["usage"]["accounted_cost_rmb"] == pytest.approx(0.00092)


def test_unknown_paid_usage_is_a_partial_result_not_completed(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    config = _mock_llm_config(tmp_path, monkeypatch)

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("simulated uncertain outcome")

    monkeypatch.setattr(httpx, "post", timeout)
    summary = run(tmp_path / "data" / "text_task.json", tmp_path / "run", "llm", config)
    assert summary["status"] == "partial_unresolved_accounting"
    assert summary["accounting_complete"] is False
    assert summary["llm_requests"] == 1
    assert summary["valid_llm_decisions"] == 0
    assert summary["experiments_completed"] == 1
    assert (tmp_path / "run" / "model.joblib").is_file()
    assert summary["usage"]["pending_reserved_rmb"] > 0
