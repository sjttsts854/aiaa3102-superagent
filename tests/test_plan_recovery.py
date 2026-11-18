"""Paid plan failures stay charged, receive safe feedback, and cannot loop."""

import importlib
import json
from dataclasses import asdict

import httpx
import pytest

from superagent.demo import make_demo
from superagent.planner import LLMPlanner
from superagent.runner import run
from superagent.usage import InvalidPlanError, LLMConfig


planner_module = importlib.import_module("superagent.planner")


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    config = LLMConfig(
        base_url="https://example.com/v1", model="test-fixed-model",
        api_key_env="PLAN_RECOVERY_TEST_KEY", model_family="deepseek",
        model_release_date="2026-10-01", pricing_as_of="2026-10-21",
        input_rmb_per_million=2., output_rmb_per_million=8.,
        price_source="https://example.com/pricing", model_source="https://example.com/model",
        verified=True,
    )
    monkeypatch.setenv(config.api_key_env, "local-test-secret")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    return path


def install_responses(monkeypatch, contents):
    iterator = iter(contents)
    contexts = []

    def fake_post(config, payload, api_key):
        contexts.append(json.loads(payload["messages"][1]["content"]))
        # An unexpected retry exhausts this iterator and fails the test.
        content = next(iterator)
        body = {
            "model": config.model,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", config.base_url))

    monkeypatch.setattr(planner_module, "post_completion", fake_post)
    return contexts


def demo_task(tmp_path):
    make_demo(tmp_path / "data")
    path = tmp_path / "data/tabular_task.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("predict_path")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_malformed_json_feedback_then_valid_plan_clears_it(tmp_path, monkeypatch, config_path):
    valid = '{"action":"experiment","candidate":{"algorithm":"logistic_regression","params":{"C":0.1}},"reason":"different regularization"}'
    contexts = install_responses(monkeypatch, [valid + "}", "not JSON", valid, '{"action":"stop","reason":"done"}'])
    planner = LLMPlanner(config_path, tmp_path / "run")
    catalog = {"logistic_regression": {"C": "float"}}
    for _ in range(2):
        with pytest.raises(InvalidPlanError):
            planner.propose({}, [], catalog)
    assert planner.propose({}, [], catalog)["action"] == "experiment"
    assert planner.propose({}, [], catalog)["action"] == "stop"
    assert "previous_plan_error" not in contexts[0]
    assert "Invalid JSON" in contexts[1]["previous_plan_error"]
    assert "extra closing braces" in contexts[2]["previous_plan_error"]
    assert "previous_plan_error" not in contexts[3]
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(4 * .00036)
    assert planner.summary()["accounting_complete"] is True


def test_feedback_never_echoes_raw_exception_or_model_text(tmp_path, monkeypatch, config_path):
    planner = LLMPlanner(config_path, tmp_path / "run")
    contexts = []

    def request(messages):
        contexts.append(json.loads(messages[1]["content"]))
        if len(contexts) == 1:
            raise InvalidPlanError("private-token-1234 C:/private/data.csv malicious raw model output")
        return {"action": "stop", "reason": "done"}

    monkeypatch.setattr(planner, "_request_json", request)
    with pytest.raises(InvalidPlanError):
        planner.propose({}, [], {})
    planner.propose({}, [], {})
    serialized = json.dumps(contexts[1])
    assert "previous_plan_error" in contexts[1]
    assert "private-token" not in serialized
    assert "data.csv" not in serialized
    assert "malicious" not in serialized


def test_duplicate_candidate_gets_specific_feedback_and_is_charged(tmp_path, monkeypatch, config_path):
    candidate = {"algorithm": "logistic_regression", "params": {}}
    duplicate = json.dumps({"action": "experiment", "candidate": candidate, "reason": "repeat"})
    contexts = install_responses(monkeypatch, [duplicate, '{"action":"stop","reason":"done"}'])
    planner = LLMPlanner(config_path, tmp_path / "run")
    history = [{"candidate": candidate, "status": "success", "score": .7}]
    with pytest.raises(InvalidPlanError):
        planner.propose({}, history, {"logistic_regression": {}})
    planner.propose({}, history, {"logistic_regression": {}})
    assert "Repeated candidate" in contexts[1]["previous_plan_error"]
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(2 * .00036)


def test_three_consecutive_invalid_plans_stop_and_keep_charged_baseline(tmp_path, monkeypatch, config_path):
    bad = '{"action":"stop","reason":"extra brace"}}'
    contexts = install_responses(monkeypatch, [bad] * 3)
    output = tmp_path / "run"
    summary = run(demo_task(tmp_path), output, "llm", config_path)
    assert len(contexts) == summary["planning_attempts"] == summary["llm_requests"] == 3
    assert summary["invalid_plan_count"] == summary["consecutive_invalid_plans"] == 3
    assert summary["planner_stop_reason"] == "consecutive_invalid_plan_limit"
    assert summary["status"] == "completed_with_warnings"
    assert summary["experiments_completed"] == summary["selected_trial"] == 1
    assert (output / "model.joblib").is_file()
    assert summary["usage"]["accounted_cost_rmb"] == pytest.approx(3 * .00036)
    assert summary["usage"]["accounting_complete"] is True
    assert summary["usage"]["blocked"] is False
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "invalid_plan" for row in events) == 3
    assert any(row["event"] == "planner_stopped" and row["reason"] == "consecutive_invalid_plan_limit" for row in events)


def test_valid_decision_resets_consecutive_failure_counter(tmp_path, monkeypatch, config_path):
    valid = '{"action":"experiment","candidate":{"algorithm":"logistic_regression","params":{"C":0.1}},"reason":"different regularization"}'
    contexts = install_responses(monkeypatch, ["bad", "bad", valid, "bad", "bad", '{"action":"stop","reason":"done"}'])
    summary = run(demo_task(tmp_path), tmp_path / "run", "llm", config_path)
    assert summary["status"] == "completed"
    assert summary["invalid_plan_count"] == 4
    assert summary["consecutive_invalid_plans"] == 0
    assert summary["planning_attempts"] == summary["llm_requests"] == 6
    assert summary["experiments_completed"] == 2
    assert summary["planner_stop_reason"] == "planner_requested_stop"
    assert "previous_plan_error" not in contexts[3]
    assert summary["usage"]["accounted_cost_rmb"] == pytest.approx(6 * .00036)


def test_preflight_budget_exhaustion_is_a_normal_zero_call_stop(tmp_path, monkeypatch, config_path):
    contexts = install_responses(monkeypatch, [])
    summary = run(demo_task(tmp_path), tmp_path / "run", "llm", config_path, budget_limit=.001)
    assert contexts == []
    assert summary["status"] == "completed"
    assert summary["planner_stop_reason"] == "budget_exhausted"
    assert summary["planning_attempts"] == 1
    assert summary["llm_requests"] == summary["invalid_plan_count"] == 0
    assert summary["warnings"] == []
    assert summary["usage"]["accounted_cost_rmb"] == 0
    assert summary["usage"]["accounting_complete"] is True
    assert summary["usage"]["blocked"] is False
