"""Exercise metering, actual payloads and feedback with no paid API calls."""
import json
from dataclasses import asdict
from pathlib import Path
import httpx
import pytest

from superagent.coding_planner import CodingPlanner
from superagent.api_check import check_api, check_config
from superagent.planner import LLMPlanner
from superagent.network import RequestDeadlineExceeded
from superagent.reconciliation import _read_ledger
from superagent.usage import AccountingError, AgentPlannerError, BudgetExhausted, InvalidPlanError, LLMConfig


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = LLMConfig(base_url="https://example.com/v1", model="test", api_key_env="TIER_TEST_KEY",
                       model_family="deepseek", model_release_date="2026-01-01", pricing_as_of="2026-10-21",
                       input_rmb_per_million=.5, output_rmb_per_million=2.,
                       price_source="https://example.com/prices", model_source="https://example.com/models",
                       verified=True, max_output_tokens=16384, request_policy="three_tier")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    monkeypatch.setenv("TIER_TEST_KEY", "fake-local-secret")
    return config, path, tmp_path / "run"


def responses(monkeypatch, outcomes):
    seen = []
    values = iter(outcomes)
    def post(config, payload, key):
        seen.append((config, payload))
        value = next(values)
        if isinstance(value, Exception):
            raise value
        finish, outputs = value
        body = {"model": "test", "usage": {"prompt_tokens": 100, "completion_tokens": outputs},
                "choices": [{"finish_reason": finish, "message": {
                    "content": json.dumps({"action": "stop", "reason": "Enough evidence."})}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", config.base_url))
    monkeypatch.setattr("superagent.planner.post_completion", post)
    return seen


def test_length_upgrade_uses_actual_school_control_and_keeps_both_charges(setup, monkeypatch):
    config, path, run = setup
    seen = responses(monkeypatch, [("length", 8192), ("stop", 200)])
    planner = CodingPlanner(path, run)
    with pytest.raises(InvalidPlanError):
        planner.propose({}, [], {})
    assert planner.propose({}, [], {})["action"] == "stop"
    assert [p["max_tokens"] for _, p in seen] == [8192, 16384]
    assert [c.request_deadline_seconds for c, _ in seen] == [240, 360]
    assert all(p["chat_template_kwargs"] == {"enable_thinking": False} for _, p in seen)
    assert all("reasoning_effort" not in p for _, p in seen)
    calls, cost, violated = _read_ledger(run / "llm_calls.jsonl", planner.config)
    assert float(cost) == pytest.approx((200 * .5 + 8392 * 2) / 1_000_000)
    assert len(calls) == 2 and not violated
    assert calls["llm-0001"]["start"]["max_output_tokens"] == 16384
    assert calls["llm-0001"]["start"]["request_parameters"]["max_tokens"] == 8192


def test_second_length_stops_without_third_request(setup, monkeypatch):
    _, path, run = setup
    seen = responses(monkeypatch, [("length", 8192), ("length", 16384)])
    planner = CodingPlanner(path, run)
    with pytest.raises(InvalidPlanError):
        planner.propose({}, [], {})
    with pytest.raises(AgentPlannerError, match="stop instead"):
        planner.propose({}, [], {})
    with pytest.raises(AgentPlannerError):
        planner.propose({}, [], {})
    assert len(seen) == 2
    assert planner.summary()["accounting_complete"]


def test_difficult_once_after_two_distinct_execution_failures(setup, monkeypatch):
    _, path, run = setup
    seen = responses(monkeypatch, [("stop", 100), ("stop", 100)])
    planner = CodingPlanner(path, run)
    history = [{"status": "failed", "candidate": {"algorithm": "generated_python", "source": s}}
               for s in ("a", "b")]
    planner.propose({}, history, {})
    planner.propose({}, history, {})
    assert [p["chat_template_kwargs"]["enable_thinking"] for _, p in seen] == [True, False]
    assert [c.request_deadline_seconds for c, _ in seen] == [600, 240]


def test_timeout_preserves_reservation_and_never_retries(setup, monkeypatch):
    _, path, run = setup
    seen = responses(monkeypatch, [RequestDeadlineExceeded("local deadline")])
    planner = CodingPlanner(path, run)
    with pytest.raises(AccountingError):
        planner.propose({}, [], {})
    with pytest.raises(AgentPlannerError):
        planner.propose({}, [], {})
    assert len(seen) == 1
    summary = planner.summary()
    assert summary["unknown_usage"] and summary["pending_reserved_rmb"] == pytest.approx(.040768)
    assert summary["accounted_cost_rmb"] == 0


def test_budget_denial_does_not_consume_difficult_allowance(setup, monkeypatch):
    _, path, run = setup
    seen = responses(monkeypatch, [])
    planner = CodingPlanner(path, run, budget_limit=.001)
    history = [{"status": "failed", "candidate": {"algorithm": "generated_python", "source": s}}
               for s in ("a", "b")]
    with pytest.raises(BudgetExhausted):
        planner.propose({}, history, {})
    assert not planner.request_policy.difficult_used and not seen


def test_actual_smaller_output_cap_enforced_and_reauditable(setup, monkeypatch):
    _, path, run = setup
    responses(monkeypatch, [("stop", 8193)])
    planner = CodingPlanner(path, run)
    with pytest.raises(AccountingError, match="reserved limits"):
        planner.propose({}, [], {})
    _, cost, violated = _read_ledger(run / "llm_calls.jsonl", planner.config)
    assert violated and cost > 0


@pytest.mark.parametrize("field,value", [("enable_thinking", "false"),
    ("request_deadline_seconds", 0), ("request_deadline_seconds", float("inf")),
    ("request_policy", "unknown"), ("max_output_tokens", 8192), ("reasoning_effort", "low")])
def test_invalid_policy_config_rejected(setup, field, value):
    config, _, _ = setup
    values = asdict(config)
    values[field] = value
    with pytest.raises(ValueError):
        LLMConfig(**values).validate()


def test_connection_probe_uses_default_tier_and_remains_a_single_call(setup, monkeypatch):
    config, path, run = setup
    seen = []
    def post(active, payload, key):
        seen.append((active, payload))
        assert active.request_policy is None
        assert active.max_output_tokens == 8192
        assert active.request_deadline_seconds == active.timeout_seconds == 240
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["max_tokens"] == 8192
        assert "reasoning_effort" not in payload
        return httpx.Response(200, request=httpx.Request("POST", config.base_url), json={
            "model": config.model, "usage": {"prompt_tokens": 100, "completion_tokens": 4},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]})
    monkeypatch.setattr("superagent.planner.post_completion", post)
    result = check_api(path, run)
    assert result["status"] == "passed" and len(seen) == 1
    assert result["usage"]["request_count"] == 1
    calls, _, violated = _read_ledger(run / "llm_calls.jsonl", LLMConfig.load(run / "llm_config.snapshot.json"))
    assert not violated
    assert calls["llm-0001"]["start"]["request_parameters"]["tier"] == "default"
    assert calls["llm-0001"]["start"]["max_output_tokens"] == 16384


def test_connection_probe_length_is_charged_without_rescue_retry(setup, monkeypatch):
    _, path, run = setup
    seen = responses(monkeypatch, [("length", 8192)])
    result = check_api(path, run)
    assert result["status"] == "failed" and len(seen) == 1
    assert result["usage"]["accounting_complete"]
    assert result["usage"]["accounted_cost_rmb"] > 0


def test_config_check_reports_real_default_separately_from_reservation(setup, monkeypatch):
    _, path, _ = setup
    seen = responses(monkeypatch, [])
    result = check_config(path)
    assert result["request_policy"] == "three_tier"
    assert result["output_reservation_ceiling"] == 16384
    assert result["default_request"]["max_output_tokens"] == 8192
    assert result["default_request"]["deadline_seconds"] == 240
    assert result["default_request"]["thinking"] is False
    assert result["network_request_sent"] is False and not seen


def test_catalog_planner_cannot_silently_ignore_three_tier_policy(setup, monkeypatch):
    _, path, run = setup
    seen = responses(monkeypatch, [])
    planner = LLMPlanner(path, run)
    with pytest.raises(AgentPlannerError, match="llm-code"):
        planner.propose({}, [], {})
    assert not seen and planner.summary()["request_count"] == 0


def test_run_rejects_wrong_mode_before_data_or_output(setup, monkeypatch):
    from superagent.runner import run as run_task
    _, path, output = setup
    responses(monkeypatch, [])
    with pytest.raises(ValueError, match="llm-code"):
        run_task(Path("nonexistent-task.json"), output, "llm", path)
    assert not output.exists()


def test_campaign_rejects_wrong_mode_without_output(setup, monkeypatch, tmp_path):
    from superagent.campaign import campaign
    from superagent.demo import make_demo
    _, path, output = setup
    responses(monkeypatch, [])
    make_demo(tmp_path / "data")
    task = tmp_path / "data/tabular_task.json"
    raw = json.loads(task.read_text())
    raw.pop("predict_path")
    task.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="llm-code"):
        campaign(task, [path], output, planner_mode="llm")
    assert not output.exists()
