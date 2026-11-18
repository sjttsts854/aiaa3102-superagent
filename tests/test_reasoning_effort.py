"""Reasoning controls use fake transport; no school or provider API is called."""
import importlib
import json
from dataclasses import asdict, replace

import httpx
import pytest

from superagent.coding_planner import CodingPlanner
from superagent.planner import LLMPlanner
from superagent.usage import InvalidPlanError, LLMConfig


@pytest.fixture
def config():
    return LLMConfig(
        base_url="https://provider.example/v1", model="test-reasoning-model",
        api_key_env="REASONING_EFFORT_TEST_KEY", model_family="deepseek",
        model_release_date="2026-01-01", pricing_as_of="2026-10-21",
        input_rmb_per_million=2., output_rmb_per_million=8.,
        price_source="https://provider.example/prices", model_source="https://provider.example/model",
        verified=True,
    )


def make_planner(tmp_path, monkeypatch, config, planner_type=LLMPlanner, omit_field=False):
    raw = asdict(config)
    if omit_field:
        raw.pop("reasoning_effort")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv(config.api_key_env, "fake-secret-for-test-only")
    return planner_type(path, tmp_path / "run")


def install_transport(monkeypatch, config, content='{"action":"stop","reason":"done"}'):
    payloads = []
    body = {"model": config.model, "usage": {
        "prompt_tokens": 100, "completion_tokens": 80,
        "completion_tokens_details": {"reasoning_tokens": 60},
    }, "choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    def fake_post(received_config, payload, api_key):
        assert received_config == config and api_key == "fake-secret-for-test-only"
        payloads.append(payload)
        return httpx.Response(200, json=body, request=httpx.Request("POST", config.base_url))

    monkeypatch.setattr(importlib.import_module("superagent.planner"), "post_completion", fake_post)
    return payloads, body


@pytest.mark.parametrize("family", ["deepseek", "qwen"])
@pytest.mark.parametrize("omit_field", [True, False])
def test_default_none_preserves_legacy_payload(tmp_path, monkeypatch, config, family, omit_field):
    config = replace(config, model_family=family)
    planner = make_planner(tmp_path, monkeypatch, config, omit_field=omit_field)
    payloads, _ = install_transport(monkeypatch, config)
    messages = [{"role": "user", "content": "Return one JSON object."}]
    assert planner._request_json(messages) == {"action": "stop", "reason": "done"}
    assert planner.config.reasoning_effort is None
    assert payloads == [{"model": config.model, "messages": messages, "max_tokens": config.max_output_tokens,
                         "stream": False, "response_format": {"type": "json_object"}}]


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_explicit_deepseek_effort_is_forwarded_and_snapshotted(tmp_path, monkeypatch, config, effort):
    config = replace(config, reasoning_effort=effort)
    planner = make_planner(tmp_path, monkeypatch, config)
    payloads, body = install_transport(monkeypatch, config)
    assert planner.propose({}, [], {}) == {"action": "stop", "reason": "done"}
    assert payloads[0]["reasoning_effort"] == effort
    assert payloads[0]["max_tokens"] == 1000
    assert payloads[0]["response_format"] == {"type": "json_object"}
    snapshot = (tmp_path / "run/llm_config.snapshot.json").read_text(encoding="utf-8")
    assert json.loads(snapshot) == asdict(config)
    assert LLMConfig.load(tmp_path / "run/llm_config.snapshot.json") == config
    assert "fake-secret-for-test-only" not in snapshot
    # The reported completion total already includes reasoning: charge it once.
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.00084)
    assert planner.summary()["accounting_complete"] is True
    assert planner.summary()["pending_reserved_rmb"] == 0
    rows = [json.loads(line) for line in planner.ledger.log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["reserved_rmb"] == pytest.approx(.04)
    assert rows[-1]["input_tokens"] == 100 and rows[-1]["output_tokens"] == 80
    assert rows[-1]["response"] == body


@pytest.mark.parametrize("value", [False, True, 0, 1, 0.5, [], {}, "", "medium", "LOW", "low "])
def test_invalid_effort_is_rejected_during_config_load(tmp_path, config, value):
    path = tmp_path / "bad-config.json"
    path.write_text(json.dumps({**asdict(config), "reasoning_effort": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="reasoning_effort"):
        LLMConfig.load(path)


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_explicit_effort_is_not_silently_sent_to_other_family(config, effort):
    with pytest.raises(ValueError, match="only supported for DeepSeek"):
        replace(config, model_family="qwen", reasoning_effort=effort).validate()


def test_code_planner_inherits_low_effort_without_changing_paid_error_handling(tmp_path, monkeypatch, config):
    config = replace(config, reasoning_effort="low")
    planner = make_planner(tmp_path, monkeypatch, config, planner_type=CodingPlanner)
    payloads, body = install_transport(monkeypatch, config, content="not complete JSON")
    with pytest.raises(InvalidPlanError):
        planner.propose({"data_type": "tabular"}, [], {})
    assert payloads[0]["reasoning_effort"] == "low"
    assert planner.summary()["request_count"] == 1
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.00084)
    assert planner.summary()["accounting_complete"] is True
    rows = [json.loads(line) for line in planner.ledger.log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["response"] == body
