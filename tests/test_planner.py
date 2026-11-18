import json
from dataclasses import replace

import httpx
import pytest

from superagent.planner import LLMPlanner, OfflinePlanner
from superagent.usage import AccountingError, BudgetExhausted, BudgetLedger, InvalidPlanError, LLMConfig


@pytest.fixture
def config():
    return LLMConfig(
        base_url="https://example.com/v1", model="test-fixed-model",
        api_key_env="SUPERAGENT_TEST_KEY", model_family="deepseek",
        model_release_date="2026-10-01", pricing_as_of="2026-10-21",
        input_rmb_per_million=2.0, output_rmb_per_million=8.0,
        price_source="https://example.com/pricing", model_source="https://example.com/release",
        verified=True,
    )


def make_planner(tmp_path, monkeypatch, config):
    from dataclasses import asdict
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    monkeypatch.setenv(config.api_key_env, "secret-test-key")
    return LLMPlanner(path, tmp_path / "run")


def response(content='{"action":"stop","reason":"done"}', usage=None):
    return {
        "id": "response-1", "model": "test-fixed-model",
        "usage": usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 20},
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
    }


def transport(body):
    return httpx.Response(200, json=body, request=httpx.Request("POST", "https://example.com/v1/chat/completions"))


def test_budget_preflight_sends_nothing(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, replace(config, budget_rmb=0.001))
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("Request must not be sent."))
    with pytest.raises(BudgetExhausted):
        planner.propose({}, [], {})
    assert planner.summary()["request_count"] == 0


def test_config_snapshot_preserves_settings_without_secret(tmp_path, monkeypatch, config):
    from dataclasses import asdict
    planner = make_planner(tmp_path, monkeypatch, config)
    snapshot = (tmp_path / "run" / "llm_config.snapshot.json").read_text(encoding="utf-8")
    assert json.loads(snapshot) == asdict(config)
    assert "SUPERAGENT_TEST_KEY" in snapshot
    assert "secret-test-key" not in snapshot
    assert planner.summary()["within_budget"] is True


def test_cache_discounts_are_not_applied(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, config)
    body = response(usage={"prompt_tokens": 1000, "completion_tokens": 100,
                           "prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 100})
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: transport(body))
    assert planner.propose({}, [], {})["action"] == "stop"
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(0.0028)
    records = [json.loads(line) for line in planner.ledger.log_path.read_text().splitlines()]
    assert records[-1]["input_tokens"] == 1000
    assert records[0]["messages"]
    assert "secret-test-key" not in planner.ledger.log_path.read_text()


def test_invalid_plan_still_charged(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, config)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: transport(response("not json")))
    with pytest.raises(InvalidPlanError):
        planner.propose({}, [], {})
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(0.00036)
    assert planner.summary()["accounting_complete"] is True


@pytest.mark.parametrize("content", [
    '{"action":[],"reason":"invalid"}',
    '{"action":"experiment","candidate":{"algorithm":"unlisted","params":{}},"reason":"invalid"}',
    '{"action":"stop"}',
])
def test_invalid_schema_still_charged(tmp_path, monkeypatch, config, content):
    planner = make_planner(tmp_path, monkeypatch, config)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: transport(response(content)))
    with pytest.raises(InvalidPlanError):
        planner.propose({}, [], {})
    assert planner.summary()["accounted_cost_rmb"] > 0


def test_context_bound_rejected_before_request(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, config)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("Oversized input must not be sent."))
    with pytest.raises(Exception, match="input-token bound"):
        planner.propose({"large": "x" * 16000}, [], {})
    assert planner.summary()["request_count"] == 0


def test_missing_usage_blocks_all_later_calls(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, config)
    body = response()
    body.pop("usage")
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: transport(body))
    with pytest.raises(AccountingError):
        planner.propose({}, [], {})
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("No retry with unknown usage."))
    with pytest.raises(AccountingError):
        planner.propose({}, [], {})
    assert planner.summary()["unknown_usage"] is True
    assert planner.summary()["pending_reserved_rmb"] > 0
    assert planner.summary()["accounting_complete"] is False
    assert planner.summary()["within_budget"] is False


def test_zero_input_usage_is_unknown_and_blocks(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, config)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: transport(
        response(usage={"prompt_tokens": 0, "completion_tokens": 20})
    ))
    with pytest.raises(AccountingError):
        planner.propose({}, [], {})
    assert planner.summary()["unknown_usage"] is True
    assert planner.summary()["within_budget"] is False
    assert planner.summary()["blocked"] is True


def test_zero_output_usage_is_allowed(tmp_path, config):
    ledger = BudgetLedger(config, tmp_path / "calls.jsonl")
    ledger.begin([{"role": "user", "content": "JSON"}])
    ledger.settle(response(usage={"prompt_tokens": 100, "completion_tokens": 0}))
    assert ledger.summary()["accounting_complete"] is True
    assert ledger.summary()["accounted_cost_rmb"] == pytest.approx(0.0002)


@pytest.mark.parametrize("returned_model", [None, "unverified-other-model"])
def test_model_mismatch_is_preserved_without_guessing_price(tmp_path, monkeypatch, config, returned_model):
    planner = make_planner(tmp_path, monkeypatch, config)
    body = response()
    body["model"] = returned_model
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: transport(body))
    with pytest.raises(AccountingError):
        planner.propose({}, [], {})
    records = [json.loads(line) for line in planner.ledger.log_path.read_text().splitlines()]
    assert records[-1]["response"] == body
    assert records[-1]["actual_cost_rmb"] is None
    assert planner.summary()["accounted_cost_rmb"] == 0
    assert planner.summary()["unknown_usage"] is True
    assert planner.summary()["blocked"] is True


def test_explicitly_verified_response_alias_is_accounted(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, replace(config, accepted_response_models=["verified-provider-alias"]))
    body = response()
    body["model"] = "verified-provider-alias"
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: transport(body))
    assert planner.propose({}, [], {})["action"] == "stop"
    assert planner.summary()["accounting_complete"] is True
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(0.00036)


def test_timeout_is_auditable_and_not_free(tmp_path, monkeypatch, config):
    planner = make_planner(tmp_path, monkeypatch, config)
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("simulated")
    monkeypatch.setattr(httpx, "post", timeout)
    with pytest.raises(AccountingError):
        planner.propose({}, [], {})
    records = [json.loads(line) for line in planner.ledger.log_path.read_text().splitlines()]
    assert records[-1]["event"] == "accounting_unresolved"
    assert records[-1]["actual_cost_rmb"] is None
    assert planner.summary()["blocked"]


@pytest.mark.parametrize("changes", [
    {"verified": False}, {"model_family": "other"},
    {"model_release_date": "2026-10-22"}, {"pricing_as_of": "2026-11-01"},
    {"budget_rmb": 20.01}, {"input_rmb_per_million": 0},
    {"output_rmb_per_million": float("nan")}, {"max_output_tokens": True},
    {"accepted_response_models": "not-a-list"}, {"accepted_response_models": [""]},
    {"accepted_response_models": [123]},
])
def test_invalid_config_is_rejected(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes).validate()


def test_provider_overrun_is_charged_then_blocked(tmp_path, config):
    ledger = BudgetLedger(config, tmp_path / "calls.jsonl")
    ledger.begin([{"role": "user", "content": "JSON please"}])
    with pytest.raises(AccountingError):
        ledger.settle(response(usage={"prompt_tokens": 16001, "completion_tokens": 20}))
    assert ledger.summary()["accounted_cost_rmb"] > 0
    assert ledger.summary()["blocked"]
    assert not ledger.summary()["unknown_usage"]
    assert ledger.summary()["within_budget"] is True


def test_existing_ledger_cannot_reset_budget(tmp_path, config):
    path = tmp_path / "calls.jsonl"
    ledger = BudgetLedger(config, path)
    ledger.begin([{"role": "user", "content": "JSON"}])
    with pytest.raises(AccountingError):
        BudgetLedger(config, path)


def test_actual_budget_overrun_is_visible(tmp_path, config):
    ledger = BudgetLedger(config, tmp_path / "calls.jsonl")
    ledger.begin([{"role": "user", "content": "JSON"}])
    with pytest.raises(AccountingError):
        ledger.settle(response(usage={"prompt_tokens": 10000001, "completion_tokens": 20}))
    assert ledger.summary()["accounted_cost_rmb"] > config.budget_rmb
    assert ledger.summary()["within_budget"] is False
    assert ledger.summary()["accounting_complete"] is True


def test_offline_adapts_to_validation_winner():
    planner = OfflinePlanner()
    catalog = {"logistic_regression": {}, "random_forest": {}}
    history = [
        {"candidate": {"algorithm": "logistic_regression", "params": {}}, "status": "success", "score": 0.9},
        {"candidate": {"algorithm": "random_forest", "params": {}}, "status": "success", "score": 0.6},
    ]
    plan = planner.propose({}, history, catalog)
    assert plan["candidate"] == {"algorithm": "logistic_regression", "params": {"C": 0.1}}
    assert planner.summary()["is_llm_evaluation"] is False
