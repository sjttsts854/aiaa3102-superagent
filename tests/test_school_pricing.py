import json
from dataclasses import asdict, replace

import pytest

from superagent.usage import AccountingError, BudgetLedger, LLMConfig


@pytest.fixture
def school_config():
    return LLMConfig(
        base_url="https://aigc-api.hkust-gz.edu.cn/api/v2",
        model="Qwen3.8-27B", accepted_response_models=["Qwen3.8-27B-w8a8"],
        api_key_env="AIGC_API_KEY", model_family="qwen",
        model_release_date="2026-08-14", pricing_as_of="2026-11-14",
        input_rmb_per_million=0.10, output_rmb_per_million=0.34,
        price_source="https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-model-list/",
        model_source="https://github.com/QwenLM/Qwen3.8", verified=True,
        pricing_basis="provider_current", response_format="prompt_only",
    )


def test_school_prices_are_calculated_and_labeled_on_every_row(tmp_path, school_config):
    ledger = BudgetLedger(school_config, tmp_path / "calls.jsonl")
    ledger.begin([{"role": "user", "content": "Return a JSON plan."}])
    ledger.settle({
        "model": "Qwen3.8-27B-w8a8",
        "usage": {"prompt_tokens": 10000, "completion_tokens": 1000,
                  "prompt_cache_hit_tokens": 9000},
    })
    summary = ledger.summary()
    assert summary["accounted_cost_rmb"] == pytest.approx(0.00134)
    assert summary["calculated_cost_rmb"] == pytest.approx(0.00134)
    assert summary["course_pricing_compliant"] is False
    assert summary["within_budget"] is True
    assert summary["budget_pricing_basis"] == "provider_current"
    assert summary["provider_billed_cost_rmb"] is None
    records = [json.loads(line) for line in ledger.log_path.read_text().splitlines()]
    for record in records:
        assert record["pricing_basis"] == "provider_current"
        assert record["pricing_as_of"] == "2026-11-14"
        assert record["price_source"] == school_config.price_source
        assert record["course_pricing_compliant"] is False
        assert "not a provider invoice" in record["cost_semantics"]
    assert records[0]["pricing"]["basis"] == "provider_current"
    assert records[-1]["input_tokens"] == 10000
    assert records[-1]["output_tokens"] == 1000


def test_old_configs_default_to_official_cutoff_and_json_mode(tmp_path, school_config):
    raw = asdict(school_config)
    raw.pop("pricing_basis")
    raw.pop("response_format")
    raw["pricing_as_of"] = "2026-10-21"
    path = tmp_path / "old_config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = LLMConfig.load(path)
    assert config.pricing_basis == "official_standard"
    assert config.response_format == "json_object"
    summary = BudgetLedger(config, tmp_path / "calls.jsonl").summary()
    assert summary["course_pricing_compliant"] is True
    assert summary["budget_pricing_basis"] == "official_standard"


@pytest.mark.parametrize("changes", [
    {"pricing_basis": "official_standard"},
    {"pricing_basis": "unknown"},
    {"pricing_basis": []},
    {"pricing_as_of": "2026-02-30"},
    {"pricing_as_of": "20261114"},
    {"pricing_as_of": "2026-W46-6"},
    {"pricing_as_of": None},
    {"model_release_date": "2026-10-22"},
    {"budget_rmb": 20.01},
    {"response_format": "unsupported"},
    {"response_format": []},
    {"verified": False},
])
def test_school_basis_does_not_bypass_validation(school_config, changes):
    with pytest.raises(ValueError):
        replace(school_config, **changes).validate()


@pytest.mark.parametrize("body", [
    {"model": "Qwen3.8-27B-w8a8"},
    {"model": "Qwen3.8-27B-w8a8", "usage": {"prompt_tokens": 0, "completion_tokens": 1}},
    {"model": "unverified-backend", "usage": {"prompt_tokens": 100, "completion_tokens": 1}},
])
def test_unresolved_school_usage_retains_basis_and_stops_requests(tmp_path, school_config, body):
    ledger = BudgetLedger(school_config, tmp_path / "calls.jsonl")
    ledger.begin([{"role": "user", "content": "JSON please"}])
    with pytest.raises(AccountingError):
        ledger.settle(body)
    with pytest.raises(AccountingError):
        ledger.begin([{"role": "user", "content": "Do not retry"}])
    summary = ledger.summary()
    assert summary["accounting_complete"] is False
    assert summary["within_budget"] is False
    assert summary["course_pricing_compliant"] is False
    record = json.loads(ledger.log_path.read_text().splitlines()[-1])
    assert record["event"] == "accounting_unresolved"
    assert record["response"] == body
    assert record["pricing_basis"] == "provider_current"
    assert record["pricing_as_of"] == "2026-11-14"
    assert record["calculated_cost_rmb"] is None
