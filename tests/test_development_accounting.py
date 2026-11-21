"""Synthetic development history only: no API calls or task/test data."""
from dataclasses import asdict
import json

import pytest

from superagent.campaign import _previous_costs
from superagent.development_accounting import collect_development_history
from superagent.usage import AccountingError, BudgetLedger, LLMConfig
from superagent.utils import file_hash, write_json


HASHES = {"train_path": "train-hash", "train_labels_path": "labels-hash",
          "validation_path": "validation-hash", "validation_labels_path": "validation-labels-hash"}
PRICING = {("official_standard", "2026-10-21")}


def make_run(path, *, failure="ReadTimeout", completed=True, zero=False, response=None):
    path.mkdir()
    config = LLMConfig(
        base_url="https://provider.example/v1", model="test-model", api_key_env="UNUSED_TEST_KEY",
        model_family="deepseek", model_release_date="2026-01-01", pricing_as_of="2026-10-21",
        input_rmb_per_million=1.0, output_rmb_per_million=2.0, budget_rmb=.1,
        price_source="https://provider.example/pricing", model_source="https://provider.example/model",
        verified=True, max_input_tokens=10000, max_output_tokens=100,
    )
    write_json(path / "llm_config.snapshot.json", asdict(config))
    write_json(path / "manifest.json", {"data_sha256": {k.removesuffix("_path"): v for k, v in HASHES.items()}})
    ledger = BudgetLedger(config, path / "llm_calls.jsonl")
    if not zero:
        if completed:
            ledger.begin([{"role": "user", "content": "synthetic known call"}])
            ledger.settle({"model": config.model, "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        if failure:
            ledger.begin([{"role": "user", "content": "synthetic failed call"}])
            ledger.unresolved(f"Transport or response decoding failure: {failure}", response=response,
                              diagnostic={"exception_type": failure, "http_status": 503})
    usage = ledger.summary()
    write_json(path / "usage_summary.json", usage)
    write_json(path / "summary.json", {"status": "partial_unresolved_accounting" if failure and not zero else "completed",
                                      "usage": usage})
    return path / "summary.json"


def make_campaign(path, *, runs=(), previous=(), charged=(), status="partial_accounting", pinned=()):
    path.parent.mkdir(exist_ok=True)
    write_json(path, {"status": status, "data_sha256": HASHES, "pricing_basis": "official_standard",
                      "pricing_as_of": "2026-10-21", "previous_campaigns": list(map(str, previous)),
                      "runs": [{"output": str(run.parent), "summary": json.loads(run.read_text())} for run in runs],
                      "charged_run_directories": list(map(str, charged)), "history_ledger_records": list(pinned)})
    return path


def collect(*paths):
    return collect_development_history(list(paths), HASHES, PRICING)


def change_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    write_json(path, value)


def change_ledger(path, mutate):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_nested_flattened_history_counts_once_without_changing_originals(tmp_path):
    pilot = make_run(tmp_path / "pilot", failure=None)
    run = make_run(tmp_path / "unknown")
    first = make_campaign(tmp_path / "first.json", runs=[run], previous=[pilot])
    second = make_campaign(tmp_path / "second.json", previous=[first, pilot], charged=[pilot.parent, run.parent],
                           status="development_completed_unresolved_accounting")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = collect(second)
    assert result["known_cost_rmb"] == pytest.approx(.00028)
    assert result["unresolved_reserved_rmb"] == pytest.approx(.0102)
    assert result["budget_allocation_rmb"] == pytest.approx(.01048)
    assert result["accounting_complete"] is False
    assert result["actual_total_cost_known"] is False
    assert result["course_pricing_compliant"] is False
    assert len(result["run_directories"]) == 2
    assert set(result["source_records"]) == set(map(str, [pilot, run, first, second]))
    pending, = result["unresolved_calls"]
    assert pending["actual_cost_rmb"] is pending["input_tokens"] is pending["output_tokens"] is None
    assert pending["ledger_sha256"] == file_hash(run.parent / "llm_calls.jsonl")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_duplicate_explicit_roots_and_overlapping_owners_are_rejected(tmp_path):
    run = make_run(tmp_path / "run")
    campaign = make_campaign(tmp_path / "campaign.json", runs=[run])
    with pytest.raises(ValueError, match="Duplicate explicit"):
        collect(run, run)
    with pytest.raises(ValueError, match="overlap"):
        collect(campaign, run)


def test_duplicate_campaign_run_entries_and_cycles_are_rejected(tmp_path):
    run = make_run(tmp_path / "run")
    campaign = make_campaign(tmp_path / "campaign.json", runs=[run, run])
    with pytest.raises(ValueError, match="Duplicate run"):
        collect(campaign)
    make_campaign(campaign, previous=[campaign])
    with pytest.raises(ValueError, match="cycle"):
        collect(campaign)


@pytest.mark.parametrize("field,value", [("reserved_rmb", .001), ("model_requested", "wrong-model")])
def test_original_reservation_and_model_must_match_snapshot(tmp_path, field, value):
    run = make_run(tmp_path / "run")
    change_ledger(run.parent / "llm_calls.jsonl", lambda rows: rows[2].update({field: value}))
    with pytest.raises(ValueError, match="configuration snapshot"):
        collect(run)


def test_known_cost_cannot_be_replaced_by_a_forged_total(tmp_path):
    run = make_run(tmp_path / "run")
    change_ledger(run.parent / "llm_calls.jsonl", lambda rows: rows[1].update(calculated_cost_rmb=0))
    with pytest.raises(ValueError, match="token arithmetic"):
        collect(run)


@pytest.mark.parametrize("response", [{"model": "wrong"}, {"model": "test-model"},
                                      {"model": "test-model", "usage": {"prompt_tokens": -1}}])
def test_wrong_model_missing_usage_and_invalid_usage_cannot_escape(tmp_path, response):
    run = make_run(tmp_path / "run", response=response)
    with pytest.raises(ValueError, match="Only documented transport"):
        collect(run)


@pytest.mark.parametrize("mutation", [lambda rows: rows[-1].update(diagnostic=None),
                                      lambda rows: rows[-1].update(reason="Response did not provide usage statistics."),
                                      lambda rows: rows.pop()])
def test_pending_or_unexplained_unknown_request_cannot_escape(tmp_path, mutation):
    run = make_run(tmp_path / "run")
    change_ledger(run.parent / "llm_calls.jsonl", mutation)
    with pytest.raises(ValueError, match="Only documented transport"):
        collect(run)


@pytest.mark.parametrize("failure", ["ReadTimeout", "ConnectError", "JSONDecodeError", "HTTPStatusError"])
def test_recognized_transport_and_decoding_failures_keep_original_reserves(tmp_path, failure):
    assert collect(make_run(tmp_path / "run", failure=failure))["unresolved_reserved_rmb"] == pytest.approx(.0102)


@pytest.mark.parametrize("failure", ["RequestDeadlineExceeded", "RequestWorkerError"])
def test_request_process_failures_block_strict_history_and_retain_development_reserves(tmp_path, failure):
    original = make_run(tmp_path / "original", failure=failure)
    original_files = {path: path.read_bytes() for path in original.parent.iterdir() if path.is_file()}

    with pytest.raises(ValueError, match="unresolved usage; matching reconciliation is required"):
        _previous_costs([original], [], HASHES, PRICING)

    history = collect(original)
    assert history["known_cost_rmb"] == pytest.approx(.00014)
    assert history["unresolved_reserved_rmb"] == pytest.approx(.0102)
    assert history["budget_allocation_rmb"] == pytest.approx(.01034)
    assert history["accounting_complete"] is False
    assert history["actual_total_cost_known"] is False
    pending, = history["unresolved_calls"]
    assert pending["actual_cost_rmb"] is pending["input_tokens"] is pending["output_tokens"] is None

    # A development continuation adds its own known cost without clearing,
    # settling or rewriting the earlier request's unresolved reservation.
    continued = make_run(tmp_path / "continued", failure=None)
    campaign = make_campaign(tmp_path / "continued_campaign.json", runs=[continued], previous=[original],
                             status="development_completed_unresolved_accounting",
                             pinned=history["ledger_records"])
    combined = collect(campaign)
    assert combined["known_cost_rmb"] == pytest.approx(.00028)
    assert combined["unresolved_reserved_rmb"] == pytest.approx(.0102)
    assert combined["budget_allocation_rmb"] == pytest.approx(.01048)
    assert combined["unresolved_calls"] == history["unresolved_calls"]
    assert combined["accounting_complete"] is False
    assert combined["actual_total_cost_known"] is False
    assert {path: path.read_bytes() for path in original_files} == original_files


def test_http_failure_requires_error_status(tmp_path):
    run = make_run(tmp_path / "run", failure="HTTPStatusError")
    change_ledger(run.parent / "llm_calls.jsonl", lambda rows: rows[-1]["diagnostic"].update(http_status=200))
    with pytest.raises(ValueError, match="valid recorded error"):
        collect(run)


def test_run_must_stop_at_unresolved_request(tmp_path):
    run = make_run(tmp_path / "run")
    def append_extra(rows):
        rows.extend([{**row, "call_id": "llm-0003"} for row in rows[-2:]])
    change_ledger(run.parent / "llm_calls.jsonl", append_extra)
    with pytest.raises(ValueError, match="continued with a pending or unresolved"):
        collect(run)


def test_run_cannot_start_another_request_while_one_is_pending(tmp_path):
    run = make_run(tmp_path / "run")
    change_ledger(run.parent / "llm_calls.jsonl", lambda rows: rows.insert(1, rows.pop(2)))
    with pytest.raises(ValueError, match="continued with a pending or unresolved"):
        collect(run)


def test_inherited_fingerprints_cannot_be_silently_replaced(tmp_path):
    run = make_run(tmp_path / "run")
    history = collect(run)
    campaign = make_campaign(tmp_path / "campaign.json", previous=[run], pinned=history["ledger_records"])
    assert collect(campaign)["unresolved_reserved_rmb"] == pytest.approx(.0102)
    ledger = run.parent / "llm_calls.jsonl"
    ledger.write_text(ledger.read_text().replace('"event": ', '"event":'), encoding="utf-8")
    with pytest.raises(ValueError, match="Inherited history fingerprints"):
        collect(campaign)


def test_valid_but_changed_configuration_breaks_pinned_fingerprint(tmp_path):
    run = make_run(tmp_path / "run")
    campaign = make_campaign(tmp_path / "campaign.json", previous=[run], pinned=collect(run)["ledger_records"])
    change_json(run.parent / "llm_config.snapshot.json", lambda c: c.update(budget_rmb=.2))
    with pytest.raises(ValueError, match="Inherited history fingerprints"):
        collect(campaign)


@pytest.mark.parametrize("status", ["running", "", "pending"])
def test_original_run_must_be_terminal(tmp_path, status):
    run = make_run(tmp_path / "run")
    change_json(run, lambda s: s.update(status=status))
    with pytest.raises(ValueError, match="terminal"):
        collect(run)


def test_missing_original_summary_is_rejected(tmp_path):
    run = make_run(tmp_path / "run")
    campaign = make_campaign(tmp_path / "campaign.json", runs=[run])
    run.unlink()
    with pytest.raises(FileNotFoundError):
        collect(campaign)


def test_only_explicit_zero_call_account_may_have_no_ledger(tmp_path):
    run = make_run(tmp_path / "run", zero=True)
    result = collect(run)
    assert result["known_cost_rmb"] == result["unresolved_reserved_rmb"] == 0
    change_json(run, lambda s: s["usage"].update(blocked=True))
    with pytest.raises(ValueError, match="zero-call"):
        collect(run)


def test_data_and_pricing_must_match(tmp_path):
    run = make_run(tmp_path / "run")
    with pytest.raises(ValueError, match="task data and pricing"):
        collect_development_history([run], {**HASHES, "train_path": "other"}, PRICING)
    with pytest.raises(ValueError, match="task data and pricing"):
        collect_development_history([run], HASHES, {("provider_current", "2026-11-14")})


def test_actual_reservation_limit_violation_cannot_escape(tmp_path):
    run = make_run(tmp_path / "run", zero=True)
    config = LLMConfig.load(run.parent / "llm_config.snapshot.json")
    ledger = BudgetLedger(config, run.parent / "llm_calls.jsonl")
    ledger.begin([{"role": "user", "content": "synthetic call"}])
    with pytest.raises(AccountingError):
        ledger.settle({"model": config.model, "usage": {"prompt_tokens": 10001, "completion_tokens": 20}})
    write_json(run, {"status": "partial_accounting_limits_violated", "usage": ledger.summary()})
    with pytest.raises(ValueError, match="limit violations"):
        collect(run)


def test_original_reservation_must_fit_remaining_run_budget(tmp_path):
    run = make_run(tmp_path / "run")
    change_json(run.parent / "llm_config.snapshot.json", lambda c: c.update(budget_rmb=.001))
    with pytest.raises(ValueError, match="reservation exceeded"):
        collect(run)
