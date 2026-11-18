"""Synthetic provider evidence only. No real provider/account/test data is used."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from superagent.reconciliation import reconcile_run_usage
from superagent.usage import AccountingError, BudgetLedger, LLMConfig
from superagent.utils import file_hash, write_json


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = LLMConfig(
        base_url="https://provider.example/v1", model="test-model", api_key_env="UNUSED_TEST_KEY",
        model_family="deepseek", model_release_date="2026-01-01", pricing_as_of="2026-10-21",
        input_rmb_per_million=1.0, output_rmb_per_million=2.0, budget_rmb=.1,
        price_source="https://provider.example/pricing", model_source="https://provider.example/model",
        verified=True, max_input_tokens=10000, max_output_tokens=100,
        accepted_response_models=["test-model-backend"],
    )
    write_json(run / "llm_config.snapshot.json", asdict(config))
    ledger = BudgetLedger(config, run / "llm_calls.jsonl")
    ledger.begin([{"role": "user", "content": "first synthetic call"}])
    ledger.settle({"id": "known-response", "model": config.model,
                   "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    ledger.begin([{"role": "user", "content": "second synthetic call"}])
    ledger.unresolved("Synthetic timeout")
    write_json(run / "usage_summary.json", ledger.summary())
    return run


def evidence_for(run, *, status="completed", call_id="llm-0002"):
    starts = [json.loads(row) for row in (run / "llm_calls.jsonl").read_text().splitlines()
              if json.loads(row)["event"] == "request_started"]
    start = next(row for row in starts if row["call_id"] == call_id)
    resolution = {"call_id": call_id, "requested_model": start["model_requested"],
                  "request_started_at": start["time_utc"], "status": status,
                  "provider_record_reference": f"synthetic-export-row-{call_id}"}
    if status == "completed":
        resolution.update(provider_model="test-model-backend", input_tokens=200, output_tokens=30)
    else:
        resolution["confirmation"] = "Synthetic provider statement explicitly confirms no execution."
    return {"schema_version": 1, "ledger_sha256": file_hash(run / "llm_calls.jsonl"),
            "config_sha256": file_hash(run / "llm_config.snapshot.json"),
            "provenance": {"kind": "provider_export", "reference": "synthetic-test-export",
                           "recorded_at": "2099-01-01T00:00:00+00:00", "reviewed_by": "synthetic-test-reviewer"},
            "resolutions": [resolution]}


def reconcile(tmp_path, run, evidence=None, name="reconciled"):
    path = tmp_path / f"{name}-evidence.json"
    write_json(path, evidence if evidence is not None else evidence_for(run))
    return reconcile_run_usage(run, path, tmp_path / name)


def test_reconciles_completed_tokens_without_modifying_originals(tmp_path, run_dir):
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    result = reconcile(tmp_path, run_dir)
    usage = result["usage"]
    assert usage["accounting_complete"] is True
    assert usage["resume_allowed"] is True
    assert usage["accounted_cost_rmb"] == pytest.approx(.00014 + .00026)
    assert usage["previous_recorded_cost_rmb"] == pytest.approx(.00014)
    assert usage["provider_billed_cost_rmb"] is None
    assert result["new_api_calls"] == 0
    assert result["status"] == "completed"
    assert result["evidence_sha256"] == file_hash(tmp_path / "reconciled/evidence.json")
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before
    assert json.loads((run_dir / "usage_summary.json").read_text())["blocked"] is True


def test_explicit_nonexecution_only_charges_prior_calls(tmp_path, run_dir):
    result = reconcile(tmp_path, run_dir, evidence_for(run_dir, status="not_executed"))
    assert result["usage"]["accounted_cost_rmb"] == pytest.approx(.00014)
    assert result["resolved_calls"][0]["calculated_cost_rmb"] == 0


@pytest.mark.parametrize("mutate,match", [
    (lambda e: e.update(ledger_sha256="changed"), "fingerprints"),
    (lambda e: e.update(config_sha256="changed"), "fingerprints"),
    (lambda e: e["resolutions"][0].update(call_id="llm-0001"), "unresolved call"),
    (lambda e: e["resolutions"][0].update(call_id="missing"), "unresolved call"),
    (lambda e: e["resolutions"][0].update(requested_model="other"), "model or request start"),
    (lambda e: e["resolutions"][0].update(provider_model="unverified-backend"), "verified pricing"),
    (lambda e: e["resolutions"][0].update(request_started_at="2026-01-01T00:00:00Z"), "start time"),
    (lambda e: e["resolutions"][0].update(input_tokens=0), "positive input_tokens"),
    (lambda e: e["resolutions"][0].update(input_tokens=True), "positive input_tokens"),
    (lambda e: e["resolutions"][0].update(output_tokens=-1), "output_tokens"),
    (lambda e: e["resolutions"][0].update(output_tokens=2.5), "output_tokens"),
    (lambda e: e["resolutions"][0].update(status="not_found"), "explicitly state"),
    (lambda e: e["resolutions"].append(e["resolutions"][0].copy()), "unique unresolved"),
    (lambda e: e["provenance"].update(reference=""), "nonempty"),
    (lambda e: e["provenance"].update(reviewed_by=""), "nonempty"),
    (lambda e: e["provenance"].update(kind="reservation_estimate"), "originate"),
    (lambda e: e["provenance"].update(recorded_at="2020-01-01T00:00:00Z"), "predates"),
    (lambda e: e["provenance"].update(recorded_at="2099-01-01T00:00:00"), "timezone"),
    (lambda e: e.update(api_key="must-not-be-imported"), "unsupported fields"),
])
def test_rejects_mismatched_missing_or_guessed_evidence(tmp_path, run_dir, mutate, match):
    evidence = evidence_for(run_dir)
    mutate(evidence)
    with pytest.raises(ValueError, match=match):
        reconcile(tmp_path, run_dir, evidence)
    assert not (tmp_path / "reconciled").exists()


def test_nonexecution_needs_confirmation_and_forbids_token_override(tmp_path, run_dir):
    evidence = evidence_for(run_dir, status="not_executed")
    evidence["resolutions"][0]["confirmation"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        reconcile(tmp_path, run_dir, evidence)
    evidence["resolutions"][0].update(confirmation="explicit provider confirmation", input_tokens=0)
    with pytest.raises(ValueError, match="unsupported fields"):
        reconcile(tmp_path, run_dir, evidence)


def test_ledger_change_after_evidence_is_detected(tmp_path, run_dir):
    evidence = evidence_for(run_dir)
    with (run_dir / "llm_calls.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="fingerprints"):
        reconcile(tmp_path, run_dir, evidence)


def test_recomputes_completed_cost_instead_of_accepting_a_forged_total(tmp_path, run_dir):
    path = run_dir / "llm_calls.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[1]["calculated_cost_rmb"] = 0
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(ValueError, match="token arithmetic"):
        reconcile(tmp_path, run_dir)


def test_missing_evidence_and_repeated_output_are_rejected(tmp_path, run_dir):
    with pytest.raises(FileNotFoundError):
        reconcile_run_usage(run_dir, tmp_path / "missing.json", tmp_path / "out")
    reconcile(tmp_path, run_dir)
    with pytest.raises(ValueError, match="already exists"):
        reconcile(tmp_path, run_dir)


@pytest.mark.parametrize("field,value", [("input_tokens", 10001), ("output_tokens", 101), ("input_tokens", 200000)])
def test_recorded_limit_violations_remain_blocked(tmp_path, run_dir, field, value):
    evidence = evidence_for(run_dir)
    evidence["resolutions"][0][field] = value
    result = reconcile(tmp_path, run_dir, evidence)
    assert result["status"] == "partial_accounting_limits_violated"
    assert result["usage"]["accounting_complete"] is True
    assert result["usage"]["blocked"] is True
    assert result["usage"]["resume_allowed"] is False
    assert result["usage"]["accounted_cost_rmb"] > 0


def add_another_unresolved_call(run):
    path = run / "llm_calls.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    # Legacy fixture with two unmatched requests; no real API or ledger reset.
    extra = [{**row, "call_id": "llm-0003"} for row in rows[-2:]]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows + extra))


def test_partial_evidence_preserves_unresolved_reservation(tmp_path, run_dir):
    add_another_unresolved_call(run_dir)
    evidence = evidence_for(run_dir)
    result = reconcile(tmp_path, run_dir, evidence)
    assert result["status"] == "partial_unresolved_accounting"
    assert result["usage"]["accounting_complete"] is False
    assert result["usage"]["blocked"] is True
    assert result["unresolved_call_ids"] == ["llm-0003"]
    assert result["usage"]["pending_reserved_rmb"] == pytest.approx(.0102)
    evidence["resolutions"] += evidence_for(run_dir, status="not_executed", call_id="llm-0003")["resolutions"]
    second = reconcile(tmp_path, run_dir, evidence, name="fully-reconciled")
    assert second["usage"]["accounting_complete"] is True
    assert second["usage"]["request_count"] == 3
    assert second["usage"]["pending_reserved_rmb"] == 0


def test_provider_record_cannot_be_reused_for_two_calls(tmp_path, run_dir):
    add_another_unresolved_call(run_dir)
    evidence = evidence_for(run_dir)
    evidence["resolutions"] += evidence_for(run_dir, call_id="llm-0003")["resolutions"]
    evidence["resolutions"][1]["provider_record_reference"] = evidence["resolutions"][0]["provider_record_reference"]
    with pytest.raises(ValueError, match="distinct provider record"):
        reconcile(tmp_path, run_dir, evidence)


def test_prior_token_limit_violation_is_not_erased(tmp_path, run_dir):
    config = LLMConfig.load(run_dir / "llm_config.snapshot.json")
    other = tmp_path / "limit-run"
    other.mkdir()
    write_json(other / "llm_config.snapshot.json", asdict(config))
    ledger = BudgetLedger(config, other / "llm_calls.jsonl")
    ledger.begin([{"role": "user", "content": "known overrun"}])
    with pytest.raises(AccountingError):
        ledger.settle({"model": config.model, "usage": {"prompt_tokens": 10001, "completion_tokens": 0}})
    # Add a historical unmatched request without changing the overrun record.
    rows = [json.loads(x) for x in (run_dir / "llm_calls.jsonl").read_text().splitlines()][-2:]
    for row in rows:
        row["accounted_cost_rmb"] = .010001
    with (other / "llm_calls.jsonl").open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result = reconcile(tmp_path, other)
    assert result["usage"]["accounting_complete"] is True
    assert result["usage"]["resume_allowed"] is False
    assert result["status"] == "partial_accounting_limits_violated"
