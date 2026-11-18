"""End-to-end accounting imports use synthetic records and a mocked runner."""
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from superagent.data import load_task
from superagent.demo import make_demo
from superagent.reconciliation import reconcile_run_usage, verify_reconciliation
from superagent.usage import BudgetLedger, LLMConfig
from superagent.utils import file_hash, write_json


module = importlib.import_module("superagent.campaign")


@pytest.fixture
def context(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    task_path = tmp_path / "data/tabular_task.json"
    raw = json.loads(task_path.read_text())
    raw.pop("predict_path")
    write_json(task_path, raw)
    task = load_task(task_path)
    hashes = {"train_path": file_hash(task.train_path)}
    config = LLMConfig(
        base_url="https://provider.example/v1", model="test-model", api_key_env="UNUSED",
        model_family="deepseek", model_release_date="2026-01-01", pricing_as_of="2026-10-21",
        input_rmb_per_million=1., output_rmb_per_million=2., budget_rmb=1,
        price_source="https://provider.example/prices", model_source="https://provider.example/model",
        verified=True, max_input_tokens=10000, max_output_tokens=100,
    )
    config_path = tmp_path / "config.json"
    write_json(config_path, asdict(config))

    def make_run(folder, unresolved=False, score=.99, budget=None):
        folder.mkdir(parents=True)
        cfg = replace(config, budget_rmb=budget) if budget is not None else config
        write_json(folder / "llm_config.snapshot.json", asdict(cfg))
        write_json(folder / "manifest.json", {"data_sha256": {"train": hashes["train_path"]}})
        ledger = BudgetLedger(cfg, folder / "llm_calls.jsonl")
        ledger.begin([{"role": "user", "content": "synthetic call"}])
        ledger.settle({"id": "known", "model": cfg.model, "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        if unresolved:
            ledger.begin([{"role": "user", "content": "synthetic timeout"}])
            ledger.unresolved("Synthetic timeout")
        summary = {"status": "partial_unresolved_accounting" if unresolved else "completed",
                   "usage": ledger.summary(), "llm_requests": ledger.calls,
                   "best_validation_score": score, "selected_candidate": {"algorithm": "random_forest", "params": {}},
                   "model": str(folder / "model.joblib"), "source_sha256": "test-source"}
        write_json(folder / "summary.json", summary)
        write_json(folder / "usage_summary.json", ledger.summary())
        return summary

    new_budgets = []
    def mock_run(task_path, output, *args, budget_limit):
        new_budgets.append(budget_limit)
        summary = make_run(output, score=.6, budget=budget_limit)
        summary["valid_llm_decisions"] = 1
        write_json(output / "summary.json", summary)
        return summary
    monkeypatch.setattr(module, "run", mock_run)
    return {"task": task_path, "hashes": hashes, "config": config_path,
            "make_run": make_run, "budgets": new_budgets, "tmp": tmp_path}


def make_proof(context, run, name="proof", record=None):
    rows = [json.loads(line) for line in (run / "llm_calls.jsonl").read_text().splitlines()]
    start = next(row for row in rows if row["event"] == "request_started" and row["call_id"] == "llm-0002")
    evidence = {"schema_version": 1, "ledger_sha256": file_hash(run / "llm_calls.jsonl"),
                "config_sha256": file_hash(run / "llm_config.snapshot.json"),
                "provenance": {"kind": "provider_export", "reference": "synthetic provider export",
                               "recorded_at": "2099-01-01T00:00:00Z", "reviewed_by": "synthetic reviewer"},
                "resolutions": [{"call_id": "llm-0002", "requested_model": "test-model",
                                 "request_started_at": start["time_utc"], "status": "completed",
                                 "provider_record_reference": record or name, "provider_model": "test-model",
                                 "input_tokens": 200, "output_tokens": 30}]}
    path = context["tmp"] / f"{name}-evidence.json"
    write_json(path, evidence)
    output = context["tmp"] / name
    reconcile_run_usage(run, path, output)
    return output / "reconciliation.json"


def make_past_campaign(context):
    root = context["tmp"] / "past"
    known, unresolved = root / "planner_01", root / "planner_02"
    first = context["make_run"](known)
    second = context["make_run"](unresolved, unresolved=True)
    past = {"status": "partial_accounting", "data_sha256": context["hashes"],
            "pricing_basis": "official_standard", "pricing_as_of": "2026-10-21",
            "accounting_complete": False, "previous_cost_rmb": .02,
            "cumulative_cost_rmb": .02028, "runs": [
                {"model": "test-model", "output": str(known), "summary": first},
                {"model": "test-model", "output": str(unresolved), "summary": second}],
            "previous_campaigns": []}
    write_json(root / "campaign.json", past)
    return root / "campaign.json", unresolved


def continue_campaign(context, previous, proofs=None, name="next"):
    return module.campaign(context["task"], [context["config"]], context["tmp"] / name,
                           previous=previous, reconciliations=proofs or [])


def test_partial_campaign_recomputes_cost_and_preserves_training_outcome(context):
    past, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    original = {p: p.read_bytes() for p in past.parent.rglob("*.json*")}
    result = continue_campaign(context, [past], [proof])
    assert result["previous_cost_rmb"] == pytest.approx(.02 + .00014 + .0004)
    assert context["budgets"] == pytest.approx([1 - .02054])
    assert result["cumulative_cost_rmb"] == pytest.approx(.02068)
    assert result["selection"]["validation_score"] == .6  # never import old .99 selection
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert result["reconciliations"][0]["sha256"] == file_hash(proof)
    assert str(pending) in result["charged_run_directories"]


def test_standalone_partial_run_can_only_carry_reconciled_cost(context):
    pending = context["tmp"] / "standalone"
    context["make_run"](pending, unresolved=True)
    proof = make_proof(context, pending)
    with pytest.raises(ValueError, match="unresolved"):
        continue_campaign(context, [pending / "summary.json"])
    result = continue_campaign(context, [pending / "summary.json"], [proof])
    assert result["previous_cost_rmb"] == pytest.approx(.0004)
    assert json.loads((pending / "summary.json").read_text())["status"] == "partial_unresolved_accounting"


def test_reconciliation_validation_is_read_only_and_rejects_tampered_total(context):
    _, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    before = set(context["tmp"].rglob("*"))
    assert verify_reconciliation(proof)["usage"]["accounted_cost_rmb"] == pytest.approx(.0004)
    assert set(context["tmp"].rglob("*")) == before
    body = json.loads(proof.read_text())
    body["usage"]["accounted_cost_rmb"] = 0
    write_json(proof, body)
    with pytest.raises(ValueError, match="recomputed"):
        verify_reconciliation(proof)


@pytest.mark.parametrize("target", ["ledger", "config", "evidence"])
def test_changed_pinned_inputs_block_continuation(context, target):
    past, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    path = {"ledger": pending / "llm_calls.jsonl", "config": pending / "llm_config.snapshot.json",
            "evidence": proof.parent / "evidence.json"}[target]
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="fingerprint|recomputed"):
        continue_campaign(context, [past], [proof])
    assert context["budgets"] == []


def test_following_campaign_revalidates_inherited_sidecar(context):
    past, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    continue_campaign(context, [past], [proof])
    next_path = context["tmp"] / "next/campaign.json"
    result = continue_campaign(context, [next_path], name="third")
    assert result["previous_cost_rmb"] == pytest.approx(.02068)
    assert len(result["reconciliations"]) == 1
    with (pending / "llm_calls.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="fingerprints"):
        continue_campaign(context, [context["tmp"] / "third/campaign.json"], name="fourth")


@pytest.mark.parametrize("duplicate", ["sidecar", "run", "ledger_copy", "provider_record"])
def test_double_counting_or_reused_provider_evidence_is_rejected(context, duplicate):
    past, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    previous, proofs = [past], [proof]
    if duplicate == "sidecar":
        proofs.append(proof)
    elif duplicate == "run":
        previous.append(pending / "summary.json")
    else:
        another = context["tmp"] / "another"
        if duplicate == "ledger_copy":
            shutil.copytree(pending, another)
        else:
            context["make_run"](another, unresolved=True)
        other_proof = make_proof(context, another, name="proof2", record="proof" if duplicate == "provider_record" else "other-provider-record")
        previous.append(another / "summary.json")
        proofs.append(other_proof)
    with pytest.raises(ValueError, match="Duplicate|duplicate|Provider records"):
        continue_campaign(context, previous, proofs)
    assert context["budgets"] == []


def test_running_history_cannot_be_authorized_by_reconciliation(context):
    past, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    value = json.loads(past.read_text())
    value["status"] = "running"
    write_json(past, value)
    with pytest.raises(ValueError, match="unfinished"):
        continue_campaign(context, [past], [proof])


def test_all_nested_calls_must_be_settled_not_just_one(context):
    past, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    another = past.parent / "planner_03"
    summary = context["make_run"](another, unresolved=True)
    value = json.loads(past.read_text())
    value["runs"].append({"model": "test-model", "output": str(another), "summary": summary})
    value["cumulative_cost_rmb"] = .02042
    write_json(past, value)
    with pytest.raises(ValueError, match="blocked|unresolved"):
        continue_campaign(context, [past], [proof])


def test_wrong_task_and_unmatched_proof_are_rejected(context):
    past, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    with pytest.raises(ValueError, match="does not match"):
        continue_campaign(context, [], [proof])
    manifest = pending / "manifest.json"
    write_json(manifest, {"data_sha256": {"train": "other-training-file"}})
    with pytest.raises(ValueError, match="match task data"):
        continue_campaign(context, [past], [proof])


def test_cli_exposes_offline_reconciliation_and_campaign_option(context, monkeypatch, capsys):
    cli = importlib.import_module("superagent.__main__")
    _, pending = make_past_campaign(context)
    proof = make_proof(context, pending)
    evidence = context["tmp"] / "proof-evidence.json"
    monkeypatch.setattr(sys, "argv", ["superagent", "reconcile-usage", "--run", str(pending),
                                    "--evidence", str(evidence), "--output", str(context["tmp"] / "cli-proof")])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["usage"]["accounting_complete"] is True
    monkeypatch.setattr(sys, "argv", ["superagent", "campaign", "--task", str(context["task"]),
                                    "--llm-configs", str(context["config"]), "--output", str(context["tmp"] / "cli-next"),
                                    "--previous", str(pending / "summary.json"), "--reconciliations", str(proof)])
    assert cli.main() == 0
    capsys.readouterr()
    result = json.loads((context["tmp"] / "cli-next/campaign.json").read_text())
    assert result["previous_cost_rmb"] == pytest.approx(.0004)
