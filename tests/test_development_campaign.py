"""Campaign orchestration checks; ledger authenticity is tested separately.

The collector is replaced at its boundary so these tests exercise real campaign
budgeting and selection without API requests or course test-set data.
"""
import copy
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from superagent.demo import make_demo
from superagent.usage import LLMConfig


campaign_module = importlib.import_module("superagent.campaign")
history_module = importlib.import_module("superagent.development_accounting")


@pytest.fixture
def campaign_task(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    task = tmp_path / "data" / "tabular_task.json"
    document = json.loads(task.read_text(encoding="utf-8"))
    document.pop("predict_path")
    task.write_text(json.dumps(document), encoding="utf-8")
    base = LLMConfig.load(Path(__file__).parents[1] / "config/hkust-deepseek-flash.json")
    monkeypatch.setattr(campaign_module.LLMConfig, "load", lambda path: replace(
        base, model=Path(path).stem, budget_rmb=1.0))
    return task


def history(known=0, reserved=0, unresolved=None, sources=(), runs=()):
    return {
        "known_cost_rmb": known,
        "unresolved_reserved_rmb": reserved,
        "unresolved_calls": copy.deepcopy(unresolved or []),
        "source_records": [str(path.resolve()) for path in sources],
        "run_directories": [str(path.resolve()) for path in runs],
        "ledger_records": [],
    }


def unresolved_call(directory, reserve):
    return {"run_directory": str(directory.resolve()), "call_id": 2,
            "error_type": "ReadTimeout", "reserved_rmb": reserve,
            "actual_cost_rmb": None}


def run_summary(output, *, score=.7, cost=.1, complete=True,
                status=None, blocked=False):
    summary = {
        "status": status or ("completed" if complete else "partial_unresolved_accounting"),
        "usage": {"accounted_cost_rmb": cost, "accounting_complete": complete,
                  "unknown_usage": not complete, "blocked": blocked},
        "llm_requests": 2, "valid_llm_decisions": 1, "best_validation_score": score,
        "selected_candidate": {"algorithm": "random_forest", "params": {}},
        "model": str(output / "model.joblib"), "source_sha256": "same-source",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def test_development_deducts_known_and_reserved_preserves_unknown_and_selects_only_new_runs(
        campaign_task, tmp_path, monkeypatch):
    old_directory = tmp_path / "old_run"
    run_summary(old_directory, score=.99, cost=.2, complete=False)
    old_summary = old_directory / "summary.json"
    original_bytes = old_summary.read_bytes()
    original_unknown = unresolved_call(old_directory, .15)
    collector_calls = []

    def collect(paths, data_hashes, pricing):
        collector_calls.append(paths)
        assert set(data_hashes) == {"train_path"}
        assert len(pricing) == 1
        return history(.2, .15, [original_unknown], [old_summary], [old_directory])

    monkeypatch.setattr(history_module, "collect_development_history", collect)
    limits = []

    def run(task, output, mode, config, *timeouts, budget_limit):
        limits.append(budget_limit)
        return run_summary(output, score=.7 if config.stem == "planner-a" else .8)

    monkeypatch.setattr(campaign_module, "run", run)
    output = tmp_path / "development"
    result = campaign_module.campaign(campaign_task, [Path("planner-a"), Path("planner-b")],
                                      output, previous=[old_summary], development_reserves=True)

    assert limits == pytest.approx([.65, .55])
    assert collector_calls == [[old_summary]]
    assert result["previous_cost_rmb"] == pytest.approx(.2)
    assert result["campaign_cost_rmb"] == pytest.approx(.2)
    assert result["cumulative_cost_rmb"] == pytest.approx(.4)
    assert result["unresolved_reserved_rmb"] == pytest.approx(.15)
    assert result["budget_committed_rmb"] == pytest.approx(.55)
    assert result["unresolved_calls"] == [original_unknown]
    assert result["status"] == "development_completed_unresolved_accounting"
    assert result["execution_complete"] is True
    assert result["accounting_complete"] is False
    assert result["course_pricing_compliant"] is False
    assert result["test_evaluated"] is False
    assert result["selection"]["planner_model"] == "planner-b"
    assert result["selection"]["validation_score"] == .8
    assert result["selection"]["model_path"] == str(output / "planner_02/model.joblib")
    assert old_summary.read_bytes() == original_bytes
    assert json.loads((output / "campaign.json").read_text(encoding="utf-8")) == result


def test_new_timeout_reserves_budget_and_moves_to_next_planner_without_retrying(
        campaign_task, tmp_path, monkeypatch):
    output = tmp_path / "development"
    timed_out = output / "planner_01"
    pending = unresolved_call(timed_out, .12)
    collector_paths = []

    def collect(paths, data_hashes, pricing):
        collector_paths.append(paths)
        if not paths:
            return history()
        assert paths == [timed_out / "summary.json"]
        return history(.03, .12, [pending], paths, [timed_out])

    monkeypatch.setattr(history_module, "collect_development_history", collect)
    calls = []

    def run(task, directory, mode, config, *timeouts, budget_limit):
        calls.append((config.stem, directory, budget_limit))
        if config.stem == "planner-a":
            return run_summary(directory, score=.99, cost=.03, complete=False)
        return run_summary(directory, score=.72, cost=.04)

    monkeypatch.setattr(campaign_module, "run", run)
    result = campaign_module.campaign(campaign_task, [Path("planner-a"), Path("planner-b")],
                                      output, development_reserves=True)

    assert [call[:2] for call in calls] == [
        ("planner-a", output / "planner_01"), ("planner-b", output / "planner_02")]
    assert [call[2] for call in calls] == pytest.approx([1, .85])
    assert collector_paths == [[], [timed_out / "summary.json"]]
    assert result["cumulative_cost_rmb"] == pytest.approx(.07)
    assert result["unresolved_reserved_rmb"] == pytest.approx(.12)
    assert result["budget_committed_rmb"] == pytest.approx(.19)
    assert result["unresolved_calls"] == [pending]
    assert result["accounting_complete"] is False
    assert result["status"] == "development_partial_unresolved_accounting"
    assert result["selection"]["planner_model"] == "planner-b"
    assert result["selection"]["validation_score"] == .72
    assert json.loads((timed_out / "summary.json").read_text(encoding="utf-8"))["usage"]["accounting_complete"] is False


@pytest.mark.parametrize("error", ["response model differs from configured model",
                                   "reported token usage exceeds configured token limit"])
def test_invalid_current_accounting_stops_instead_of_bypassing_to_next_planner(
        campaign_task, tmp_path, monkeypatch, error):
    collector_calls = []

    def collect(paths, data_hashes, pricing):
        collector_calls.append(paths)
        if not paths:
            return history()
        raise ValueError(error)

    monkeypatch.setattr(history_module, "collect_development_history", collect)
    called = []

    def run(task, output, mode, config, *timeouts, budget_limit):
        called.append(config.stem)
        return run_summary(output, score=.99, complete=False, blocked=True)

    monkeypatch.setattr(campaign_module, "run", run)
    result = campaign_module.campaign(campaign_task, [Path("planner-a"), Path("planner-b")],
                                      tmp_path / "development", development_reserves=True)
    assert called == ["planner-a"]
    assert len(collector_calls) == 2
    assert result["status"] == "partial_accounting"
    assert result["accounting_complete"] is False
    assert result["selection"] is None


def test_known_token_violation_also_stops_next_planner(campaign_task, tmp_path, monkeypatch):
    collector_calls = []

    def collect(paths, data_hashes, pricing):
        collector_calls.append(paths)
        return history()

    monkeypatch.setattr(history_module, "collect_development_history", collect)
    called = []

    def run(task, output, mode, config, *timeouts, budget_limit):
        called.append(config.stem)
        return run_summary(output, status="partial_accounting_limits_violated", blocked=True)

    monkeypatch.setattr(campaign_module, "run", run)
    result = campaign_module.campaign(campaign_task, [Path("planner-a"), Path("planner-b")],
                                      tmp_path / "development", development_reserves=True)
    assert called == ["planner-a"]
    assert collector_calls == [[]]
    assert result["status"] == "partial_accounting"
    assert result["selection"] is None


def test_default_mode_still_stops_at_first_unknown(campaign_task, tmp_path, monkeypatch):
    def forbidden_collector(*args):
        pytest.fail("Default mode must not allocate development reservations")

    monkeypatch.setattr(history_module, "collect_development_history", forbidden_collector)
    calls = []

    def run(task, output, mode, config, *timeouts, budget_limit):
        calls.append(config.stem)
        return run_summary(output, score=.99, complete=False)

    monkeypatch.setattr(campaign_module, "run", run)
    result = campaign_module.campaign(campaign_task, [Path("planner-a"), Path("planner-b")],
                                      tmp_path / "strict")
    assert calls == ["planner-a"]
    assert result["development_reserves"] is False
    assert result["status"] == "partial_accounting"
    assert result["accounting_complete"] is False
    assert result["selection"] is None


@pytest.mark.parametrize("development, expected_status, selected", [
    (False, "partial", False), (True, "completed", True)])
def test_warning_completion_is_selectable_only_with_explicit_development_mode(
        campaign_task, tmp_path, monkeypatch, development, expected_status, selected):
    monkeypatch.setattr(history_module, "collect_development_history", lambda *args: history())
    monkeypatch.setattr(campaign_module, "run", lambda task, output, *args, **kwargs:
                        run_summary(output, status="completed_with_warnings"))
    result = campaign_module.campaign(campaign_task, [Path("planner-a")], tmp_path / "campaign",
                                      development_reserves=development)
    assert result["status"] == expected_status
    assert (result["selection"] is not None) is selected
    assert result["accounting_complete"] is True


@pytest.mark.parametrize("known, reserved", [(.7, .3), (.7, .4)])
def test_known_cost_plus_reservation_exhaustion_prevents_any_new_run(
        campaign_task, tmp_path, monkeypatch, known, reserved):
    pending = unresolved_call(tmp_path / "old", reserved)
    monkeypatch.setattr(history_module, "collect_development_history", lambda *args:
                        history(known, reserved, [pending]))
    monkeypatch.setattr(campaign_module, "run", lambda *args, **kwargs:
                        pytest.fail("Exhausted budget must prevent new requests"))
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="budget is exhausted"):
        campaign_module.campaign(campaign_task, [Path("planner-a")], output,
                                 development_reserves=True)
    assert not output.exists()


def test_development_cannot_mix_reservations_with_reconciliation(
        campaign_task, tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "collect_development_history", lambda *args:
                        pytest.fail("Mixed accounting modes must fail before collecting history"))
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="strict accounting mode"):
        campaign_module.campaign(campaign_task, [Path("planner-a")], output,
                                 development_reserves=True, reconciliations=[tmp_path / "evidence.json"])
    assert not output.exists()


def test_real_ledgers_remain_unresolved_across_successive_development_campaigns(tmp_path, monkeypatch):
    from test_development_accounting import make_run
    from superagent.data import load_task
    from superagent.utils import file_hash, write_json

    make_demo(tmp_path / "data")
    task_path = tmp_path / "data/tabular_task.json"
    task_document = json.loads(task_path.read_text(encoding="utf-8"))
    task_document.pop("predict_path")
    write_json(task_path, task_document)
    task = load_task(task_path)
    hashes = {"train": file_hash(task.train_path)}

    old = make_run(tmp_path / "old")
    write_json(old.parent / "manifest.json", {"data_sha256": hashes})
    original_files = {path: path.read_bytes() for path in old.parent.iterdir() if path.is_file()}
    config_path = old.parent / "llm_config.snapshot.json"
    calls = []

    def run(task, output, mode, config, *timeouts, budget_limit):
        calls.append((output, budget_limit))
        summary_path = make_run(output, failure=None)
        write_json(output / "manifest.json", {"data_sha256": hashes})
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(llm_requests=1, valid_llm_decisions=1, best_validation_score=.7,
                       selected_candidate={"algorithm": "random_forest", "params": {}},
                       model=str(output / "model.joblib"), source_sha256="same-source")
        write_json(summary_path, summary)
        return summary

    monkeypatch.setattr(campaign_module, "run", run)
    first = campaign_module.campaign(task_path, [config_path], tmp_path / "first",
                                    budget_rmb=.04, previous=[old], development_reserves=True)
    second = campaign_module.campaign(task_path, [config_path], tmp_path / "second",
                                     budget_rmb=.04, previous=[tmp_path / "first/campaign.json"],
                                     development_reserves=True)

    assert [amount for _, amount in calls] == pytest.approx([.02966, .02952])
    assert first["cumulative_cost_rmb"] == pytest.approx(.00028)
    assert second["previous_cost_rmb"] == pytest.approx(.00028)
    assert second["cumulative_cost_rmb"] == pytest.approx(.00042)
    assert second["unresolved_reserved_rmb"] == pytest.approx(.0102)
    assert second["budget_committed_rmb"] == pytest.approx(.01062)
    assert second["unresolved_calls"] == first["unresolved_calls"]
    assert second["accounting_complete"] is False
    assert second["status"] == "development_completed_unresolved_accounting"
    assert len(second["charged_run_directories"]) == 3
    assert second["selection"]["run_directory"] == str(tmp_path / "second/planner_01")
    assert {path: path.read_bytes() for path in original_files} == original_files


@pytest.mark.parametrize("development", [False, True])
def test_baseline_only_run_with_invalid_llm_responses_cannot_be_selected(
        campaign_task, tmp_path, monkeypatch, development):
    monkeypatch.setattr(history_module, "collect_development_history", lambda *args: history())

    def run(task, output, *args, **kwargs):
        summary = run_summary(output, score=.99)
        summary["valid_llm_decisions"] = 0
        return summary

    monkeypatch.setattr(campaign_module, "run", run)
    result = campaign_module.campaign(campaign_task, [Path("planner-a")], tmp_path / "campaign",
                                      development_reserves=development)
    assert result["selection"] is None
    assert result["execution_complete"] is False
    assert result["status"] == "partial"


def test_blocked_accounting_cannot_select_a_run_even_if_summary_says_completed(
        campaign_task, tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "collect_development_history", lambda *args: history())
    monkeypatch.setattr(campaign_module, "run", lambda task, output, *args, **kwargs:
                        run_summary(output, blocked=True))
    result = campaign_module.campaign(campaign_task, [Path("planner-a"), Path("planner-b")],
                                      tmp_path / "campaign", development_reserves=True)
    assert len(result["runs"]) == 1
    assert result["selection"] is None
    assert result["execution_complete"] is False
    assert result["status"] == "partial_accounting"
