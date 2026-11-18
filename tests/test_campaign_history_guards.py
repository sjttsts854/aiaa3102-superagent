"""Regression tests for consuming only settled, completed campaign history.

No provider calls are made. These guards prevent unfinished or blocked history
from making its already committed budget available to a second campaign.
"""

import importlib
import json
from pathlib import Path

import pytest

from superagent.data import load_task
from superagent.demo import make_demo
from superagent.usage import LLMConfig
from superagent.utils import file_hash


module = importlib.import_module("superagent.campaign")


@pytest.fixture
def context(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    task_path = tmp_path / "data/tabular_task.json"
    raw = json.loads(task_path.read_text())
    raw.pop("predict_path")
    task_path.write_text(json.dumps(raw))
    task = load_task(task_path)
    config = LLMConfig.load(Path(__file__).parents[1] / "config/hkust-deepseek-flash.json")
    monkeypatch.setattr(module.LLMConfig, "load", lambda path: config)
    calls = []

    def fake_run(task, output, *args, budget_limit):
        calls.append(budget_limit)
        return {
            "status": "completed", "llm_requests": 1, "valid_llm_decisions": 1,
            "usage": {"accounted_cost_rmb": .1, "accounting_complete": True, "blocked": False},
            "best_validation_score": .8, "selected_candidate": {"algorithm": "logistic_regression", "params": {}},
            "model": str(output / "model.joblib"), "source_sha256": "test-source",
        }

    monkeypatch.setattr(module, "run", fake_run)
    previous = tmp_path / "previous"
    previous.mkdir()
    hashes = {"train_path": file_hash(task.train_path)}
    past = {
        "status": "completed", "data_sha256": hashes,
        "pricing_basis": config.pricing_basis, "pricing_as_of": config.pricing_as_of,
        "accounting_complete": True, "cumulative_cost_rmb": .2,
        "runs": [], "previous_campaigns": [],
    }
    return task_path, config, calls, previous, past


def _continue(tmp_path, context, past):
    task, _, _, previous, _ = context
    path = previous / "campaign.json"
    path.write_text(json.dumps(past))
    return module.campaign(task, [Path("mock-config")], tmp_path / "next", previous=[path])


@pytest.mark.parametrize("status", ["running", "partial_accounting", "partial_source_changed", "partial", "failed", None])
def test_noncompleted_campaign_cannot_reuse_budget_even_if_accounting_flag_true(tmp_path, context, status):
    past = context[4]
    past["status"] = status
    with pytest.raises(ValueError, match="completed|unfinished|unresolved"):
        _continue(tmp_path, context, past)
    assert context[2] == []
    assert not (tmp_path / "next").exists()


@pytest.mark.parametrize("location", ["top_level", "usage", "nested_run"])
def test_blocked_campaign_is_rejected_despite_completed_status_and_known_usage(tmp_path, context, location):
    past = context[4]
    if location == "top_level":
        past["blocked"] = True
    elif location == "usage":
        past["usage"] = {"blocked": True, "accounting_complete": True}
    else:
        past["runs"] = [{"summary": {"status": "partial_accounting_limits_violated",
                                    "usage": {"blocked": True, "accounting_complete": True}}}]
    with pytest.raises(ValueError, match="blocked"):
        _continue(tmp_path, context, past)
    assert context[2] == []
    assert not (tmp_path / "next").exists()


def _pilot(context, status="completed", blocked=False, embedded_usage=True):
    _, config, _, previous, past = context
    usage = {"accounted_cost_rmb": .2, "accounting_complete": True, "blocked": blocked}
    summary = {"status": status}
    if embedded_usage:
        summary["usage"] = usage
    (previous / "summary.json").write_text(json.dumps(summary))
    (previous / "usage_summary.json").write_text(json.dumps(usage))
    (previous / "manifest.json").write_text(json.dumps({"data_sha256": {"train": past["data_sha256"]["train_path"]}}))
    (previous / "llm_config.snapshot.json").write_text(json.dumps({"pricing_basis": config.pricing_basis, "pricing_as_of": config.pricing_as_of}))
    return previous / "summary.json"


@pytest.mark.parametrize("status", ["running", "partial_accounting", "failed", None])
def test_noncompleted_standalone_pilot_cannot_be_converted_to_settled_history(tmp_path, context, status):
    path = _pilot(context, status=status)
    with pytest.raises(ValueError, match="completed|unfinished|unresolved"):
        module.campaign(context[0], [Path("mock-config")], tmp_path / "next", previous=[path])
    assert context[2] == []
    assert not (tmp_path / "next").exists()


@pytest.mark.parametrize("embedded_usage", [True, False])
def test_blocked_standalone_pilot_is_rejected_with_known_accounting(tmp_path, context, embedded_usage):
    path = _pilot(context, blocked=True, embedded_usage=embedded_usage)
    with pytest.raises(ValueError, match="blocked|unresolved"):
        module.campaign(context[0], [Path("mock-config")], tmp_path / "next", previous=[path])
    assert context[2] == []


@pytest.mark.parametrize("kind", ["campaign", "pilot"])
def test_completed_unblocked_history_remains_usable_and_is_charged(tmp_path, context, kind):
    if kind == "campaign":
        result = _continue(tmp_path, context, context[4])
    else:
        path = _pilot(context)
        result = module.campaign(context[0], [Path("mock-config")], tmp_path / "next", previous=[path])
    assert context[2] == pytest.approx([.8])
    assert result["previous_cost_rmb"] == pytest.approx(.2)
    assert result["cumulative_cost_rmb"] == pytest.approx(.3)
    assert result["status"] == "completed"
