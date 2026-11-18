import importlib
import json
from pathlib import Path

import pytest

from superagent.demo import make_demo
from superagent.usage import LLMConfig

module = importlib.import_module("superagent.campaign")


def setup_campaign(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    task = tmp_path / "data" / "tabular_task.json"
    raw = json.loads(task.read_text())
    raw.pop("predict_path")
    task.write_text(json.dumps(raw))
    config = LLMConfig.load(Path(__file__).parents[1] / "config/hkust-deepseek-flash.json")
    monkeypatch.setattr(module.LLMConfig, "load", lambda p: config)
    return task


def fake_summary(output, score, cost=.2, complete=True):
    return {"status": "completed" if complete else "partial_unresolved_accounting",
        "usage": {"accounted_cost_rmb": cost, "accounting_complete": complete},
        "llm_requests": 2, "valid_llm_decisions": 1, "best_validation_score": score,
        "selected_candidate": {"algorithm": "random_forest", "params": {}},
        "model": str(output / "model.joblib"), "source_sha256": "same-source"}


def test_campaign_shares_budget_selects_validation_winner_and_carries_cost(tmp_path, monkeypatch):
    task = setup_campaign(tmp_path, monkeypatch)
    limits = []
    scores = iter([.7, .8, .75])
    def run(task, output, *args, budget_limit):
        limits.append(budget_limit)
        return fake_summary(output, next(scores))
    monkeypatch.setattr(module, "run", run)
    result = module.campaign(task, [Path("a"), Path("b")], tmp_path / "first")
    assert limits == pytest.approx([1, .8])
    assert result["selection"]["validation_score"] == .8
    assert result["campaign_cost_rmb"] == pytest.approx(.4)
    second = module.campaign(task, [Path("a")], tmp_path / "second", previous=[tmp_path / "first/campaign.json"])
    assert limits[-1] == pytest.approx(.6)
    assert second["cumulative_cost_rmb"] == pytest.approx(.6)
    assert second["test_evaluated"] is False


def test_unresolved_usage_stops_other_planners_and_cannot_be_reset(tmp_path, monkeypatch):
    task = setup_campaign(tmp_path, monkeypatch)
    calls = []
    def run(task, output, *args, **kwargs):
        calls.append(output)
        return fake_summary(output, .8, complete=False)
    monkeypatch.setattr(module, "run", run)
    result = module.campaign(task, [Path("a"), Path("b")], tmp_path / "first")
    assert len(calls) == 1
    assert result["status"] == "partial_accounting"
    assert result["selection"] is None
    with pytest.raises(ValueError, match="unresolved"):
        module.campaign(task, [Path("a")], tmp_path / "second", previous=[tmp_path / "first/campaign.json"])


def test_campaign_rejects_prediction_path_before_running(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    with pytest.raises(ValueError, match="omit predict_path"):
        module.campaign(tmp_path / "data/tabular_task.json", [Path("a")], tmp_path / "run")
    assert not (tmp_path / "run").exists()
