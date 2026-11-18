"""Coding protocol/repair tests use a fake HTTP transport, never the school API."""
import importlib
import json
from copy import deepcopy
from dataclasses import asdict

import httpx
import pytest

from superagent.coding_planner import CodingPlanner, MAX_SOURCE_BYTES, _messages, source_digest, validate_code_plan
from superagent.usage import AgentPlannerError, InvalidPlanError, LLMConfig


SOURCE = "from sklearn.svm import SVC\n\ndef build_model(X_train, y_train, seed):\n    return SVC(C=1, probability=True, random_state=seed)\n"
CHANGED = SOURCE.replace("C=1", "C=2")
PROFILE = {"data_type": "tabular", "metric": "accuracy", "training_rows": 80,
           "feature_count": 3, "class_count": 2,
           "feature_schema": [{"name": "column_0", "kind": "numeric", "dtype": "float64", "unique_count": 80}]}


def plan(source=SOURCE, name="kernel_svm"):
    return {"action": "experiment", "candidate": {"algorithm": "generated_python", "name": name, "source": source},
            "reason": "Try a nonlinear decision boundary after the measured baseline."}


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    config = LLMConfig(base_url="https://provider.example/v1", model="test-code-model",
                       api_key_env="CODING_PLANNER_TEST_KEY", model_family="deepseek",
                       model_release_date="2026-01-01", pricing_as_of="2026-10-21",
                       input_rmb_per_million=2., output_rmb_per_million=8.,
                       price_source="https://provider.example/prices", model_source="https://provider.example/model",
                       verified=True)
    monkeypatch.setenv(config.api_key_env, "local-fake-key-not-live")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    return path


def install_responses(monkeypatch, responses):
    contexts, messages = [], []
    values = iter(responses)

    def fake_post(config, payload, api_key):
        messages.append(payload["messages"])
        contexts.append(json.loads(payload["messages"][1]["content"]))
        response = next(values)
        content = json.dumps(response) if isinstance(response, dict) else response
        body = {"model": config.model, "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", config.base_url))

    monkeypatch.setattr(importlib.import_module("superagent.planner"), "post_completion", fake_post)
    return contexts, messages


def test_generates_real_module_outside_old_catalog_and_meters_call(tmp_path, monkeypatch, config_path):
    contexts, messages = install_responses(monkeypatch, [plan()])
    planner = CodingPlanner(config_path, tmp_path / "run")
    result = planner.propose(PROFILE, [], {"logistic_regression": {"C": "float"}})
    assert result == plan()
    assert "SVC" in result["candidate"]["source"]
    assert contexts[0]["baseline_library_reference"] == ["logistic_regression"]
    assert "catalog configuration" in messages[0][0]["content"]
    assert planner.summary()["request_count"] == 1
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.0006)
    assert planner.summary()["accounting_complete"] is True


def test_bad_json_remains_charged_then_corrective_context_clears(tmp_path, monkeypatch, config_path):
    contexts, _ = install_responses(monkeypatch, [json.dumps(plan()) + "}", plan(), {"action": "stop", "reason": "No further evidence warrants a trial."}])
    planner = CodingPlanner(config_path, tmp_path / "run")
    with pytest.raises(InvalidPlanError):
        planner.propose(PROFILE, [], {})
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.0006)
    planner.propose(PROFILE, [], {})
    planner.propose(PROFILE, [], {})
    assert "previous_plan_error" not in contexts[0]
    assert "Invalid JSON" in contexts[1]["previous_plan_error"]
    assert "previous_plan_error" not in contexts[2]
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.0018)
    calls = [json.loads(line) for line in (tmp_path / "run/llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [row for row in calls if row["event"] == "request_completed"]
    assert len(completed) == 3
    assert completed[0]["response"]["choices"][0]["message"]["content"].endswith("}}")


def test_repair_prompt_contains_only_latest_complete_source_and_safe_feedback(tmp_path, monkeypatch, config_path):
    contexts, _ = install_responses(monkeypatch, [plan(CHANGED)])
    planner = CodingPlanner(config_path, tmp_path / "run")
    earlier = "# EARLIER_SOURCE_ONLY\n" + SOURCE
    broken = "# LATEST_BROKEN_SOURCE\n" + SOURCE.replace("return SVC", "return MissingEstimator")
    history = [
        {"candidate": {"algorithm": "logistic_regression", "params": {}}, "status": "success", "score": .55},
        {"candidate": plan(earlier)["candidate"], "status": "success", "score": .7, "training_score": .9, "generalization_gap": .2, "duration_seconds": 1.3},
        {"candidate": plan(broken)["candidate"], "status": "failed", "score": None, "error_type": "NameError",
         "error_feedback": "NameError: MissingEstimator is not defined at generated source line 5", "error": "RAW_PRIVATE_EXCEPTION C:/private/credentials.json"},
    ]
    planner.propose({**PROFILE, "name": "PRIVATE_TASK_NAME", "train_path": "C:/private/train.csv"}, history, {})
    context = contexts[0]
    assert context["latest_generated_candidate"]["source"] == broken
    assert context["latest_generated_candidate"]["source_sha256"] == source_digest(broken)
    assert "MissingEstimator" in context["latest_generated_candidate"]["error_feedback"]
    assert len(context["validation_history"]) == 3
    assert context["validation_history"][1]["generalization_gap"] == .2
    serialized = json.dumps(context)
    for secret in ("EARLIER_SOURCE_ONLY", "PRIVATE_TASK_NAME", "credentials.json", "train.csv", "RAW_PRIVATE_EXCEPTION"):
        assert secret not in serialized
    assert serialized.count("LATEST_BROKEN_SOURCE") == 1


def test_name_change_cannot_evade_exact_source_duplicate_guard(tmp_path, monkeypatch, config_path):
    contexts, _ = install_responses(monkeypatch, [plan(name="renamed"), plan(CHANGED)])
    planner = CodingPlanner(config_path, tmp_path / "run")
    history = [{"candidate": plan()["candidate"], "status": "success", "score": .7}]
    with pytest.raises(InvalidPlanError, match="repeated generated source"):
        planner.propose(PROFILE, history, {})
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.0006)
    assert planner.propose(PROFILE, history, {})["candidate"]["source"] == CHANGED
    assert "Change the actual implementation" in contexts[1]["previous_plan_error"]


def test_code_schema_error_gets_code_specific_safe_feedback(tmp_path, monkeypatch, config_path):
    bad = {"action": "experiment", "candidate": {"algorithm": "random_forest", "params": {}}, "reason": "wrong mode"}
    contexts, _ = install_responses(monkeypatch, [bad, plan()])
    planner = CodingPlanner(config_path, tmp_path / "run")
    with pytest.raises(InvalidPlanError):
        planner.propose(PROFILE, [], {})
    planner.propose(PROFILE, [], {})
    assert "generated_python" in contexts[1]["previous_plan_error"]
    assert planner.summary()["request_count"] == 2


@pytest.mark.parametrize("change", [
    {"name": ""}, {"name": "x" * 81}, {"name": "line\nname"},
    {"source": ""}, {"source": None}, {"source": "x" * (MAX_SOURCE_BYTES + 1)},
    {"source": "中" * 3400}, {"source": "\ud800"}, {"algorithm": "random_forest"}, {"params": {}},
])
def test_invalid_candidates_are_rejected(change):
    proposed = plan()
    proposed["candidate"].update(change)
    with pytest.raises(InvalidPlanError):
        validate_code_plan(proposed)


def test_syntax_error_is_left_for_backend_to_report_as_a_repairable_trial():
    # Source policy/syntax checking is deliberately not a hidden planner retry.
    source = "def build_model(\n"
    assert validate_code_plan(plan(source))["candidate"]["source"] == source
    assert validate_code_plan({"action": "stop", "reason": "No additional useful trial."})["action"] == "stop"


def test_context_compaction_preserves_all_results_and_latest_code():
    history = [{"candidate": plan(SOURCE + f"# trial {i}\n", name="candidate" + str(i))["candidate"],
                "status": "success", "score": .7, "training_score": .8,
                "generalization_gap": .1, "duration_seconds": 1.2} for i in range(80)]
    messages = _messages(PROFILE, history, {"random_forest": {}}, None, 16000)
    context = json.loads(messages[1]["content"])
    assert len(context["validation_history"]) == len(history)
    assert context["latest_generated_candidate"]["source"] == history[-1]["candidate"]["source"]
    assert "history_columns" in context
    assert sum(len(m["content"].encode("utf-8")) + 256 for m in messages) + 256 <= 16000


def test_unfittable_context_does_not_send_request_or_truncate_source(tmp_path, monkeypatch, config_path):
    planner = CodingPlanner(config_path, tmp_path / "run")
    called = []
    monkeypatch.setattr(planner, "_request_json", lambda messages: called.append(messages))
    long_source = "# " + "x" * 9900
    history = [{"candidate": plan(long_source, "long")["candidate"], "status": "failed",
                "error_type": "SyntaxError", "error_feedback": "failure " * 120} for _ in range(100)]
    with pytest.raises(AgentPlannerError, match="complete latest module"):
        planner.propose(PROFILE, history, {})
    assert not called
    assert planner.summary()["request_count"] == 0


def test_feedback_redacts_paths_urls_and_excludes_arbitrary_profile_keys(tmp_path, monkeypatch, config_path):
    contexts, _ = install_responses(monkeypatch, [plan()])
    planner = CodingPlanner(config_path, tmp_path / "run")
    profile = {**PROFILE, "labels": ["private-class"], "samples": ["private-row"]}
    history = [{"candidate": {"algorithm": "baseline"}, "status": "failed", "error_type": "ValueError",
                "error_feedback": "ValueError at 'C:\\secret\\data.csv', '/private/data/file.csv', https://example.com/private"}]
    planner.propose(profile, history, {})
    rendered = json.dumps(contexts[0])
    assert "local_path" in rendered and "[url]" in rendered
    for secret in ("secret", "data.csv", "file.csv", "private-row", "private-class", "https://"):
        assert secret not in rendered


def test_image_profile_and_multiclass_contract_are_available_without_paths(tmp_path, monkeypatch, config_path):
    contexts, messages = install_responses(monkeypatch, [plan()])
    planner = CodingPlanner(config_path, tmp_path / "run")
    profile = {"data_type": "image", "metric": "accuracy", "class_count": 3,
               "image_shape": [32, 32, 3], "n_channels": 3, "image_dtype": "uint8",
               "image_pixel_range": [0, 255], "image_directory": "C:/private/images",
               "feature_schema": [{"name": "column_0", "kind": "image", "dtype": "object"}]}
    planner.propose(profile, [], {"image_logistic": {}})
    passed = contexts[0]["training_profile"]
    assert passed["image_shape"] == [32, 32, 3] and passed["class_count"] == 3
    assert passed["image_dtype"] == "uint8" and passed["image_pixel_range"] == [0, 255]
    assert "image_directory" not in passed
    assert "multiclass" in messages[0][0]["content"]


@pytest.mark.parametrize("nested_reason,top_reason,action_note", [
    ("Explain the implementation here.", "Top-level explanation takes precedence.", False),
    ("Explanation supplied only within the candidate.", None, False),
    ("Explanation supplied only within the candidate.", None, True),
    (None, "Ordinary top-level explanation.", True),
    ("", "Empty unused metadata is still a bounded string.", False),
    ("x" * 2000, None, False),
])
def test_observed_metadata_normalizes_without_mutating_input(nested_reason, top_reason, action_note):
    proposed = plan()
    proposed.pop("reason")
    if nested_reason is not None:
        proposed["candidate"]["reason"] = nested_reason
    if top_reason is not None:
        proposed["reason"] = top_reason
    if action_note:
        proposed["action_note"] = "experiment"
    original = deepcopy(proposed)
    normalized = validate_code_plan(proposed)
    assert normalized == {"action": "experiment", "candidate": plan()["candidate"],
                          "reason": top_reason if top_reason is not None else nested_reason}
    assert normalized is not proposed and normalized["candidate"] is not proposed["candidate"]
    assert proposed == original


def test_matching_stop_action_note_is_normalized_without_mutation():
    proposed = {"action": "stop", "action_note": "stop", "reason": "Enough validation evidence."}
    original = deepcopy(proposed)
    assert validate_code_plan(proposed) == {"action": "stop", "reason": proposed["reason"]}
    assert proposed == original


@pytest.mark.parametrize("metadata", [None, False, 17, [], {}, "x" * 2001])
def test_candidate_reason_metadata_remains_bounded_string(metadata):
    proposed = plan()
    proposed["candidate"]["reason"] = metadata
    original = deepcopy(proposed)
    with pytest.raises(InvalidPlanError):
        validate_code_plan(proposed)
    assert proposed == original


@pytest.mark.parametrize("action_note", [None, False, 1, [], {}, "stop", "experiment ", "x" * 2001])
def test_action_note_must_match_action_exactly(action_note):
    proposed = {**plan(), "action_note": action_note}
    original = deepcopy(proposed)
    with pytest.raises(InvalidPlanError):
        validate_code_plan(proposed)
    assert proposed == original


@pytest.mark.parametrize("location,field,value", [
    ("top", "explanation", "Other unknown metadata"),
    ("top", "notes", "Other unknown metadata"),
    ("candidate", "action_note", "experiment"),
    ("candidate", "params", {}),
    ("candidate", "explanation", "Other unknown metadata"),
    ("candidate", "algorithm", "random_forest"),
    ("candidate", "source", None),
])
def test_metadata_normalization_does_not_weaken_other_fields(location, field, value):
    proposed = plan()
    proposed["candidate"]["reason"] = proposed.pop("reason")
    proposed["action_note"] = "experiment"
    (proposed if location == "top" else proposed["candidate"])[field] = value
    with pytest.raises(InvalidPlanError):
        validate_code_plan(proposed)


@pytest.mark.parametrize("top_reason", [None, "", 17])
def test_existing_invalid_top_reason_is_not_replaced_by_nested_reason(top_reason):
    proposed = plan()
    proposed["reason"] = top_reason
    proposed["candidate"]["reason"] = "Valid candidate metadata does not override an explicit top-level value."
    with pytest.raises(InvalidPlanError):
        validate_code_plan(proposed)


def test_observed_provider_metadata_is_accepted_and_raw_paid_response_retained(tmp_path, monkeypatch, config_path):
    # These shapes reproduce the extra metadata seen in genuine provider replies.
    first = plan()
    first["candidate"]["reason"] = "The provider repeats its rationale inside the candidate."
    second = plan(CHANGED)
    second["candidate"]["reason"] = second.pop("reason")
    second["action_note"] = "experiment"
    originals = deepcopy([first, second])
    contexts, _ = install_responses(monkeypatch, [first, second])
    planner = CodingPlanner(config_path, tmp_path / "run")
    result1 = planner.propose(PROFILE, [], {})
    result2 = planner.propose(PROFILE, [{"candidate": result1["candidate"], "status": "success", "score": .7}], {})
    assert result1 == plan() and result2 == plan(CHANGED)
    assert "previous_plan_error" not in contexts[1]
    assert planner.summary()["request_count"] == 2
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.0012)
    assert planner.summary()["accounting_complete"] is True
    calls = [json.loads(line) for line in (tmp_path / "run/llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [row for row in calls if row["event"] == "request_completed"]
    assert [json.loads(row["response"]["choices"][0]["message"]["content"]) for row in completed] == originals
    assert [first, second] == originals


def test_reason_metadata_cannot_bypass_exact_source_duplicate_guard(tmp_path, monkeypatch, config_path):
    proposed = plan(name="metadata_renamed")
    proposed["candidate"]["reason"] = "Different metadata is not a different implementation."
    install_responses(monkeypatch, [proposed])
    planner = CodingPlanner(config_path, tmp_path / "run")
    with pytest.raises(InvalidPlanError, match="repeated generated source"):
        planner.propose(PROFILE, [{"candidate": plan()["candidate"], "status": "success", "score": .7}], {})
    assert planner.summary()["accounted_cost_rmb"] == pytest.approx(.0006)
