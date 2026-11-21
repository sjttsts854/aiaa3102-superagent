"""The experimental tier state machine needs no API, credentials, or network."""
import pytest

from superagent.request_policy import PolicyExhausted, ThreeTierPolicy


def candidate(source="source_a", status="success", score=0.5, algorithm="generated_python"):
    return {"candidate": {"algorithm": algorithm, "source": source}, "status": status, "score": score}


BASELINE = candidate(algorithm="logistic_regression", score=0.7)
TWO_FAILURES = [BASELINE, candidate("a", "failed"), candidate("b", "failed")]
THREE_SUCCESS = [BASELINE, candidate("a", score=0.6), candidate("b", score=0.7), candidate("c", score=0.65)]


def settings(tier):
    return tier.name, tier.thinking, tier.max_output_tokens, tier.deadline_seconds


def test_default_and_choose_does_not_spend_allowances():
    policy = ThreeTierPolicy()
    tier = policy.choose([])
    assert settings(tier) == ("default", False, 8192, 240)
    assert tier.reason
    assert policy.choose([]) == tier
    assert not policy.long_code_used and not policy.difficult_used


def test_length_upgrade_only_spends_allowance_when_sent():
    policy = ThreeTierPolicy()
    normal = policy.choose([])
    policy.mark_sent(normal)
    recovery = policy.record_length(normal)
    assert settings(recovery) == ("long_code", False, 16384, 360)
    assert policy.choose(TWO_FAILURES) == recovery  # length precedes thinking
    assert not policy.long_code_used
    policy.mark_sent(recovery)
    assert policy.long_code_used
    assert policy.choose([]).name == "default"
    with pytest.raises(PolicyExhausted):
        policy.record_length(recovery)
    with pytest.raises(PolicyExhausted):
        policy.choose(TWO_FAILURES)


def test_later_default_length_cannot_reuse_successful_long_recovery():
    policy = ThreeTierPolicy()
    normal = policy.choose([])
    policy.mark_sent(normal)
    recovery = policy.record_length(normal)
    policy.mark_sent(recovery)
    normal = policy.choose([])
    policy.mark_sent(normal)
    with pytest.raises(PolicyExhausted):
        policy.record_length(normal)


@pytest.mark.parametrize("history", [TWO_FAILURES, THREE_SUCCESS])
def test_difficult_thresholds_and_single_use_for_same_history(history):
    policy = ThreeTierPolicy()
    tier = policy.choose(history)
    assert settings(tier) == ("difficult", True, 16384, 600)
    assert tier.reason
    assert policy.choose(history) == tier
    assert not policy.difficult_used  # budget rejection can leave it unused
    policy.mark_sent(tier)
    assert policy.difficult_used
    normal = policy.choose(history)
    assert normal.name == "default" and "return" in normal.reason
    policy.mark_sent(normal)
    assert policy.choose(history).name == "default"
    with pytest.raises(PolicyExhausted):
        policy.mark_sent(tier)


@pytest.mark.parametrize("history", [
    [BASELINE, candidate("a", "failed")],  # first repair stays non-thinking
    [candidate("baseline", "failed", algorithm="logistic_regression"), candidate("a", "failed")],
    [BASELINE, candidate("a", "failed"), candidate("a", "failed")],
    [BASELINE, candidate("a", "failed"), candidate("b"), candidate("c", "failed")],
    [BASELINE, candidate("a", "failed"), candidate("b", "pending")],
    [BASELINE, candidate("", "failed"), candidate("b", "failed")],
    [BASELINE, candidate("a"), candidate("b")],
    [BASELINE, candidate("a"), candidate("b"), candidate("a")],
    [BASELINE, candidate("a"), candidate("b"), candidate("c", score=0.71)],
    [BASELINE, candidate("a"), candidate("b"), candidate("c", score=float("nan"))],
    [BASELINE, candidate("a"), candidate("b"), candidate("c", score=True)],
    [candidate("a"), candidate("b"), candidate("c")],  # baseline unavailable
    [BASELINE, {"status": "failed", "error_type": "ReadTimeout"}],
])
def test_history_without_qualifying_execution_evidence_stays_default(history):
    assert ThreeTierPolicy().choose(history).name == "default"


def test_success_resets_consecutive_failure_streak():
    history = TWO_FAILURES + [candidate("c", score=0.65), candidate("d", "failed")]
    assert ThreeTierPolicy().choose(history).name == "default"


def test_thinking_length_stops_without_trying_another_large_request():
    policy = ThreeTierPolicy()
    tier = policy.choose(TWO_FAILURES)
    policy.mark_sent(tier)
    with pytest.raises(PolicyExhausted):
        policy.record_length(tier)
    assert not policy.long_code_used


@pytest.mark.parametrize("history", [[], TWO_FAILURES])
def test_transport_failure_never_retries_or_enters_rescue(history):
    policy = ThreeTierPolicy()
    tier = policy.choose(history)
    policy.mark_sent(tier)
    policy.record_transport_failure()
    for _ in range(2):
        with pytest.raises(PolicyExhausted, match="transport"):
            policy.choose(TWO_FAILURES)
    with pytest.raises(PolicyExhausted):
        policy.mark_sent(tier)
    assert policy.difficult_used == (tier.name == "difficult")
    assert not policy.long_code_used


def test_length_cannot_be_reported_before_request_sent():
    policy = ThreeTierPolicy()
    with pytest.raises(ValueError):
        policy.record_length(policy.choose([]))
    assert not policy.long_code_used
