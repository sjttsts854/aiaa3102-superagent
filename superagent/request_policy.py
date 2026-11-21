"""Pure, run-scoped request tiers for the three-tier experiment.

Choosing a tier never spends its allowance. The caller must mark it sent only
after budget checks, and must retain its own usage ledger on every outcome.
There is no transport, sleep, accounting reset, or automatic retry here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any


class PolicyExhausted(RuntimeError):
    """This run must stop requesting candidates under the experimental policy."""


@dataclass(frozen=True)
class TierDecision:
    name: str
    thinking: bool
    max_output_tokens: int
    deadline_seconds: int
    reason: str


_SETTINGS = {
    "default": (False, 8192, 240),
    "long_code": (False, 16384, 360),
    "difficult": (True, 16384, 600),
}


def _decision(name: str, reason: str) -> TierDecision:
    return TierDecision(name, *_SETTINGS[name], reason)


def _source_id(entry: Mapping[str, Any]) -> str | None:
    source = entry["candidate"].get("source")
    if not isinstance(source, str) or not source.strip():
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _difficulty_reason(history: Sequence[Mapping[str, Any]]) -> str | None:
    generated = [
        entry for entry in history
        if isinstance(entry, Mapping)
        and isinstance(entry.get("candidate"), Mapping)
        and entry["candidate"].get("algorithm") == "generated_python"
    ]
    # A single execution error is repaired first without thinking. Two different
    # failed modules establish that the feedback-driven repair also failed.
    if len(generated) >= 2:
        previous, latest = generated[-2:]
        previous_id, latest_id = _source_id(previous), _source_id(latest)
        if (previous.get("status") == latest.get("status") == "failed"
                and previous_id is not None and latest_id is not None
                and previous_id != latest_id):
            return "Two consecutive distinct generated modules failed execution after a feedback-driven repair."

    baseline = next((
        _score(entry.get("score")) for entry in history
        if isinstance(entry, Mapping)
        and isinstance(entry.get("candidate"), Mapping)
        and entry["candidate"].get("algorithm") != "generated_python"
        and entry.get("status") == "success"
        and _score(entry.get("score")) is not None
    ), None)
    if baseline is None:
        return None
    successful_sources = set()
    for entry in generated:
        if entry.get("status") != "success":
            continue
        value = _score(entry.get("score"))
        identity = _source_id(entry)
        if value is None or identity is None:
            continue
        # Higher is better for the validation metrics accepted by the runner.
        if value > baseline:
            return None
        successful_sources.add(identity)
    if len(successful_sources) >= 3:
        return "At least three distinct successful generated modules have not exceeded the baseline validation score."
    return None


class ThreeTierPolicy:
    """A sequential state machine whose allowances last for one whole run."""

    def __init__(self) -> None:
        self.long_code_used = False
        self.difficult_used = False
        self._long_code_pending = False
        self._last_sent: TierDecision | None = None
        self._terminal_reason: str | None = None

    def _check_active(self) -> None:
        if self._terminal_reason is not None:
            raise PolicyExhausted(self._terminal_reason)

    def choose(self, history: Sequence[Mapping[str, Any]]) -> TierDecision:
        """Select settings without consuming a one-time allowance or doing I/O."""
        self._check_active()
        if self._long_code_pending:
            return _decision("long_code", "The previous default response was truncated; use the single longer non-thinking request.")
        reason = _difficulty_reason(history)
        if reason is not None and not self.difficult_used:
            return _decision("difficult", reason)
        if self._last_sent is not None and self._last_sent.name == "difficult":
            reason = "The single thinking rescue has been sent; return to the default request settings."
        elif reason is not None:
            reason = "Difficulty persists, but the run's single thinking rescue is already used; keep default settings."
        else:
            reason = "Use default settings; execution history does not yet justify the single thinking rescue."
        return _decision("default", reason)

    def mark_sent(self, tier: TierDecision) -> None:
        """Consume an allowance only at the actual request-send boundary."""
        self._check_active()
        if tier.name not in _SETTINGS or (tier.thinking, tier.max_output_tokens, tier.deadline_seconds) != _SETTINGS[tier.name]:
            raise ValueError("Unrecognized request tier settings.")
        if tier.name == "long_code":
            if self.long_code_used or not self._long_code_pending:
                raise PolicyExhausted("The one-time long-code request is unavailable.")
            self.long_code_used = True
            self._long_code_pending = False
        elif self._long_code_pending:
            raise ValueError("The pending length recovery must be selected before another tier.")
        elif tier.name == "difficult":
            if self.difficult_used:
                raise PolicyExhausted("The one-time thinking rescue has already been sent.")
            self.difficult_used = True
        self._last_sent = tier

    def record_length(self, tier: TierDecision) -> TierDecision:
        """Schedule the single length recovery, or terminate on further length.

        This returns the next decision for logging/convenience. It does not send
        that request or consume its allowance; the next choose() returns it too.
        A thinking response already has the larger allowance and cannot upgrade.
        """
        self._check_active()
        if tier != self._last_sent:
            raise ValueError("A length response must belong to the last sent request.")
        if tier.name == "default" and not self.long_code_used and not self._long_code_pending:
            self._long_code_pending = True
            return self.choose(())
        self._terminal_reason = "Response was truncated after the available length allowance; stop instead of repeatedly increasing tokens."
        raise PolicyExhausted(self._terminal_reason)

    def record_transport_failure(self) -> None:
        """Stop the run on an ambiguous transport failure; never retry it here."""
        self._terminal_reason = "A request failed at the transport layer; stop without retrying or clearing its usage reservation."
