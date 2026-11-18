"""Explicit-basis token accounting; unknown usage is never treated as free."""

from __future__ import annotations

import json
import math
import ipaddress
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class AgentPlannerError(RuntimeError):
    """A planner cannot safely continue this request."""


class BudgetExhausted(AgentPlannerError):
    """The maximum cost of another request does not fit the budget."""


class AccountingError(AgentPlannerError):
    """Actual cost cannot be established or violates the configured limits."""


class InvalidPlanError(AgentPlannerError):
    """The paid response did not contain an executable, valid plan."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key_env: str
    model_family: str
    model_release_date: str
    pricing_as_of: str
    input_rmb_per_million: float
    output_rmb_per_million: float
    price_source: str
    model_source: str
    verified: bool
    pricing_basis: str = "official_standard"
    response_format: str = "json_object"
    content_parser: str = "json"
    bypass_system_proxy: bool = False
    connect_ip: str | None = None
    accepted_response_models: list[str] = field(default_factory=list)
    budget_rmb: float = 20.0
    max_input_tokens: int = 16000
    max_output_tokens: int = 1000
    timeout_seconds: float = 60.0
    reasoning_effort: str | None = None

    @classmethod
    def load(cls, path: Path | str) -> "LLMConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("LLM config must be a JSON object.")
            config = cls(**raw)
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load LLM configuration: {type(exc).__name__}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.verified is not True:
            raise ValueError(
                "LLM config is not verified. Confirm the model release, selected "
                "pricing basis and source, then set verified=true."
            )
        if not isinstance(self.pricing_basis, str) or self.pricing_basis not in ("official_standard", "provider_current"):
            raise ValueError("pricing_basis must be official_standard or provider_current.")
        if not isinstance(self.response_format, str) or self.response_format not in ("json_object", "prompt_only"):
            raise ValueError("response_format must be json_object or prompt_only.")
        if self.content_parser not in ("json", "qwen_think_json"):
            raise ValueError("content_parser must be json or qwen_think_json.")
        if self.content_parser == "qwen_think_json" and self.model_family != "qwen":
            raise ValueError("qwen_think_json is only supported for Qwen configurations.")
        if type(self.bypass_system_proxy) is not bool:
            raise ValueError("bypass_system_proxy must be a boolean.")
        if self.connect_ip is not None:
            if not isinstance(self.connect_ip, str):
                raise ValueError("connect_ip must be an IP address or null.")
            try:
                ipaddress.ip_address(self.connect_ip)
            except ValueError as exc:
                raise ValueError("connect_ip must be a valid IP address.") from exc
            if not self.bypass_system_proxy or urlparse(self.base_url).scheme != "https":
                raise ValueError("A pinned connect_ip requires HTTPS and bypass_system_proxy=true.")
        if self.model_family not in {"qwen", "deepseek"}:
            raise ValueError("Only Qwen or DeepSeek is allowed by the assignment.")
        if self.reasoning_effort is not None:
            if not isinstance(self.reasoning_effort, str) or self.reasoning_effort not in ("none", "low", "high", "max"):
                raise ValueError("reasoning_effort must be null or one of none, low, high, max.")
            if self.model_family != "deepseek":
                raise ValueError("reasoning_effort is only supported for DeepSeek configurations.")
        try:
            release = date.fromisoformat(self.model_release_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("model_release_date must be an ISO calendar date.") from exc
        if release > date(2026, 10, 21):
            raise ValueError("Model must have been released on or before 2026-10-21.")
        try:
            pricing_date = date.fromisoformat(self.pricing_as_of)
            if pricing_date.isoformat() != self.pricing_as_of:
                raise ValueError("Use YYYY-MM-DD.")
        except (TypeError, ValueError) as exc:
            raise ValueError("pricing_as_of must be an ISO calendar date in YYYY-MM-DD form.") from exc
        if self.pricing_basis == "official_standard" and self.pricing_as_of != "2026-10-21":
            raise ValueError("Official standard pricing must be recorded as of 2026-10-21.")
        for name in ("model", "api_key_env", "price_source", "model_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "REPLACE" in value:
                raise ValueError(f"{name} must be completed with a real value.")
        if not isinstance(self.accepted_response_models, list) or any(
            not isinstance(model, str) or not model.strip() or "REPLACE" in model
            for model in self.accepted_response_models
        ):
            raise ValueError("accepted_response_models must be a list of nonempty, explicitly verified model aliases.")
        for name in ("base_url", "price_source", "model_source"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a URL.")
            parsed = urlparse(value)
            local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if (parsed.scheme != "https" and not local_http) or not parsed.hostname:
                raise ValueError(f"{name} requires HTTPS (HTTP is allowed for localhost).")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{name} must not contain credentials, query parameters, or fragments.")
        for name in ("input_rmb_per_million", "output_rmb_per_million", "budget_rmb", "timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if self.budget_rmb > 20:
            raise ValueError("The per-task LLM budget cannot exceed RMB 20.")
        for name in ("max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class BudgetLedger:
    """Append-only per-run ledger, with a pre-call maximum-cost reservation.

    A configured input-token ceiling is a conservative local estimate, not a
    provider-enforced cap. Any observed violation blocks further requests and
    remains visible in the ledger. Resume/retry after uncertain usage is not
    supported: reconcile it with the provider first.
    """

    def __init__(self, config: LLMConfig, log_path: Path | str):
        config.validate()
        self.config = config
        self.log_path = Path(log_path)
        if self.log_path.exists() and self.log_path.stat().st_size:
            raise AccountingError("Existing LLM ledger cannot be reset or resumed automatically; use a fresh run directory.")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.spent = Decimal("0")
        self.calls = 0
        self.pending: dict[str, Any] | None = None
        self.blocked = False
        self.unknown_usage = False

    def _cost(self, inputs: int, outputs: int) -> Decimal:
        return (
            Decimal(inputs) * Decimal(str(self.config.input_rmb_per_million))
            + Decimal(outputs) * Decimal(str(self.config.output_rmb_per_million))
        ) / Decimal(1_000_000)

    def _pricing_labels(self) -> dict[str, Any]:
        return {
            "pricing_basis": self.config.pricing_basis,
            "pricing_as_of": self.config.pricing_as_of,
            "price_source": self.config.price_source,
            # This flag describes only the selected pricing policy, not whether
            # the model, complete run, or submitted result is course compliant.
            "course_pricing_compliant": (
                self.config.pricing_basis == "official_standard"
                and self.config.pricing_as_of == "2026-10-21"
                and self.config.verified
            ),
            "cost_semantics": "Calculated from reported tokens and configured rates; not a provider invoice or confirmed account deduction.",
        }

    def _append(self, record: dict[str, Any]) -> None:
        # Every row remains interpretable even when the request never settles.
        record = {"time_utc": timestamp(), **self._pricing_labels(), **record}
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    def begin(self, messages: list[dict[str, str]]) -> str:
        if self.blocked or self.pending is not None:
            raise AccountingError("Further LLM requests are blocked by unresolved or invalid accounting.")
        # UTF-8 bytes plus generous message framing is conservative for normal
        # byte-based tokenizers. Provider-added/hidden tokens cannot be predicted.
        estimated_upper = sum(len(m["content"].encode("utf-8")) + 256 for m in messages) + 256
        if estimated_upper > self.config.max_input_tokens:
            raise AgentPlannerError("Planner context exceeds the configured conservative input-token bound.")
        reserve = self._cost(self.config.max_input_tokens, self.config.max_output_tokens)
        if self.spent + reserve > Decimal(str(self.config.budget_rmb)):
            raise BudgetExhausted("Remaining budget cannot reserve the next request's configured maximum cost.")
        self.calls += 1
        call_id = f"llm-{self.calls:04d}"
        self.pending = {"call_id": call_id, "reserved_rmb": float(reserve)}
        self._append({
            "event": "request_started", "call_id": call_id,
            "model_requested": self.config.model, "messages": messages,
            "max_output_tokens": self.config.max_output_tokens,
            "max_input_tokens": self.config.max_input_tokens,
            "estimated_input_upper_bound": estimated_upper,
            "reserved_rmb": float(reserve),
            "pricing": {
                "basis": self.config.pricing_basis,
                "as_of": self.config.pricing_as_of,
                "input_rmb_per_million": self.config.input_rmb_per_million,
                "output_rmb_per_million": self.config.output_rmb_per_million,
                "source": self.config.price_source,
                "cache_discounts_applied": False,
            },
        })
        return call_id

    def unresolved(self, reason: str, response: dict | None = None, diagnostic: dict | None = None) -> None:
        self.blocked = True
        self.unknown_usage = True
        self._append({
            "event": "accounting_unresolved", **(self.pending or {}),
            "reason": reason, "response": response,
            "diagnostic": diagnostic,
            "actual_cost_rmb": None,
            "calculated_cost_rmb": None,
            "provider_billed_cost_rmb": None,
            "accounted_cost_rmb": float(self.spent),
        })
        # Keep the reservation in summary; it is explicitly not actual usage.

    def settle(self, response: dict[str, Any]) -> None:
        if self.pending is None:
            raise AccountingError("No request is pending settlement.")
        response_model = response.get("model")
        accepted_models = {self.config.model, *self.config.accepted_response_models}
        if not isinstance(response_model, str) or response_model not in accepted_models:
            self.unresolved("Returned model is missing or not covered by the verified model/pricing configuration.", response)
            raise AccountingError("Unverified response model; token usage was preserved, but its cost is unknown. Further calls are blocked.")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            self.unresolved("Response did not provide usage statistics.", response)
            raise AccountingError("Missing LLM usage. Further API calls are blocked.")
        inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (inputs, outputs))
                or inputs == 0):
            self.unresolved("Response provided invalid token counts.", response)
            raise AccountingError("Invalid token usage. Further API calls are blocked.")
        cost = self._cost(inputs, outputs)
        self.spent += cost
        call = self.pending
        self.pending = None
        violated = (
            inputs > self.config.max_input_tokens
            or outputs > self.config.max_output_tokens
            or self.spent > Decimal(str(self.config.budget_rmb))
        )
        self._append({
            "event": "request_completed", **call,
            "model_requested": self.config.model,
            "model_returned": response.get("model"),
            "provider_response_id": response.get("id"),
            "input_tokens": inputs, "output_tokens": outputs,
            "usage": usage, "response": response,
            # Retain the old key for existing ledger consumers. Both values
            # describe token-based calculation, never verified provider billing.
            "actual_cost_rmb": float(cost),
            "calculated_cost_rmb": float(cost),
            "provider_billed_cost_rmb": None,
            "accounted_cost_rmb": float(self.spent),
            "reservation_limit_violated": violated,
        })
        if violated:
            self.blocked = True
            raise AccountingError("Provider usage exceeded the reserved limits; actual usage was logged and further calls are blocked.")

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "llm", "model": self.config.model,
            **self._pricing_labels(),
            "accepted_response_models": self.config.accepted_response_models,
            "model_family": self.config.model_family,
            "model_release_date": self.config.model_release_date,
            "model_source": self.config.model_source,
            "network_route": {
                "bypass_system_proxy": self.config.bypass_system_proxy,
                "connect_ip": self.config.connect_ip,
                "tls_server_name": urlparse(self.config.base_url).hostname,
            },
            "pricing_verified_by_user": self.config.verified,
            "configuration_verified": self.config.verified,
            "budget_rmb": self.config.budget_rmb,
            "accounted_cost_rmb": float(self.spent),
            "calculated_cost_rmb": float(self.spent),
            "provider_billed_cost_rmb": None,
            "within_budget": self.spent <= Decimal(str(self.config.budget_rmb)) and not self.unknown_usage,
            "budget_pricing_basis": self.config.pricing_basis,
            "request_count": self.calls,
            "pending_reserved_rmb": self.pending["reserved_rmb"] if self.pending else 0.0,
            "unknown_usage": self.unknown_usage,
            "blocked": self.blocked,
            "accounting_complete": not self.unknown_usage and self.pending is None,
            "input_bound_note": "Conservative byte estimate; provider-added tokens can exceed the local bound.",
        }
