"""Import provider evidence into a separate accounting record, never a raw ledger.

The importer checks identity, timestamps, fingerprints, schema and arithmetic.
It cannot authenticate a provider export or a support statement: the caller must
obtain and review genuine evidence. A timeout or an absent record alone never
proves that a request was free. No network calls or inference retries occur here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .usage import LLMConfig
from .utils import file_hash, now, write_json


def _object(value: Any, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - (optional or set()):
        raise ValueError("Evidence contains missing or unsupported fields")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _time(value: Any) -> datetime:
    try:
        result = datetime.fromisoformat(_text(value, "timestamp").replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError()
        return result.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ValueError("Evidence timestamps require an explicit timezone") from exc


def _amount(value: Any) -> Decimal:
    try:
        amount = Decimal(str(value))
        if isinstance(value, bool) or not amount.is_finite() or amount < 0:
            raise ValueError()
        return amount
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Ledger contains an invalid monetary amount") from exc


def _tokens(inputs: Any, outputs: Any) -> tuple[int, int]:
    if type(inputs) is not int or inputs <= 0 or type(outputs) is not int or outputs < 0:
        raise ValueError("Completed usage requires positive input_tokens and nonnegative output_tokens integers")
    return inputs, outputs


def _cost(config: LLMConfig, inputs: int, outputs: int) -> Decimal:
    return (Decimal(inputs) * Decimal(str(config.input_rmb_per_million))
            + Decimal(outputs) * Decimal(str(config.output_rmb_per_million))) / Decimal(1_000_000)


def _read_ledger(path: Path, config: LLMConfig) -> tuple[dict, Decimal, bool]:
    """Recompute existing charges and validate identities instead of trusting totals."""
    calls: dict[str, dict] = {}
    spent = Decimal("0")
    limits_violated = False
    accepted = {config.model, *config.accepted_response_models}
    reserved = _cost(config, config.max_input_tokens, config.max_output_tokens)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Invalid ledger row")
        if (row.get("pricing_basis"), row.get("pricing_as_of"), row.get("price_source")) != (
                config.pricing_basis, config.pricing_as_of, config.price_source):
            raise ValueError("Ledger pricing does not match its configuration snapshot")
        event = row.get("event")
        call_id = _text(row.get("call_id"), "call_id")
        _time(row.get("time_utc"))
        if event == "request_started":
            if call_id in calls:
                raise ValueError("Duplicate request_started in ledger")
            pricing = row.get("pricing", {})
            if (row.get("model_requested") != config.model
                    or row.get("max_input_tokens") != config.max_input_tokens
                    or row.get("max_output_tokens") != config.max_output_tokens
                    or _amount(row.get("reserved_rmb")) != reserved
                    or pricing.get("basis") != config.pricing_basis
                    or pricing.get("as_of") != config.pricing_as_of
                    or pricing.get("source") != config.price_source
                    or _amount(pricing.get("input_rmb_per_million")) != Decimal(str(config.input_rmb_per_million))
                    or _amount(pricing.get("output_rmb_per_million")) != Decimal(str(config.output_rmb_per_million))):
                raise ValueError("Ledger request does not match its configuration snapshot")
            calls[call_id] = {"start": row, "terminal": None}
            continue
        if event not in {"request_completed", "accounting_unresolved"} or call_id not in calls:
            raise ValueError("Unknown ledger event or terminal event without a start")
        call = calls[call_id]
        if call["terminal"] is not None:
            raise ValueError("Duplicate terminal event in ledger")
        if _time(row["time_utc"]) < _time(call["start"]["time_utc"]):
            raise ValueError("Ledger terminal event predates its request")
        if _amount(row.get("reserved_rmb")) != reserved:
            raise ValueError("Ledger reservation changed after request start")
        call["terminal"] = row
        if event == "request_completed":
            response = row.get("response")
            if not isinstance(response, dict) or response.get("model") not in accepted:
                raise ValueError("Completed ledger response has an unverified model")
            if row.get("model_requested") != config.model or row.get("model_returned") != response["model"]:
                raise ValueError("Completed ledger model identity mismatch")
            usage = response.get("usage", {})
            inputs, outputs = _tokens(usage.get("prompt_tokens"), usage.get("completion_tokens"))
            if row.get("input_tokens") != inputs or row.get("output_tokens") != outputs or row.get("usage") != usage:
                raise ValueError("Completed ledger token counts disagree with raw response")
            cost = _cost(config, inputs, outputs)
            spent += cost
            if any(_amount(row.get(key)) != cost for key in ("actual_cost_rmb", "calculated_cost_rmb")):
                raise ValueError("Completed ledger cost disagrees with token arithmetic")
            violated = inputs > config.max_input_tokens or outputs > config.max_output_tokens or spent > Decimal(str(config.budget_rmb))
            if type(row.get("reservation_limit_violated")) is not bool or row["reservation_limit_violated"] != violated:
                raise ValueError("Ledger reservation violation flag is inconsistent")
            limits_violated |= violated
        elif row.get("actual_cost_rmb") is not None or row.get("calculated_cost_rmb") is not None:
            raise ValueError("An unresolved request cannot have a known actual cost")
        if _amount(row.get("accounted_cost_rmb")) != spent:
            raise ValueError("Ledger running total is inconsistent")
    if not calls:
        raise ValueError("Ledger has no requests")
    return calls, spent, limits_violated


def _build_reconciliation(run_dir: str | Path, evidence_path: str | Path) -> tuple[dict, bytes]:
    """Validate external evidence and emit an immutable accounting-only sidecar.

    Evidence JSON schema (all listed fields required):
      schema_version: 1
      ledger_sha256, config_sha256: hashes of the original run files
      provenance: {kind, reference, recorded_at, reviewed_by}
      resolutions: [
        {call_id, requested_model, request_started_at, provider_record_reference,
         status: "completed", provider_model, input_tokens, output_tokens}
        OR {same identity fields, status: "not_executed", confirmation}
      ]

    Provenance kind is provider_export, provider_support_confirmation, or
    management_portal_record. References identify the externally obtained record;
    they are not fetched. Explicit not_executed confirmations are the only zero-
    usage resolution. The entire evidence file is pinned and copied to output.
    Subsets may be imported, but missing evidence leaves accounting blocked.
    A later import should provide the full evidence set into a new directory.
    """
    run = Path(run_dir).resolve()
    evidence_file = Path(evidence_path).resolve()
    ledger_path, config_path = run / "llm_calls.jsonl", run / "llm_config.snapshot.json"
    fingerprints = {"ledger_sha256": file_hash(ledger_path), "config_sha256": file_hash(config_path),
                    "evidence_sha256": file_hash(evidence_file)}
    evidence_bytes = evidence_file.read_bytes()
    evidence = _object(json.loads(evidence_bytes.decode("utf-8-sig")),
                       {"schema_version", "ledger_sha256", "config_sha256", "provenance", "resolutions"})
    if type(evidence["schema_version"]) is not int or evidence["schema_version"] != 1:
        raise ValueError("Unsupported reconciliation evidence schema_version")
    if any(evidence[key] != fingerprints[key] for key in ("ledger_sha256", "config_sha256")):
        raise ValueError("Evidence fingerprints do not match the original ledger/configuration")
    provenance = _object(evidence["provenance"], {"kind", "reference", "recorded_at", "reviewed_by"})
    if provenance["kind"] not in {"provider_export", "provider_support_confirmation", "management_portal_record"}:
        raise ValueError("Evidence must originate from provider records or explicit provider confirmation")
    for field in ("reference", "reviewed_by"):
        _text(provenance[field], field)
    evidence_time = _time(provenance["recorded_at"])
    if not isinstance(evidence["resolutions"], list) or not evidence["resolutions"]:
        raise ValueError("Evidence needs at least one resolution")
    config = LLMConfig.load(config_path)
    calls, known_cost, limits_violated = _read_ledger(ledger_path, config)
    unresolved = {call_id: call for call_id, call in calls.items()
                  if call["terminal"] is None or call["terminal"]["event"] == "accounting_unresolved"}
    resolved: dict[str, dict] = {}
    provider_references: set[str] = set()
    added_cost = Decimal("0")
    for record in evidence["resolutions"]:
        identity = {"call_id", "requested_model", "request_started_at", "provider_record_reference", "status"}
        if not isinstance(record, dict) or record.get("status") not in {"completed", "not_executed"}:
            raise ValueError("Resolution must explicitly state completed or not_executed")
        required = identity | ({"provider_model", "input_tokens", "output_tokens"} if record["status"] == "completed" else {"confirmation"})
        _object(record, required)
        call_id = _text(record["call_id"], "call_id")
        if call_id not in unresolved or call_id in resolved:
            raise ValueError("Resolution must match a unique unresolved call")
        start = unresolved[call_id]["start"]
        if record["requested_model"] != start["model_requested"] or _time(record["request_started_at"]) != _time(start["time_utc"]):
            raise ValueError("Resolution model or request start time does not match the ledger")
        if evidence_time < _time(start["time_utc"]):
            raise ValueError("Provider evidence predates the request")
        reference = _text(record["provider_record_reference"], "provider_record_reference")
        if reference in provider_references:
            raise ValueError("Each request needs a distinct provider record reference")
        provider_references.add(reference)
        violation = False
        if record["status"] == "completed":
            if record["provider_model"] not in {config.model, *config.accepted_response_models}:
                raise ValueError("Evidence model is not covered by the verified pricing configuration")
            inputs, outputs = _tokens(record["input_tokens"], record["output_tokens"])
            cost = _cost(config, inputs, outputs)
            violation = inputs > config.max_input_tokens or outputs > config.max_output_tokens
        else:
            _text(record["confirmation"], "confirmation")
            cost = Decimal("0")
        added_cost += cost
        limits_violated |= violation
        resolved[call_id] = {**record, "calculated_cost_rmb": float(cost),
                             "reservation_limit_violated": violation}
    total = known_cost + added_cost
    limits_violated |= total > Decimal(str(config.budget_rmb))
    outstanding = set(unresolved) - set(resolved)
    pending = sum((_amount(unresolved[key]["start"]["reserved_rmb"]) for key in outstanding), Decimal("0"))
    complete = not outstanding
    blocked = not complete or limits_violated
    status = "partial_accounting_limits_violated" if limits_violated else "completed" if complete else "partial_unresolved_accounting"
    summary = {
        "status": status, "purpose": "accounting_reconciliation_only", "mode": "llm",
        "model": config.model, "pricing_basis": config.pricing_basis,
        "pricing_as_of": config.pricing_as_of, "price_source": config.price_source,
        "budget_rmb": config.budget_rmb, "request_count": len(calls),
        "accounted_cost_rmb": float(total), "calculated_cost_rmb": float(total),
        "previous_recorded_cost_rmb": float(known_cost), "resolved_cost_rmb": float(added_cost),
        "provider_billed_cost_rmb": None, "pending_reserved_rmb": float(pending),
        "accounting_complete": complete, "unknown_usage": not complete, "blocked": blocked,
        "within_budget": complete and total <= Decimal(str(config.budget_rmb)),
        "reservation_limit_violated": limits_violated,
        "resume_allowed": complete and not blocked,
        "course_pricing_compliant": config.pricing_basis == "official_standard" and config.pricing_as_of == "2026-10-21" and config.verified,
        "cost_semantics": "Calculated from logged or externally evidenced tokens and frozen configuration rates; not a provider invoice.",
    }
    result = {"schema_version": 1, "created_at": now(), "status": status,
              "run_directory": str(run), "ledger": str(ledger_path), "config": str(config_path),
              "evidence_file": str(evidence_file), **fingerprints, "provenance": provenance,
              "resolved_calls": list(resolved.values()), "unresolved_call_ids": sorted(outstanding),
              "usage": summary, "new_api_calls": 0, "original_files_modified": False,
              "note": "Structural validation and arithmetic only; caller is responsible for authentic provider evidence. Reconciliation does not make an incomplete training run complete."}
    if any(file_hash(path) != fingerprints[key] for path, key in (
            (ledger_path, "ledger_sha256"), (config_path, "config_sha256"), (evidence_file, "evidence_sha256"))):
        raise ValueError("Input evidence or original records changed during reconciliation")
    return result, evidence_bytes


def reconcile_run_usage(run_dir: str | Path, evidence_path: str | Path,
                        output_dir: str | Path) -> dict:
    """Write an immutable sidecar using the evidence schema in _build_reconciliation."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Reconciliation output already exists; choose a new directory")
    result, evidence_bytes = _build_reconciliation(run_dir, evidence_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_bytes(evidence_bytes)
    write_json(output / "reconciliation.json", result)
    write_json(output / "usage_summary.json", result["usage"])
    return result


def verify_reconciliation(sidecar_path: str | Path) -> dict:
    """Revalidate original files, copied evidence and sidecar arithmetic read-only.

    This is deliberately a replay of evidence, not trust in a sidecar's total.
    The original external evidence path may have moved; the pinned copy must
    still exist beside reconciliation.json. No files or API calls are created.
    """
    sidecar = Path(sidecar_path).resolve()
    sidecar_hash = file_hash(sidecar)
    document = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or "run_directory" not in document:
        raise ValueError("Expected a reconciliation sidecar")
    current, _ = _build_reconciliation(document["run_directory"], sidecar.parent / "evidence.json")
    # Creation time and the historical external path are descriptive provenance;
    # every identity, fingerprint, resolution, amount and state is recomputed.
    descriptive = {"created_at", "evidence_file"}
    if set(document) != set(current) or any(document[key] != current[key] for key in set(current) - descriptive):
        raise ValueError("Reconciliation sidecar does not match recomputed evidence and original records")
    _time(document["created_at"])
    _text(document["evidence_file"], "evidence_file")
    if file_hash(sidecar) != sidecar_hash:
        raise ValueError("Reconciliation sidecar changed during validation")
    result = {**document, "usage": current["usage"], "sidecar_path": str(sidecar),
              "sidecar_sha256": sidecar_hash}
    return result
