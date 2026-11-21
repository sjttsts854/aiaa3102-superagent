"""Conservative development allocations, separate from actual token accounting.

Only documented transport/decode failures may retain their original reservation.
No tokens are fabricated and no unknown request is settled by this read-only tool.
The reservation is a planning allowance, not a guaranteed bound or provider bill.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from .reconciliation import _amount, _read_ledger
from .usage import LLMConfig
from .utils import file_hash


_TRANSPORT_ERRORS = {
    "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectError",
    "ReadError", "WriteError", "CloseError", "LocalProtocolError", "RemoteProtocolError",
    "ProxyError", "UnsupportedProtocol", "DecodingError", "TooManyRedirects",
    "HTTPStatusError", "ValueError", "JSONDecodeError",
    "RequestDeadlineExceeded", "RequestWorkerError",
}
_PATH_KEYS = {"train": "train_path", "train_labels": "train_labels_path",
              "validation": "validation_path", "validation_labels": "validation_labels_path",
              "train_images": "train_image_assets", "validation_images": "validation_image_assets"}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("History must contain a JSON object")
    return value


def _terminal(document: dict) -> None:
    if document.get("status") not in {
        "completed", "completed_with_warnings", "partial", "failed", "partial_accounting",
        "partial_unresolved_accounting", "partial_accounting_limits_violated",
        "development_completed_unresolved_accounting", "development_partial_unresolved_accounting",
    }:
        raise ValueError("Development history must be terminal, not running or missing status")
    if document.get("reconciliations"):
        raise ValueError("Reconciled history must use the strict accounting path")


def collect_development_history(paths: list[Path], data_hashes: dict,
                                pricing: set[tuple[str, str]]) -> dict:
    """Recompute known charges and retain separate original unresolved reserves.

    Each explicit root owns its complete ancestry. Flattened ancestors are counted
    once within that root, while overlapping explicit roots are rejected. Actual
    run ledgers, rather than campaign summary totals, determine the amounts.
    """
    roots = [Path(path).resolve() for path in paths]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate explicit development history")
    sources: set[Path] = set()
    runs: set[Path] = set()
    pinned_records: list[dict] = []

    def visit(path: Path, visited: set[Path], stack: set[Path]) -> set[Path]:
        path = path.resolve()
        if path in stack:
            raise ValueError("Development history contains a cycle")
        if path in visited:
            return set()
        visited.add(path)
        sources.add(path)
        document = _json(path)
        _terminal(document)
        if "runs" not in document:
            if path.name != "summary.json":
                raise ValueError("Standalone history must be the original run summary.json")
            return {path.parent}
        if document.get("data_sha256") != data_hashes or (
                document.get("pricing_basis"), document.get("pricing_as_of")) not in pricing:
            raise ValueError("Campaign history must match task data and pricing")
        pinned_records.extend(document.get("history_ledger_records", []))
        owned: set[Path] = set()
        stack = stack | {path}
        for ancestor in document.get("previous_campaigns", []):
            owned |= visit(Path(ancestor), visited, stack)
        entries = document["runs"]
        directories = [Path(record["output"]).resolve() for record in entries]
        if len(set(directories)) != len(directories):
            raise ValueError("Duplicate run entry in campaign history")
        for directory, record in zip(directories, entries):
            original = _json(directory / "summary.json")
            if record.get("summary", {}).get("status") != original.get("status"):
                raise ValueError("Embedded run status disagrees with the original summary")
            owned |= visit(directory / "summary.json", visited, stack)
        # New campaigns list inherited as well as current run directories.
        declared = [Path(directory).resolve() for directory in document.get("charged_run_directories", [])]
        if len(set(declared)) != len(declared):
            raise ValueError("Duplicate charged run directory")
        for directory in declared:
            owned |= visit(directory / "summary.json", visited, stack)
        return owned

    for root in roots:
        owned = visit(root, set(), set())
        if runs & owned:
            raise ValueError("Explicit development histories overlap in run ownership")
        runs |= owned

    known, reserved = Decimal("0"), Decimal("0")
    pending, ledger_records = [], []
    for directory in sorted(runs):
        summary = _json(directory / "summary.json")
        _terminal(summary)
        config_path, ledger_path = directory / "llm_config.snapshot.json", directory / "llm_calls.jsonl"
        config = LLMConfig.load(config_path)
        metadata = _json(directory / "manifest.json")
        hashes = {_PATH_KEYS[key]: value for key, value in metadata["data_sha256"].items() if key in _PATH_KEYS}
        if hashes != data_hashes or (config.pricing_basis, config.pricing_as_of) not in pricing:
            raise ValueError("Run history must match task data and pricing")
        usage = summary.get("usage") or _json(directory / "usage_summary.json")
        config_hash = file_hash(config_path)
        ledger_hash = file_hash(ledger_path) if ledger_path.is_file() else None
        if ledger_hash is None:
            if (type(usage.get("request_count")) is not int or usage["request_count"] != 0
                    or _amount(usage.get("accounted_cost_rmb")) != 0
                    or _amount(usage.get("pending_reserved_rmb")) != 0
                    or usage.get("accounting_complete") is not True
                    or usage.get("blocked") is not False or usage.get("unknown_usage") is not False):
                raise ValueError("Missing ledger is allowed only for an explicit zero-call completed account")
            calls, charged, violation = {}, Decimal("0"), False
        else:
            calls, charged, violation = _read_ledger(ledger_path, config)
            # The current-run ledger must stop at its first unresolved request.
            active, stopped, running_cost = None, False, Decimal("0")
            for row in map(json.loads, ledger_path.read_text(encoding="utf-8-sig").splitlines()):
                if row["event"] == "request_started":
                    if active is not None or stopped:
                        raise ValueError("Ledger continued with a pending or unresolved request")
                    if running_cost + _amount(row["reserved_rmb"]) > Decimal(str(config.budget_rmb)):
                        raise ValueError("Original request reservation exceeded its run budget")
                    active = row["call_id"]
                else:
                    if row["call_id"] != active:
                        raise ValueError("Ledger terminal event is out of request order")
                    active, stopped = None, row["event"] == "accounting_unresolved"
                    running_cost = _amount(row["accounted_cost_rmb"])
        if violation:
            raise ValueError("Actual token or budget limit violations cannot use development reservations")
        pending_for_run = Decimal("0")
        for call_id, call in calls.items():
            end, start = call["terminal"], call["start"]
            if end is not None and end["event"] == "request_completed":
                continue
            diagnostic = (end or {}).get("diagnostic")
            exception = diagnostic.get("exception_type") if isinstance(diagnostic, dict) else None
            if (end is None or end.get("response") is not None or exception not in _TRANSPORT_ERRORS
                    or end.get("reason") != f"Transport or response decoding failure: {exception}"):
                raise ValueError("Only documented transport/decode failures may retain development reserves")
            if exception == "HTTPStatusError" and (type(diagnostic.get("http_status")) is not int
                                                     or not 400 <= diagnostic["http_status"] <= 599):
                raise ValueError("HTTP failure requires a valid recorded error status")
            allocation = _amount(start["reserved_rmb"])
            if allocation <= 0:
                raise ValueError("Unknown usage requires a positive original reservation")
            pending_for_run += allocation
            pending.append({"run": str(directory), "call_id": call_id, "start": start["time_utc"],
                            "model": start["model_requested"], "reserved_rmb": float(allocation),
                            "ledger_sha256": ledger_hash, "actual_cost_rmb": None,
                            "input_tokens": None, "output_tokens": None})
        if (usage.get("request_count") != len(calls)
                or _amount(usage.get("accounted_cost_rmb")) != charged
                or _amount(usage.get("pending_reserved_rmb")) != pending_for_run):
            raise ValueError("Run usage summary disagrees with its original ledger")
        if file_hash(config_path) != config_hash or (ledger_hash and file_hash(ledger_path) != ledger_hash):
            raise ValueError("Development accounting input changed during verification")
        known += charged
        reserved += pending_for_run
        ledger_records.append({"run_directory": str(directory), "ledger_sha256": ledger_hash,
                               "config_sha256": config_hash, "known_cost_rmb": float(charged),
                               "unresolved_reserved_rmb": float(pending_for_run)})
    current_records = {Path(record["run_directory"]): record for record in ledger_records}
    for pinned in pinned_records:
        current = current_records.get(Path(pinned["run_directory"]).resolve())
        if current is None or any(pinned.get(key) != current[key] for key in (
                "ledger_sha256", "config_sha256", "known_cost_rmb", "unresolved_reserved_rmb")):
            raise ValueError("Inherited history fingerprints or allocations changed")
    return {"known_cost_rmb": float(known), "unresolved_reserved_rmb": float(reserved),
            "budget_allocation_rmb": float(known + reserved), "unresolved_calls": pending,
            "accounting_complete": not pending, "actual_total_cost_known": not pending,
            "course_pricing_compliant": False, "run_directories": [str(path) for path in sorted(runs)],
            "source_records": [str(path) for path in sorted(sources)], "ledger_records": ledger_records,
            "note": "Development allocation only. Reservations are not actual costs or guaranteed provider limits; unresolved tokens remain unknown."}
