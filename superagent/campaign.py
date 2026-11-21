"""Compare planner configurations autonomously under one cumulative task budget."""
from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from . import __version__

from .data import load_task
from .reconciliation import _read_ledger, verify_reconciliation
from .runner import run
from .usage import LLMConfig
from .utils import file_hash, now, object_hash, write_json


def _money(value) -> Decimal:
    try:
        result = Decimal(str(value))
        if isinstance(value, bool) or not result.is_finite() or result < 0:
            raise ValueError()
        return result
    except Exception as exc:
        raise ValueError("Invalid history accounting amount") from exc


def _previous_costs(paths: list[Path], reconciliation_paths: list[Path], data_hashes: dict,
                    pricing: set[tuple]) -> tuple[Decimal, set[str], set[str], list[dict]]:
    """Carry accounting only; never import a previous model or selection."""
    paths = [Path(p).resolve() for p in paths]
    explicit = [Path(p).resolve() for p in reconciliation_paths]
    if len(set(paths)) != len(paths) or len(set(explicit)) != len(explicit):
        raise ValueError("Duplicate previous campaign or reconciliation sidecar")
    documents = {path: json.loads(path.read_text(encoding="utf-8")) for path in paths}
    inherited = {}
    for past in documents.values():
        for record in past.get("reconciliations", []):
            path = Path(record["path"]).resolve()
            if path in inherited and inherited[path] != record["sha256"]:
                raise ValueError("Conflicting inherited reconciliation fingerprints")
            inherited[path] = record["sha256"]
    by_run, proof_records, ledger_hashes, provider_records = {}, {}, set(), set()
    for path in sorted(set(explicit) | set(inherited)):
        proof = verify_reconciliation(path)
        if path in inherited and proof["sidecar_sha256"] != inherited[path]:
            raise ValueError("Inherited reconciliation sidecar has changed")
        if proof["status"] != "completed" or not proof["usage"]["accounting_complete"] or proof["usage"]["blocked"]:
            raise ValueError("Reconciliation still has unresolved or blocked accounting")
        directory = str(Path(proof["run_directory"]).resolve())
        if directory in by_run or proof["ledger_sha256"] in ledger_hashes:
            raise ValueError("Duplicate reconciled run or ledger")
        references = {r["provider_record_reference"] for r in proof["resolved_calls"]}
        if provider_records & references:
            raise ValueError("Provider records cannot settle more than one run")
        provider_records |= references
        ledger_hashes.add(proof["ledger_sha256"])
        by_run[directory] = proof
        proof_records[directory] = {"path": str(path), "sha256": proof["sidecar_sha256"],
                                    "run_directory": directory, "ledger_sha256": proof["ledger_sha256"]}
    used_proofs, charged_runs, ancestors = set(), set(), set()
    total = Decimal("0")

    def history_directories(path: Path, visited: set[Path]) -> set[str]:
        """Recover identities from old flattened lineage without counting costs twice."""
        path = path.resolve()
        if path in visited:
            return set()
        visited.add(path)
        past = json.loads(path.read_text(encoding="utf-8"))
        if past.get("status") in {None, "running"}:
            raise ValueError("Previous history contains unfinished run/campaign records")
        if "cumulative_cost_rmb" not in past:
            return {str(path.parent)}
        result = {str(Path(r["output"]).resolve()) for r in past.get("runs", []) if "output" in r}
        for ancestor in past.get("previous_campaigns", []):
            result |= history_directories(Path(ancestor), visited)
        return result

    def run_metadata(directory: Path):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "llm_config.snapshot.json").read_text(encoding="utf-8"))
        mapping = {"train": "train_path", "train_labels": "train_labels_path",
                   "validation": "validation_path", "validation_labels": "validation_labels_path",
                   "train_images": "train_image_assets", "validation_images": "validation_image_assets"}
        hashes = {mapping[k]: v for k, v in manifest["data_sha256"].items() if k in mapping}
        if hashes != data_hashes or (config["pricing_basis"], config["pricing_as_of"]) not in pricing:
            raise ValueError("Previous run must match task data and pricing basis")

    def usage_cost(directory: Path, summary: dict, require_ledger: bool) -> Decimal:
        identity = str(directory.resolve())
        if summary.get("status") in {None, "running"}:
            raise ValueError("Previous run must be completed; unfinished history cannot be reused")
        original_summary = directory / "summary.json"
        if original_summary.is_file():
            original = json.loads(original_summary.read_text(encoding="utf-8"))
            if original.get("status") != summary.get("status"):
                raise ValueError("Previous run status disagrees with its original record; unfinished history cannot be reused")
        if identity in by_run:
            # A sidecar may settle cost, never change the old training outcome.
            if summary["status"] not in {"completed", "partial_unresolved_accounting"}:
                raise ValueError("Reconciliation cannot override this previous training status")
            run_metadata(directory)
            used_proofs.add(identity)
            return _money(by_run[identity]["usage"]["accounted_cost_rmb"])
        usage = summary.get("usage")
        if not usage:
            usage = json.loads((directory / "usage_summary.json").read_text(encoding="utf-8"))
        if usage.get("blocked", False) or usage.get("unknown_usage", False) or usage.get("accounting_complete") is not True or _money(usage.get("pending_reserved_rmb", 0)) != 0:
            raise ValueError("Previous run has blocked or unresolved accounting")
        amount = _money(usage.get("accounted_cost_rmb"))
        ledger = directory / "llm_calls.jsonl"
        if require_ledger or ledger.is_file():
            run_metadata(directory)
            calls, recomputed, violation = _read_ledger(ledger, LLMConfig.load(directory / "llm_config.snapshot.json"))
            if violation or any(call["terminal"] is None or call["terminal"]["event"] != "request_completed" for call in calls.values()):
                raise ValueError("Previous run ledger has blocked or unresolved accounting")
            if amount != recomputed:
                raise ValueError("Previous run summary disagrees with its ledger")
        return amount

    for path, past in documents.items():
        status = past.get("status")
        is_campaign = "cumulative_cost_rmb" in past
        if status not in ({"completed", "partial_accounting"} if is_campaign else {"completed", "partial_unresolved_accounting"}):
            raise ValueError("Previous run/campaign must be completed; unfinished or unresolved history cannot be reused")
        lineage = {str(path), *(str(Path(p).resolve()) for p in past.get("previous_campaigns", []))}
        if ancestors & lineage:
            raise ValueError("Previous campaign lineage overlaps; pass only its latest cumulative summary")
        ancestors |= lineage
        declared_runs = past.get("charged_run_directories", [])
        if len(set(declared_runs)) != len(declared_runs):
            raise ValueError("Duplicate charged run in history")
        owned_runs = {str(Path(p).resolve()) for p in declared_runs}
        for ancestor in past.get("previous_campaigns", []):
            owned_runs |= history_directories(Path(ancestor), set())
        if is_campaign:
            if past["data_sha256"] != data_hashes or (past["pricing_basis"], past["pricing_as_of"]) not in pricing:
                raise ValueError("Previous campaign must match task data and pricing basis")
            if past.get("blocked", False) or past.get("usage", {}).get("blocked", False):
                raise ValueError("Previous campaign has blocked accounting")
            records = past.get("runs", [])
            directories = [str(Path(r["output"]).resolve()) for r in records if "output" in r]
            if len(set(directories)) != len(directories):
                raise ValueError("Duplicate previous run; costs cannot be counted twice")
            if len(directories) != len(records):
                raise ValueError("Previous campaign has blocked or missing run accounting")
            owned_runs |= set(directories)
            if status == "partial_accounting" and not (set(directories) & set(by_run)):
                raise ValueError("Previous campaign has unresolved usage; matching reconciliation is required")
            if status == "completed" and past.get("accounting_complete") is not True:
                raise ValueError("Previous campaign has unresolved usage")
            # Pre-campaign-only legacy summaries have no runs/previous_cost field.
            carried = _money(past.get("previous_cost_rmb", past["cumulative_cost_rmb"] if not records else 0))
            recorded_total, recomputed_total = carried, carried
            for record in records:
                summary = record["summary"]
                recorded_total += _money(summary.get("usage", {}).get("accounted_cost_rmb"))
                recomputed_total += usage_cost(Path(record["output"]), summary, status == "partial_accounting")
            if recorded_total != _money(past["cumulative_cost_rmb"]):
                raise ValueError("Previous campaign total disagrees with its nested run costs")
            total += recomputed_total
        else:
            if status != "completed" and str(path.parent) not in by_run:
                raise ValueError("Previous run has unresolved usage; matching reconciliation is required")
            run_metadata(path.parent)
            total += usage_cost(path.parent, past, status != "completed")
            owned_runs.add(str(path.parent))
        if charged_runs & owned_runs:
            raise ValueError("Previous histories contain duplicate charged runs")
        charged_runs |= owned_runs
        # Recheck proofs already included in a previous campaign's carried cost.
        used_proofs |= owned_runs & set(by_run)
    if set(by_run) != used_proofs:
        raise ValueError("Reconciliation does not match any supplied previous history")
    return total, ancestors, charged_runs, [proof_records[key] for key in sorted(used_proofs)]


def campaign(task_path: Path, configs: list[Path], output: Path, budget_rmb: float = 1.0,
             previous: list[Path] | None = None, experiment_timeout: float = 180,
             refit_timeout: float = 300, reconciliations: list[Path] | None = None,
             development_reserves: bool = False, planner_mode: str = "llm") -> dict:
    """No test paths are permitted. Winner selection uses validation scores only.

    Previous campaign ledgers consume the same cumulative development budget.
    Default mode requires exact accounting. Explicit development mode carries
    unresolved transport reservations separately; it never asserts exact cost.
    """
    if type(budget_rmb) not in (float, int) or not math.isfinite(budget_rmb) or not 0 < budget_rmb <= 20:
        raise ValueError("Campaign budget must be positive and at most RMB 20")
    if planner_mode not in {"llm", "llm-code"}:
        raise ValueError("Campaign planner must be llm or llm-code")
    task = load_task(task_path)
    if task.predict_path is not None:
        raise ValueError("Campaign task must omit predict_path; evaluate only after selection")
    if not configs:
        raise ValueError("At least one LLM configuration is required")
    settings = [LLMConfig.load(path) for path in configs]
    if planner_mode != "llm-code" and any(config.request_policy is not None for config in settings):
        raise ValueError("The three-tier request policy requires --planner llm-code for experiments.")
    pricing = {(s.pricing_basis, s.pricing_as_of) for s in settings}
    if len(pricing) != 1:
        raise ValueError("Campaign configurations must use the same pricing basis and date")
    data_hashes = {key: file_hash(getattr(task, key)) for key in (
        "train_path", "train_labels_path", "validation_path", "validation_labels_path"
    ) if getattr(task, key, None) is not None}
    if task.data_type == "image":
        from .data import image_source_hashes
        for split in ("train", "validation"):
            image_csv = getattr(task, f"{split}_path", None)
            if image_csv is not None:
                data_hashes[f"{split}_image_assets"] = object_hash(image_source_hashes(task, image_csv))
    unresolved_calls = []
    history_records = []
    reserved = Decimal("0")
    if development_reserves:
        if reconciliations:
            raise ValueError("Use strict accounting mode to apply reconciliation evidence")
        from .development_accounting import collect_development_history
        history = collect_development_history(previous or [], data_hashes, pricing)
        prior_cost = Decimal(str(history["known_cost_rmb"]))
        reserved = Decimal(str(history["unresolved_reserved_rmb"]))
        unresolved_calls = history["unresolved_calls"]
        ancestor_paths = set(history["source_records"])
        charged_runs = set(history["run_directories"])
        history_records = history.get("ledger_records", [])
        reconciliation_records = []
    else:
        prior_cost, ancestor_paths, charged_runs, reconciliation_records = _previous_costs(
            previous or [], reconciliations or [], data_hashes, pricing)
    if prior_cost + reserved >= Decimal(str(budget_rmb)):
        raise ValueError("Cumulative task budget is exhausted")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "project_version": __version__,
        "status": "running", "created_at": now(), "task_path": str(task_path.resolve()),
        "data_sha256": data_hashes, "pricing_basis": settings[0].pricing_basis,
        "pricing_as_of": settings[0].pricing_as_of, "budget_rmb": budget_rmb,
        "previous_campaigns": sorted(ancestor_paths), "previous_cost_rmb": float(prior_cost),
        "charged_run_directories": sorted(charged_runs), "reconciliations": reconciliation_records,
        "campaign_cost_rmb": 0.0, "cumulative_cost_rmb": float(prior_cost),
        "accounting_complete": not unresolved_calls, "runs": [], "selection": None,
        "development_reserves": development_reserves,
        "planner_mode": planner_mode,
        "execution_complete": False,
        "formal_evaluation_eligible": False,
        "unresolved_calls": unresolved_calls, "unresolved_reserved_rmb": float(reserved),
        "budget_committed_rmb": float(prior_cost + reserved), "history_ledger_records": history_records,
        "course_pricing_compliant": False if development_reserves else all(s.pricing_basis == "official_standard" for s in settings),
        "cost_note": "Known calculated cost and unresolved reservations are separate. Reservations are budget controls, not actual charges or guaranteed provider bounds.",
        "test_evaluated": False, "selection_rule": "highest successful validation score; ties favor earlier run",
    }
    summary_path = output / "campaign.json"
    write_json(summary_path, result)
    spent = prior_cost
    for index, (config_path, config) in enumerate(zip(configs, settings), 1):
        remaining = Decimal(str(budget_rmb)) - spent - reserved
        if remaining <= 0:
            break
        folder = output / f"planner_{index:02d}"
        print(f"Planner {index}/{len(configs)}: {config.model}", flush=True)
        try:
            summary = run(task_path, folder, planner_mode, config_path, experiment_timeout,
                          refit_timeout, budget_limit=min(float(remaining), config.budget_rmb))
        except Exception as exc:
            usage_path = folder / "usage_summary.json"
            usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else None
            summary = {"status": "failed", "error_type": type(exc).__name__, "usage": usage or {}}
        usage = summary.get("usage", {})
        spent += Decimal(str(usage.get("accounted_cost_rmb", 0)))
        continuable_transport_failure = False
        if development_reserves and not usage.get("accounting_complete", False):
            from .development_accounting import collect_development_history
            try:
                current = collect_development_history([folder / "summary.json"], data_hashes, pricing)
                reserved += Decimal(str(current["unresolved_reserved_rmb"]))
                unresolved_calls.extend(current["unresolved_calls"])
                continuable_transport_failure = bool(current["unresolved_calls"])
            except (ValueError, OSError, KeyError):
                continuable_transport_failure = False
        if not continuable_transport_failure and not usage.get("accounting_complete", False):
            pending = _money(usage.get("pending_reserved_rmb", 0))
            reserved += pending
            unresolved_calls.append({"run": str(folder), "reserved_rmb": float(pending),
                                     "actual_cost_rmb": None, "input_tokens": None, "output_tokens": None,
                                     "development_continuation_allowed": False})
        record = {"model": config.model, "output": str(folder), "summary": summary}
        result["runs"].append(record)
        charged_runs.add(str(folder.resolve()))
        result["charged_run_directories"] = sorted(charged_runs)
        result["cumulative_cost_rmb"] = float(spent)
        result["campaign_cost_rmb"] = float(spent - prior_cost)
        result["unresolved_reserved_rmb"] = float(reserved)
        result["budget_committed_rmb"] = float(spent + reserved)
        result["unresolved_calls"] = unresolved_calls
        result["accounting_complete"] = not unresolved_calls and bool(usage.get("accounting_complete", False))
        selectable = {"completed", "completed_with_warnings"} if development_reserves else {"completed"}
        complete = (summary.get("status") in selectable and summary.get("llm_requests", 0) > 0
                    and summary.get("valid_llm_decisions", 0) > 0
                    and usage.get("accounting_complete", False) and not usage.get("blocked", False))
        if complete and (result["selection"] is None or summary["best_validation_score"] > result["selection"]["validation_score"]):
            result["selection"] = {"model_path": summary["model"], "planner_model": config.model,
                "validation_score": summary["best_validation_score"], "metric": task.metric, "candidate": summary["selected_candidate"],
                "source_sha256": summary["source_sha256"], "run_directory": str(folder)}
        write_json(summary_path, result)
        if (not usage.get("accounting_complete", False) or usage.get("blocked", False)) and not continuable_transport_failure:
            result["status"] = "partial_accounting"
            break
    if result["status"] == "running":
        completed_statuses = {"completed", "completed_with_warnings"} if development_reserves else {"completed"}
        execution_complete = (len(result["runs"]) == len(configs) and result["selection"] is not None
                              and all(r["summary"].get("status") in completed_statuses
                                      and r["summary"].get("valid_llm_decisions", 0) > 0 for r in result["runs"]))
        result["execution_complete"] = execution_complete
        if development_reserves and unresolved_calls:
            result["status"] = "development_completed_unresolved_accounting" if execution_complete else "development_partial_unresolved_accounting"
        else:
            result["status"] = "completed" if execution_complete else "partial"
    source_hashes = {r["summary"].get("source_sha256") for r in result["runs"] if r["summary"].get("source_sha256")}
    if len(source_hashes) > 1:
        result["status"] = "partial_source_changed"
    result["finished_at"] = now()
    write_json(summary_path, result)
    return result
