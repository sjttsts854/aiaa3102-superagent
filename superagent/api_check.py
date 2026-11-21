"""Separate local configuration validation from a small, metered network probe."""
from pathlib import Path
from dataclasses import asdict

from .credentials import read_api_key
from .planner import LLMPlanner
from .request_policy import ThreeTierPolicy
from .usage import LLMConfig
from .utils import now, write_json


def check_config(path: Path) -> dict:
    config = LLMConfig.load(path)
    key_present = bool(read_api_key(config.api_key_env))
    return {
        "configuration_valid": True,
        "credentials_present": key_present,
        "ready_for_live_check": key_present,
        "endpoint": config.base_url.rstrip("/") + "/chat/completions",
        "model": config.model,
        "accepted_response_models": config.accepted_response_models,
        "pricing_basis": config.pricing_basis,
        "pricing_as_of": config.pricing_as_of,
        "input_rmb_per_million": config.input_rmb_per_million,
        "output_rmb_per_million": config.output_rmb_per_million,
        "api_key_env": config.api_key_env,
        "network_request_sent": False,
        "bypass_system_proxy": config.bypass_system_proxy,
        "connect_ip": config.connect_ip,
        "request_policy": config.request_policy,
        "output_reservation_ceiling": config.max_output_tokens,
        "default_request": asdict(ThreeTierPolicy().choose([])) if config.request_policy == "three_tier" else None,
    }


def check_api(path: Path, output: Path, max_cost_rmb: float = 0.10) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("API check output directory must be empty")
    planner = LLMPlanner(path, output, budget_limit=max_cost_rmb)
    summary = {"purpose": "api_connection_check", "time": now(), "model": planner.config.model}
    try:
        planner.check_connection()
        summary.update({"status": "passed", "structured_response_valid": True})
    except Exception as exc:
        # Error strings from the planner are deliberately independent of the key.
        summary.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    summary["usage"] = planner.summary()
    write_json(output / "api_check.json", summary)
    write_json(output / "usage_summary.json", summary["usage"])
    return summary
