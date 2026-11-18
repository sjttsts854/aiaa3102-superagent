"""Strict JSON parsing with an explicit adapter for Qwen reasoning prefixes."""
from __future__ import annotations

import json
from typing import Any

from .usage import InvalidPlanError


def parse_json_content(content: str, mode: str = "json") -> Any:
    """Parse the complete response, raising InvalidPlanError on invalid input.

    ``json`` accepts only a complete JSON value. ``qwen_think_json`` also accepts
    a reasoning prefix ending in exactly one ``</think>`` marker, with or
    without an opening ``<think>`` marker. Only the entire suffix is parsed;
    JSON-like text inside the reasoning is never used as the answer. A valid
    raw JSON value is accepted first, including literal tag text in its strings.
    The caller is responsible for recording the untouched response beforehand.
    """
    if not isinstance(content, str):
        raise InvalidPlanError("Model content must be a string containing JSON.")
    if mode not in ("json", "qwen_think_json"):
        raise InvalidPlanError("Unsupported model content parser mode.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        if mode == "json":
            raise InvalidPlanError("Model content is not a complete JSON value.") from exc
        if content.count("</think>") != 1:
            raise InvalidPlanError("Qwen content must be JSON or have exactly one closing reasoning marker.") from exc
    suffix = content.split("</think>", 1)[1]
    try:
        return json.loads(suffix)
    except json.JSONDecodeError as exc:
        raise InvalidPlanError("Qwen answer after the reasoning marker is not a complete JSON value.") from exc
