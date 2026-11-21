"""Per-request direct routing; no global proxy, DNS, or certificate changes."""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import httpx


class RequestDeadlineExceeded(httpx.TimeoutException):
    """The local request process was killed and reaped at its total deadline."""


class RequestWorkerError(httpx.TransportError):
    """The transport worker failed without returning an auditable response."""


def _request_environment(config: Any, api_key: str) -> dict[str, str]:
    """The only credential passed to this process travels through stdin."""
    explicit_name = str(getattr(config, "api_key_env", "")).upper()
    return {
        name: value for name, value in os.environ.items()
        if name.upper() != explicit_name
        and not any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "COOKIE"))
        and value != api_key
    }


def _safe_worker_diagnostic(value: Any) -> dict[str, Any]:
    """Accept only class names, errno values and HTTP status from the worker."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    kind = value.get("exception_type")
    if isinstance(kind, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", kind):
        result["exception_type"] = kind
    status = value.get("http_status")
    if type(status) is int and 100 <= status <= 599:
        result["http_status"] = status
    chain = []
    for item in value.get("cause_chain", [])[:6] if isinstance(value.get("cause_chain"), list) else []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", kind):
            continue
        clean = {"type": kind}
        for name in ("errno", "winerror"):
            number = item.get(name)
            if type(number) is int:
                clean[name] = number
        chain.append(clean)
    if chain:
        result["cause_chain"] = chain
    return result


def post_completion(config: Any, payload: dict, api_key: str) -> httpx.Response:
    """Use the legacy transport, or a killable process for an opt-in deadline.

    Killing the local client does not prove the provider cancelled generation;
    callers must retain unknown token usage when this deadline is exceeded.
    """
    deadline = getattr(config, "request_deadline_seconds", None)
    if deadline is None:
        return _post_completion_direct(config, payload, api_key)
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("request_deadline_seconds must be finite and positive")
    started = time.monotonic()
    envelope = {
        "config": {name: getattr(config, name) for name in ("base_url", "timeout_seconds", "bypass_system_proxy", "connect_ip")},
        "payload": payload,
        "api_key": api_key,
    }
    # No credential is included in an argument, environment variable or file.
    stdin = json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")
    environment = _request_environment(config, api_key)
    remaining = deadline - (time.monotonic() - started)
    if remaining <= 0:
        raise RequestDeadlineExceeded("Total request deadline exceeded before transport startup")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "superagent.request_worker"],
            input=stdin,
            capture_output=True,
            timeout=remaining,
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for its child before raising. Never
        # expose TimeoutExpired output/arguments or a background request thread.
        raise RequestDeadlineExceeded("Total request deadline exceeded; local request process terminated") from None
    except OSError:
        raise RequestWorkerError("Request transport process could not start") from None
    if time.monotonic() - started > deadline:
        raise RequestDeadlineExceeded("Total request deadline exceeded before the response was delivered")
    try:
        reply = json.loads(completed.stdout)
        if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
            raise ValueError()
    except (ValueError, UnicodeDecodeError):
        raise RequestWorkerError("Request transport process returned an invalid response") from None
    if not reply["ok"]:
        diagnostic = _safe_worker_diagnostic(reply.get("diagnostic"))
        kind = diagnostic.get("exception_type")
        allowed = {name: getattr(httpx, name) for name in (
            "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
            "ConnectError", "ReadError", "WriteError", "CloseError",
            "RemoteProtocolError", "LocalProtocolError", "UnsupportedProtocol",
            "ProxyError", "DecodingError",
        )}
        error = allowed.get(kind, RequestWorkerError)("Request transport process failed; see safe diagnostic")
        error.worker_diagnostic = diagnostic
        raise error
    if completed.returncode != 0:
        raise RequestWorkerError("Request transport process exited unsuccessfully")
    try:
        status = reply["status_code"]
        if type(status) is not int or not 100 <= status <= 599:
            raise ValueError()
        body = base64.b64decode(reply["body_base64"], validate=True)
        response_headers = {}
        content_type = reply.get("content_type")
        if isinstance(content_type, str) and "\r" not in content_type and "\n" not in content_type:
            response_headers["content-type"] = content_type
        endpoint = config.base_url.rstrip("/") + "/chat/completions"
        # Retain a credential-free request for HTTPStatusError diagnostics.
        return httpx.Response(status, content=body, headers=response_headers,
                              request=httpx.Request("POST", endpoint))
    except (KeyError, TypeError, ValueError):
        raise RequestWorkerError("Request transport process returned an invalid response") from None


def _post_completion_direct(config: Any, payload: dict, api_key: str) -> httpx.Response:
    """Send to one configured endpoint, optionally using a campus-resolved IP.

    An IP override changes only the connection destination. The original DNS
    hostname remains in HTTP Host and TLS SNI, so certificate verification still
    checks the intended school server. Redirects are disabled to keep the bearer
    credential at the configured destination. This does not bypass a VPN TUN
    route; it only avoids HTTPX's environment/system proxy discovery.
    """
    endpoint = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + api_key}
    if not config.bypass_system_proxy and config.connect_ip is None:
        # Preserve the existing path for other providers and older configs.
        return httpx.post(endpoint, headers=headers, json=payload, timeout=config.timeout_seconds)

    original = httpx.URL(endpoint)
    target = original.copy_with(host=config.connect_ip) if config.connect_ip is not None else original
    headers["Host"] = original.netloc.decode("ascii")
    extensions = {"sni_hostname": original.host} if original.scheme == "https" else {}
    with httpx.Client(trust_env=False, verify=True, follow_redirects=False) as client:
        return client.post(
            target,
            headers=headers,
            json=payload,
            timeout=config.timeout_seconds,
            extensions=extensions,
        )


def safe_error_details(exc: BaseException) -> dict[str, Any]:
    """Expose useful error metadata without exception text or request contents."""
    details: dict[str, Any] = {"exception_type": type(exc).__name__}
    chain = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        item: dict[str, Any] = {"type": type(current).__name__}
        for name in ("errno", "winerror"):
            value = getattr(current, name, None)
            if type(value) is int:
                item[name] = value
        if isinstance(current, httpx.HTTPStatusError):
            status = current.response.status_code
            if type(status) is int:
                details["http_status"] = status
        chain.append(item)
        current = current.__cause__
    details["cause_chain"] = chain
    worker = _safe_worker_diagnostic(getattr(exc, "worker_diagnostic", None))
    if worker:
        details["worker_diagnostic"] = worker
    return details
