"""Per-request direct routing; no global proxy, DNS, or certificate changes."""
from __future__ import annotations

from typing import Any

import httpx


def post_completion(config: Any, payload: dict, api_key: str) -> httpx.Response:
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
    return details
