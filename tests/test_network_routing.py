import json
from types import SimpleNamespace

import httpx
import pytest

from superagent.network import post_completion, safe_error_details


def config(**changes):
    values = {
        "base_url": "https://aigc-api.hkust-gz.edu.cn/api/v2",
        "timeout_seconds": 60,
        "bypass_system_proxy": False,
        "connect_ip": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_default_route_preserves_original_request_path(monkeypatch):
    observed = []
    expected = httpx.Response(200)

    def fake_post(url, **kwargs):
        observed.append((url, kwargs))
        return expected

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("Default route must preserve the original helper"))
    assert post_completion(config(), {"model": "test"}, "test-key") is expected
    assert observed == [("https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions", {
        "headers": {"Authorization": "Bearer test-key"}, "json": {"model": "test"}, "timeout": 60,
    })]


@pytest.mark.parametrize("connect_ip", [None, "10.121.10.250"])
def test_direct_route_preserves_hostname_tls_and_proxy_isolation(monkeypatch, connect_ip):
    observed = []
    client_options = []
    original_client = httpx.Client

    def handle(request):
        observed.append(request)
        return httpx.Response(200, json={"ok": True})

    def client_factory(**kwargs):
        client_options.append(kwargs)
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("Direct route must not use the system-aware helper"))
    response = post_completion(config(bypass_system_proxy=True, connect_ip=connect_ip), {"model": "test"}, "test-key")
    assert response.json() == {"ok": True}
    assert client_options == [{"trust_env": False, "verify": True, "follow_redirects": False}]
    assert len(observed) == 1
    request = observed[0]
    assert request.url.host == (connect_ip or "aigc-api.hkust-gz.edu.cn")
    assert request.url.scheme == "https"
    assert request.url.path == "/api/v2/chat/completions"
    assert request.headers["Host"] == "aigc-api.hkust-gz.edu.cn"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.extensions["sni_hostname"] == "aigc-api.hkust-gz.edu.cn"
    assert json.loads(request.content) == {"model": "test"}


def test_direct_route_does_not_follow_cross_host_redirect(monkeypatch):
    calls = []
    original_client = httpx.Client

    def handle(request):
        calls.append(request)
        return httpx.Response(307, headers={"location": "https://other.example/collect"})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs))
    response = post_completion(config(bypass_system_proxy=True, connect_ip="10.121.10.250"), {}, "test-key")
    assert response.status_code == 307
    assert len(calls) == 1
    assert calls[0].url.host == "10.121.10.250"


def test_nonstandard_port_survives_ip_override(monkeypatch):
    calls = []
    original_client = httpx.Client

    def handle(request):
        calls.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs))
    post_completion(config(base_url="https://server.example:8443/custom", bypass_system_proxy=True, connect_ip="10.1.2.3"), {}, "test-key")
    assert calls[0].url.port == 8443
    assert calls[0].headers["Host"] == "server.example:8443"
    assert calls[0].extensions["sni_hostname"] == "server.example"
    assert calls[0].url.path == "/custom/chat/completions"


def test_error_diagnostics_preserve_only_safe_metadata():
    root = OSError(11001, "sensitive-content-must-not-appear")
    error = httpx.ConnectError("Bearer test-secret request body")
    error.__cause__ = root
    details = safe_error_details(error)
    assert details == {"exception_type": "ConnectError", "cause_chain": [
        {"type": "ConnectError"}, {"type": "OSError", "errno": 11001},
    ]}
    serialized = json.dumps(details)
    assert "sensitive" not in serialized
    assert "secret" not in serialized
    assert "Bearer" not in serialized


def test_http_diagnostic_exposes_status_not_headers_or_url():
    request = httpx.Request("POST", "https://example.com/private?token=secret", headers={"Authorization": "Bearer secret"})
    response = httpx.Response(401, request=request)
    error = httpx.HTTPStatusError("Secret exception text", request=request, response=response)
    details = safe_error_details(error)
    assert details["http_status"] == 401
    assert details["cause_chain"] == [{"type": "HTTPStatusError"}]
    assert "secret" not in json.dumps(details).lower()


def test_diagnostic_cause_cycle_is_bounded():
    first = ValueError("private-one")
    second = RuntimeError("private-two")
    first.__cause__ = second
    second.__cause__ = first
    assert len(safe_error_details(first)["cause_chain"]) == 2
