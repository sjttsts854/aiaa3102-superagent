import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from superagent import network


@pytest.fixture
def endpoint():
    state = {"requests": [], "received": threading.Event(), "closed": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["requests"].append({"path": self.path, "payload": json.loads(body),
                                      "authorization": self.headers.get("Authorization")})
            state["received"].set()
            if self.path.startswith("/stall/"):
                self.connection.settimeout(5)
                try:
                    if self.connection.recv(1) == b"":
                        state["closed"].set()
                except (ConnectionResetError, ConnectionAbortedError):
                    state["closed"].set()
                return
            status = 429 if self.path.startswith("/rate/") else 200
            data = json.dumps({"ok": status == 200, "text": "complete reply"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % server.server_port, state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def config(endpoint, **changes):
    values = {"base_url": endpoint + "/api", "timeout_seconds": 20,
              "bypass_system_proxy": True, "connect_ip": None,
              "request_deadline_seconds": 5, "api_key_env": "UNUSUAL_CREDENTIAL_NAME"}
    values.update(changes)
    return SimpleNamespace(**values)


def test_deadline_transport_returns_complete_body_without_auth_in_response(endpoint):
    url, state = endpoint
    response = network.post_completion(config(url), {"max_tokens": 8192}, "fixture-key")
    assert response.json() == {"ok": True, "text": "complete reply"}
    assert response.headers["content-type"] == "application/json"
    assert response.request.headers.get("Authorization") is None
    assert state["requests"] == [{"path": "/api/chat/completions", "payload": {"max_tokens": 8192},
                                  "authorization": "Bearer fixture-key"}]


def test_http_status_is_preserved_for_existing_accounting_handler(endpoint):
    url, _ = endpoint
    response = network.post_completion(config(url, base_url=url + "/rate"), {}, "fixture-key")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    assert network.safe_error_details(caught.value)["http_status"] == 429
    assert "fixture-key" not in json.dumps(network.safe_error_details(caught.value))


def test_wall_deadline_kills_and_reaps_request_process(endpoint, monkeypatch):
    url, state = endpoint
    processes = []
    original = subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(network.subprocess, "Popen", recording_popen)
    started = time.monotonic()
    with pytest.raises(network.RequestDeadlineExceeded) as caught:
        network.post_completion(config(url, base_url=url + "/stall", request_deadline_seconds=1.5), {}, "fixture-key")
    elapsed = time.monotonic() - started
    assert 1 <= elapsed < 3.5
    assert state["received"].is_set()
    assert len(processes) == 1 and processes[0].poll() is not None
    assert state["closed"].wait(1)
    assert len(state["requests"]) == 1
    details = network.safe_error_details(caught.value)
    assert details["exception_type"] == "RequestDeadlineExceeded"
    assert "fixture-key" not in str(caught.value) + json.dumps(details)


def test_credentials_travel_only_in_stdin(endpoint, monkeypatch):
    url, _ = endpoint
    original = subprocess.run
    recorded = []
    monkeypatch.setenv("AIGC_API_KEY", "fixture-key")
    monkeypatch.setenv("OTHER_TOKEN", "other-fixture-secret")
    monkeypatch.setenv("UNUSUAL_CREDENTIAL_NAME", "fixture-key")
    monkeypatch.setenv("UNRELATED_WITH_SAME_VALUE", "fixture-key")

    def record(*args, **kwargs):
        recorded.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(network.subprocess, "run", record)
    network.post_completion(config(url), {"model": "fixture-model"}, "fixture-key")
    args, kwargs = recorded[0]
    assert "fixture-key" not in repr(args)
    assert "fixture-key" not in repr(kwargs["env"])
    assert "other-fixture-secret" not in repr(kwargs["env"])
    assert json.loads(kwargs["input"])["api_key"] == "fixture-key"
    assert kwargs["capture_output"] is True
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)


def test_child_phase_timeout_survives_as_safe_httpx_error(endpoint):
    url, _ = endpoint
    with pytest.raises(httpx.ReadTimeout) as caught:
        network.post_completion(config(url, base_url=url + "/stall", timeout_seconds=.15), {}, "fixture-key")
    details = network.safe_error_details(caught.value)
    assert details["exception_type"] == "ReadTimeout"
    assert details["worker_diagnostic"]["exception_type"] == "ReadTimeout"
    assert "fixture-key" not in str(caught.value) + json.dumps(details)


def test_worker_protocol_failure_does_not_expose_stdout_or_stderr(monkeypatch):
    def broken(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, b"fixture-key", b"Authorization: fixture-key")
    monkeypatch.setattr(network.subprocess, "run", broken)
    with pytest.raises(network.RequestWorkerError) as caught:
        network.post_completion(config("http://127.0.0.1:1"), {}, "fixture-key")
    assert "fixture-key" not in str(caught.value) + json.dumps(network.safe_error_details(caught.value))


@pytest.mark.parametrize("deadline", [0, -1, True, float("inf"), float("nan"), "5"])
def test_invalid_deadline_is_rejected_before_network(deadline, monkeypatch):
    monkeypatch.setattr(network.subprocess, "run", lambda *a, **k: pytest.fail("No process should start"))
    with pytest.raises(ValueError, match="request_deadline_seconds"):
        network.post_completion(config("http://127.0.0.1:1", request_deadline_seconds=deadline), {}, "fixture-key")
