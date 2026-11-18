import json
from pathlib import Path

import httpx
import pytest

from superagent import credentials
from superagent.api_check import check_api, check_config
from superagent.credentials import read_api_key
from superagent.planner import LLMPlanner
from superagent.usage import LLMConfig


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@pytest.fixture(autouse=True)
def no_real_requests(monkeypatch):
    # School configs now bypass the system proxy via an explicit Client.
    # Fail loudly if any test accidentally reaches a real connection.
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("Real network requests are forbidden in unit tests"))


def mock_school_transport(monkeypatch, fake):
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: fake(str(url), **kwargs))


def test_local_key_is_read_without_shell_expansion(tmp_path, monkeypatch):
    monkeypatch.delenv("AIGC_API_KEY", raising=False)
    path = tmp_path / ".env.local"
    path.write_text('# Local only\nAIGC_API_KEY="dummy-$VALUE-$(literal)"\n', encoding="utf-8-sig")
    assert read_api_key("AIGC_API_KEY", path) == "dummy-$VALUE-$(literal)"
    monkeypatch.setenv("AIGC_API_KEY", "environment-key")
    assert read_api_key("AIGC_API_KEY", path) == "environment-key"


@pytest.mark.parametrize("value", ["AIGC_API_KEY=dummy\nAIGC_API_KEY=second", 'AIGC_API_KEY="unclosed', "AIGC_API_KEY=dummy with spaces"])
def test_invalid_local_credentials_never_echo_values(tmp_path, monkeypatch, value):
    monkeypatch.delenv("AIGC_API_KEY", raising=False)
    path = tmp_path / ".env.local"
    path.write_text(value, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        read_api_key("AIGC_API_KEY", path)
    assert "dummy" not in str(error.value)
    assert "unclosed'" not in str(error.value)


def test_config_check_needs_no_key_or_network(tmp_path, monkeypatch):
    monkeypatch.delenv("AIGC_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "LOCAL_ENV_PATH", tmp_path / ".env.local")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("Configuration check must not contact API"))
    result = check_config(CONFIG_DIR / "hkust-deepseek-flash.json")
    assert result["configuration_valid"]
    assert result["credentials_present"] is False
    assert result["network_request_sent"] is False
    assert result["endpoint"] == "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"


@pytest.mark.parametrize("filename, returned_model, expected_cost", [
    ("hkust-deepseek-flash.json", "DeepSeek-V4.1-Flash-1017", 0.0007),
    ("hkust-deepseek-pro.json", "DeepSeek-V4-Pro", 0.002925),
    ("hkust-qwen.json", "Qwen3.8-27B-w8a8", 0.000134),
])
def test_school_probe_endpoint_format_backend_and_cost(tmp_path, monkeypatch, filename, returned_model, expected_cost):
    monkeypatch.setenv("AIGC_API_KEY", "test-key-never-write-to-log")
    config = LLMConfig.load(CONFIG_DIR / filename)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        assert url == "https://10.121.10.250/api/v2/chat/completions"
        assert kwargs["headers"]["Host"] == "aigc-api.hkust-gz.edu.cn"
        assert kwargs["extensions"]["sni_hostname"] == "aigc-api.hkust-gz.edu.cn"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key-never-write-to-log"
        assert kwargs["json"]["model"] == config.model
        assert "response_format" not in kwargs["json"]
        assert kwargs["json"]["stream"] is False
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "id": "mock-school-response", "model": returned_model,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
        })

    mock_school_transport(monkeypatch, fake_post)
    result = check_api(CONFIG_DIR / filename, tmp_path / "probe")
    assert result["status"] == "passed"
    assert len(calls) == 1
    assert result["usage"]["accounted_cost_rmb"] == pytest.approx(expected_cost)
    assert result["usage"]["budget_rmb"] == 0.10
    assert result["usage"]["pricing_basis"] == "provider_current"
    assert result["usage"]["course_pricing_compliant"] is False
    for path in (tmp_path / "probe").glob("*.json*"):
        assert "test-key-never-write-to-log" not in path.read_text(encoding="utf-8")


def test_live_probe_missing_key_sends_no_request(tmp_path, monkeypatch):
    monkeypatch.delenv("AIGC_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "LOCAL_ENV_PATH", tmp_path / "absent")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("Missing key must not make a request"))
    with pytest.raises(ValueError, match="AIGC_API_KEY"):
        check_api(CONFIG_DIR / "hkust-qwen.json", tmp_path / "probe")
    assert not (tmp_path / "probe").exists()


def test_uncertain_probe_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGC_API_KEY", "test-key")
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("simulated")

    mock_school_transport(monkeypatch, timeout)
    result = check_api(CONFIG_DIR / "hkust-qwen.json", tmp_path / "probe")
    assert result["status"] == "failed"
    assert result["usage"]["unknown_usage"] is True
    assert len(calls) == 1


def test_school_planning_still_requires_valid_json(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGC_API_KEY", "test-key")

    def fake_post(url, **kwargs):
        assert "response_format" not in kwargs["json"]
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        assert payload["training_profile"]["data_type"] == "tabular"
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "model": "DeepSeek-V4.1-Flash-1017",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"action":"stop","reason":"Enough validation evidence"}'}}],
        })

    mock_school_transport(monkeypatch, fake_post)
    planner = LLMPlanner(CONFIG_DIR / "hkust-deepseek-flash.json", tmp_path / "run")
    assert planner.propose({"data_type": "tabular"}, [], {})["action"] == "stop"
