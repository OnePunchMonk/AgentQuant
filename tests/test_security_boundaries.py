"""Regression tests for agent trust boundaries."""
import sys
from types import SimpleNamespace

import pytest

from src.agent.base_planner import GeminiPlanner
from src.agent.tools.orchestrator import ToolOrchestrator
from src.agent.tools.registry import Tool, ToolRegistry
from src.strategies.codegen import register_strategy_source


def test_generated_source_never_executes():
    source = 'raise RuntimeError("executed")\nclass Example(Strategy):\n def generate_signal(self, df, params): return 0\n'
    with pytest.raises(RuntimeError, match="execution is disabled"):
        register_strategy_source(source, "example", {})


def test_unparseable_model_output_not_logged(caplog):
    GeminiPlanner._parse_json_response(None, "private synthetic secret", 1)
    assert "private synthetic secret" not in caplog.text


def test_tool_errors_do_not_return_exception_content(caplog):
    registry = ToolRegistry()
    def fail():
        raise ValueError("private synthetic secret")
    registry.register(Tool(name="fail", description="", input_schema={}, callable_fn=fail, category="test"))
    result = registry.execute_tool("fail", {})
    assert result["error"] == "tool execution failed"
    assert "private synthetic secret" not in caplog.text


def test_tool_call_limit_applies_with_multiple_calls_per_turn(monkeypatch):
    calls = []
    registry = ToolRegistry()
    registry.register(Tool(name="sample", description="", input_schema={},
                           callable_fn=lambda: calls.append(1), category="test"))
    blocks = [SimpleNamespace(type="tool_use", name="sample", input={}, id=str(i)) for i in range(5)]
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(stop_reason="tool_use", content=blocks)))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: client))
    context = SimpleNamespace(regime_label="test", regime_confidence=1.0)
    result = ToolOrchestrator(registry).run_tool_loop("test", context, "momentum", max_turns=1, max_tool_calls=2)
    assert len(calls) == 2
    assert sum(not item["result"]["success"] for item in result["tool_calls"]) == 3


def test_oversized_tool_input_is_not_executed(monkeypatch):
    calls = []
    registry = ToolRegistry()
    registry.register(Tool(name="sample", description="", input_schema={},
                           callable_fn=lambda **kw: calls.append(kw), category="test"))
    block = SimpleNamespace(type="tool_use", name="sample", input={"x": "a" * 17000}, id="1")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(stop_reason="tool_use", content=[block])))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: client))
    context = SimpleNamespace(regime_label="test", regime_confidence=1.0)
    result = ToolOrchestrator(registry).run_tool_loop("test", context, "momentum", max_turns=1)
    assert calls == []
    assert result["tool_calls"][0]["result"]["error"] == "tool input budget exceeded"


def test_dns_rebinding_pins_destination_and_tls_hostname(monkeypatch):
    from src.agent.tools.content_extraction import _pinned_get, _resolve_public_http_url
    import requests
    seen = {}
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    ip, host, err = _resolve_public_http_url("https://example.com/path?q=1")
    assert err is None
    def fake_get(session, url, **kwargs):
        seen.update(url=url, headers=kwargs["headers"], trust_env=session.trust_env,
                    pool=session.get_adapter(url).poolmanager.connection_pool_kw)
        return object()
    monkeypatch.setattr(requests.Session, "get", fake_get)
    _pinned_get("https://example.com/path?q=1", ip, host)
    assert seen["url"] == "https://93.184.216.34/path?q=1"
    assert seen["headers"]["Host"] == "example.com"
    assert seen["pool"]["assert_hostname"] == "example.com"
    assert seen["pool"]["server_hostname"] == "example.com"
    assert seen["trust_env"] is False


def test_dns_mixed_public_and_private_rejected(monkeypatch):
    from src.agent.tools.content_extraction import _resolve_public_http_url
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [
        (2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("127.0.0.1", 443))])
    assert _resolve_public_http_url("https://example.com/")[2] == "URL host is not public"
