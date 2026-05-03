"""Tests for OpenClawAdapter + OpenClawA2AExecutor.

The adapter shell + executor have been historically untested — only
``test_model_routing.py`` covered the pure ``_resolve_provider_routing``
helper. This file fills the rest:

  - Static introspection (name, display_name, description, schema)
  - __init__ defaults
  - create_executor returns an OpenClawA2AExecutor
  - executor.execute() happy path with a stub `openclaw agent` subprocess
  - executor.execute() error paths: empty message, JSON parse fallback,
    subprocess returns non-zero, asyncio.TimeoutError, generic Exception
  - executor.cancel() is a no-op

Subprocess invocation is faked via monkeypatching
``asyncio.create_subprocess_exec`` so we don't need a real openclaw
binary in the test environment.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("a2a.helpers")
pytest.importorskip("molecule_runtime.adapters.base")

from adapter import (  # noqa: E402
    Adapter,
    OpenClawAdapter,
    OpenClawA2AExecutor,
    OPENCLAW_PORT,
)
from molecule_runtime.adapters.base import AdapterConfig  # noqa: E402


class _CapturingQueue:
    def __init__(self) -> None:
        self.events: List[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


def _event_text(event: Any) -> str:
    """Pull the exact text out of an a2a Message — `event.parts[0].text`.

    Tests previously matched on `repr(event)`, which is too loose: the
    raw JSON envelope contains the assistant text as a substring even
    when the extractor failed and we fell back to `output` (the JSON
    string). Comparing exact text catches that drift instead of hiding
    it.
    """
    return event.parts[0].text


def _ctx(text: str, *, task_id: str = "task-A", context_id: str = "ctx-A"):
    """Stand up a context that extract_message_text(context) will read.

    extract_message_text inspects part.text first then part.root.text;
    MagicMock auto-attributes resolve part.root.text to a MagicMock (not
    a string), tripping a TypeError in the join. Use a dict-shaped part
    instead — simpler and matches how the a2a-sdk wire format actually
    serializes.

    task_id + context_id are pinned to plain strings (not MagicMock
    auto-attributes) because new_response_message uses them as Message
    proto fields, and the proto type-check rejects MagicMock instances
    with `TypeError: bad argument type for built-in operation`."""
    msg = MagicMock()
    msg.parts = [{"text": text, "kind": "text"}]
    msg.task_id = task_id
    msg.context_id = context_id
    ctx = MagicMock()
    ctx.task_id = task_id
    ctx.context_id = context_id
    ctx.message = msg
    return ctx


# ---- structural -----------------------------------------------------


def test_adapter_alias():
    assert Adapter is OpenClawAdapter


def test_adapter_init_default_state():
    adapter = OpenClawAdapter()
    assert adapter._gateway_process is None


def test_static_introspection():
    assert OpenClawAdapter.name() == "openclaw"
    assert OpenClawAdapter.display_name() == "OpenClaw"
    desc = OpenClawAdapter.description()
    assert "OpenClaw" in desc
    assert "SOUL" in desc or "BOOTSTRAP" in desc or "AGENTS" in desc
    schema = OpenClawAdapter.get_config_schema()
    assert "model" in schema
    assert "provider_url" in schema
    assert "gateway_port" in schema
    assert schema["gateway_port"]["default"] == OPENCLAW_PORT


@pytest.mark.asyncio
async def test_create_executor_returns_a2a_executor():
    adapter = OpenClawAdapter()
    cfg = AdapterConfig(model="openai:gpt-4o", heartbeat=None)
    executor = await adapter.create_executor(cfg)
    assert isinstance(executor, OpenClawA2AExecutor)
    assert executor._heartbeat is None




# ---- executor: gateway WS path (push parity refactor) -----------------
#
# All executor tests below mock the GatewayClient directly — no
# subprocess shell-out path remains in the executor after the
# sessions.steer push-parity refactor (#25). Tests assert WS request
# shape (method names, params), session+run lifecycle (steer when
# active, send + agent.wait when not), and assistant-text extraction
# from chat.history.


class _FakeGateway:
    """Stand-in for GatewayClient.

    Records every request() call. Test-controllable response per method
    via ``responses[method]`` (callable returning a dict) or default
    success shape. ``raise_on`` lets tests inject GatewayError /
    asyncio.TimeoutError for one specific method.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.responses: dict[str, Any] = {}
        self.raise_on: dict[str, Exception] = {}
        self.closed = False

    async def request(self, method: str, params: dict | None = None, *, timeout: float | None = None) -> dict:
        self.requests.append((method, params or {}))
        if method in self.raise_on:
            raise self.raise_on[method]
        if method in self.responses:
            handler = self.responses[method]
            return handler(params or {}) if callable(handler) else handler
        # Sensible defaults for the methods the executor routes.
        if method == "sessions.send":
            return {"runId": (params or {}).get("idempotencyKey", "run-default"), "messageSeq": 1}
        if method == "sessions.steer":
            return {"runId": (params or {}).get("idempotencyKey", "run-default"), "messageSeq": 1}
        if method == "agent.wait":
            return {"status": "ok", "endedAt": 0}
        if method == "chat.history":
            return {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "in"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "default-reply"}]},
                ]
            }
        if method == "sessions.abort":
            return {"ok": True}
        return {}

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_executor_empty_message_short_circuits():
    """Whitespace-only messages return early without touching the
    gateway. Mirrors the prior subprocess behavior."""
    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("   "), queue)
    assert len(queue.events) == 1
    assert "No message provided" in _event_text(queue.events[0])


@pytest.mark.asyncio
async def test_executor_happy_path_send_then_history(monkeypatch):
    """First message: no active run → sessions.send → agent.wait →
    chat.history. The most recent assistant text from chat.history
    is returned as the A2A response."""
    fake = _FakeGateway()
    fake.responses["sessions.send"] = lambda p: {"runId": "run-1", "messageSeq": 1}
    fake.responses["chat.history"] = lambda p: {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "what is 2+2"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "openclaw says 4"}]},
        ]
    }

    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake  # skip lazy init

    queue = _CapturingQueue()
    await executor.execute(_ctx("what is 2+2"), queue)

    methods = [m for m, _ in fake.requests]
    # Exact RPC sequence — no sessions.steer because no active run.
    assert methods == ["sessions.send", "agent.wait", "chat.history"], methods
    # send was called with our session key + the user message.
    send_method, send_params = fake.requests[0]
    assert send_params["key"] == OpenClawA2AExecutor._SESSION_KEY
    assert send_params["message"] == "what is 2+2"
    assert send_params["idempotencyKey"].startswith("send-")
    # Reply is the assistant text from chat.history.
    assert _event_text(queue.events[0]) == "openclaw says 4"


@pytest.mark.asyncio
async def test_executor_steers_when_run_active():
    """Mid-flight arrival: an active run is recorded → sessions.steer
    fires (NOT sessions.send) → execute() returns a placeholder
    immediately. Push parity with codex/turn-steer + claude-code
    notifications/claude/channel."""
    fake = _FakeGateway()
    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake
    # Simulate "run already in flight" — set the active marker.
    executor._active_run_id = "run-already-running"

    queue = _CapturingQueue()
    await executor.execute(_ctx("steered prompt"), queue)

    methods = [m for m, _ in fake.requests]
    # Exact one-call shape — no fallback to send.
    assert methods == ["sessions.steer"], methods
    steer_method, steer_params = fake.requests[0]
    assert steer_params["key"] == OpenClawA2AExecutor._SESSION_KEY
    assert steer_params["message"] == "steered prompt"
    assert steer_params["idempotencyKey"].startswith("steer-")
    # Placeholder delivered as A2A response.
    assert "steered into in-flight openclaw run" in _event_text(queue.events[0])


@pytest.mark.asyncio
async def test_executor_steer_failure_falls_through_to_send():
    """If sessions.steer raises (NoActiveRun / not steerable / transport
    hiccup), executor falls through to the sessions.send path so the
    message still gets processed."""
    from gateway_client import GatewayError

    fake = _FakeGateway()
    fake.raise_on["sessions.steer"] = GatewayError("sessions.steer", "NoActiveRun", "no active run")

    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake
    executor._active_run_id = "stale-run-id"  # we believe one is active, but the gateway disagrees

    queue = _CapturingQueue()
    await executor.execute(_ctx("retry me"), queue)

    methods = [m for m, _ in fake.requests]
    # Steer attempted, then fall-through to send/wait/history.
    assert methods == ["sessions.steer", "sessions.send", "agent.wait", "chat.history"]
    # Reply is real text (not the placeholder).
    text = _event_text(queue.events[0])
    assert "steered into" not in text


@pytest.mark.asyncio
async def test_executor_clears_active_run_on_completion():
    """After sessions.send + agent.wait + chat.history complete,
    `_active_run_id` is cleared so the NEXT message takes the send
    path again (correct steady-state — steer is only meaningful while
    a run is genuinely in flight)."""
    fake = _FakeGateway()
    fake.responses["sessions.send"] = lambda p: {"runId": "run-X", "messageSeq": 1}

    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake

    queue = _CapturingQueue()
    await executor.execute(_ctx("first"), queue)

    assert executor._active_run_id is None


@pytest.mark.asyncio
async def test_executor_returns_placeholder_when_history_empty():
    """If chat.history returns no assistant message (gateway race or
    silence), the executor returns a clear placeholder rather than
    empty-string-as-reply (which the canvas renders as a phantom blank
    bubble)."""
    fake = _FakeGateway()
    fake.responses["chat.history"] = lambda p: {"messages": []}

    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake

    queue = _CapturingQueue()
    await executor.execute(_ctx("question"), queue)

    text = _event_text(queue.events[0])
    assert "no assistant text" in text


@pytest.mark.asyncio
async def test_executor_handles_send_failure_gracefully():
    """sessions.send failing returns a clean error message, not an
    exception leak."""
    from gateway_client import GatewayError

    fake = _FakeGateway()
    fake.raise_on["sessions.send"] = GatewayError("sessions.send", "BadParams", "missing key")

    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake

    queue = _CapturingQueue()
    await executor.execute(_ctx("hi"), queue)

    text = _event_text(queue.events[0])
    assert "OpenClaw sessions.send failed" in text
    assert "BadParams" in text


# ---- _assistant_text_from_history -------------------------------------


def test_assistant_text_from_history_picks_most_recent():
    """When history has multiple assistant messages (multi-turn), the
    helper returns the most recent one."""
    from adapter import _assistant_text_from_history

    hist = {
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "first reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "follow-up"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "second reply"}]},
        ]
    }
    assert _assistant_text_from_history(hist) == "second reply"


def test_assistant_text_from_history_handles_string_content():
    """Some chat.history entries carry `content` as a plain string,
    not a structured parts list. Extractor must accept both shapes."""
    from adapter import _assistant_text_from_history

    hist = {"messages": [{"role": "assistant", "content": "plain string reply"}]}
    assert _assistant_text_from_history(hist) == "plain string reply"


def test_assistant_text_from_history_skips_user_rows():
    """User-role rows must never be returned, even if they're the last
    row (race where the user's message is appended but the assistant
    response is still rendering)."""
    from adapter import _assistant_text_from_history

    hist = {
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "old"}]},
            {"role": "user", "content": [{"type": "text", "text": "new q"}]},
        ]
    }
    assert _assistant_text_from_history(hist) == "old"


def test_assistant_text_from_history_returns_empty_on_no_assistant():
    """Pin: an all-user history returns empty string. Caller decides
    how to surface (the executor returns a placeholder)."""
    from adapter import _assistant_text_from_history

    assert _assistant_text_from_history({"messages": []}) == ""
    assert _assistant_text_from_history({}) == ""
    assert _assistant_text_from_history({"messages": [{"role": "user", "content": "q"}]}) == ""


# ---- executor: cancel -----------------------------------------------


@pytest.mark.asyncio
async def test_cancel_no_gateway_is_noop():
    """cancel() before _ensure_gateway has connected is a no-op."""
    executor = OpenClawA2AExecutor(heartbeat=None)
    assert executor._gateway is None
    result = await executor.cancel(MagicMock(), _CapturingQueue())
    assert result is None


@pytest.mark.asyncio
async def test_cancel_no_active_run_is_noop():
    """Gateway connected but no run active → cancel() doesn't fire abort."""
    fake = _FakeGateway()
    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake

    await executor.cancel(MagicMock(), _CapturingQueue())
    assert not any(m == "sessions.abort" for m, _ in fake.requests)


@pytest.mark.asyncio
async def test_cancel_with_active_run_fires_abort():
    """Active run + gateway → sessions.abort with key + runId."""
    fake = _FakeGateway()
    executor = OpenClawA2AExecutor(heartbeat=None)
    executor._gateway = fake
    executor._active_run_id = "run-cancel-me"

    await executor.cancel(MagicMock(), _CapturingQueue())

    aborts = [(m, p) for m, p in fake.requests if m == "sessions.abort"]
    assert len(aborts) == 1
    _, params = aborts[0]
    assert params["key"] == OpenClawA2AExecutor._SESSION_KEY
    assert params["runId"] == "run-cancel-me"
