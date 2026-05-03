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


def _ctx(text: str, *, task_id: str = "task-A"):
    """Stand up a context that extract_message_text(context) will read.

    extract_message_text inspects part.text first then part.root.text;
    MagicMock auto-attributes resolve part.root.text to a MagicMock (not
    a string), tripping a TypeError in the join. Use a dict-shaped part
    instead — simpler and matches how the a2a-sdk wire format actually
    serializes."""
    msg = MagicMock()
    msg.parts = [{"text": text, "kind": "text"}]
    msg.task_id = task_id
    ctx = MagicMock()
    ctx.task_id = task_id
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


# ---- executor: happy path ------------------------------------------


def _make_subprocess_factory(*, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
    """Build a fake create_subprocess_exec that returns a process whose
    communicate() returns the given stdout/stderr/returncode."""

    captured_calls: List[dict] = []

    async def fake_exec(*args, **kwargs):
        captured_calls.append({"args": args, "kwargs": kwargs})
        proc = MagicMock()
        proc.returncode = returncode

        async def fake_communicate():
            return stdout, stderr

        proc.communicate = fake_communicate
        return proc

    return fake_exec, captured_calls


@pytest.mark.asyncio
async def test_executor_happy_path_extracts_payload_text(monkeypatch):
    # Top-level "payloads" — the real shape `openclaw agent --json`
    # emits. A previous version of this test (and the adapter) wrapped
    # payloads under "result", which never matched the CLI and caused
    # the canvas to render the raw envelope dict.
    payload = {
        "payloads": [{"text": "openclaw answered: 42"}],
        "meta": {"finalAssistantVisibleText": "openclaw answered: 42"},
    }
    fake_exec, calls = _make_subprocess_factory(stdout=json.dumps(payload).encode())
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", fake_exec
    )

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("ping"), queue)

    assert len(queue.events) == 1
    # Exact equality, not substring-in-repr: a substring check would still
    # pass if the extractor silently fell back to `output` (the raw JSON
    # string), which contains the literal text inside the `payloads`
    # array. Comparing exact event text catches that drift.
    assert _event_text(queue.events[0]) == "openclaw answered: 42"
    # Subprocess was invoked with the expected fixed flags. --local
    # bypasses the openclaw gateway (which requires interactive device
    # pairing + scope-upgrade approval) and runs the embedded agent
    # against the auth-profiles.json setup() prepped.
    args = calls[0]["args"]
    assert args[0:3] == ("openclaw", "agent", "--local")
    assert "--session-id" in args
    assert "--message" in args
    assert "ping" in args
    assert "--json" in args


@pytest.mark.asyncio
async def test_executor_empty_message_short_circuits(monkeypatch):
    """If the inbound has no usable text, executor must NOT spawn a
    subprocess; it should emit 'No message provided' immediately."""
    fake_exec, calls = _make_subprocess_factory(stdout=b"")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx(""), queue)

    assert len(queue.events) == 1
    assert _event_text(queue.events[0]) == "No message provided"
    assert calls == []  # subprocess NOT spawned


# ---- executor: result-shape edge cases ------------------------------


@pytest.mark.asyncio
async def test_executor_concatenates_multiple_payloads(monkeypatch):
    """openclaw turns that include an interim narration ("Let me check!")
    plus the actual answer come back as multiple payload entries. The
    extractor must surface both, joined by a blank line, so the user
    doesn't see a truncated reply."""
    payload = {
        "payloads": [
            {"text": "Let me check!"},
            {"text": "Yep — codex and hermes are online."},
        ]
    }
    fake_exec, _ = _make_subprocess_factory(stdout=json.dumps(payload).encode())
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("hi"), queue)

    # Exact equality on the joined output — a substring-in-repr would
    # still pass if extraction failed and the raw JSON dropped both
    # texts as substrings inside the dumped envelope.
    assert _event_text(queue.events[0]) == \
        "Let me check!\n\nYep — codex and hermes are online."


@pytest.mark.asyncio
async def test_executor_falls_back_to_meta_visible_text_when_no_payloads(monkeypatch):
    """If `payloads` is missing/empty but `meta.finalAssistantVisibleText`
    has the rendered reply, surface that instead of the raw envelope."""
    payload = {"payloads": [], "meta": {"finalAssistantVisibleText": "rendered reply"}}
    fake_exec, _ = _make_subprocess_factory(stdout=json.dumps(payload).encode())
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("hi"), queue)

    text = _event_text(queue.events[0])
    assert text == "rendered reply"
    # Regression: meta envelope key name should NOT leak into the reply.
    assert "finalAssistantVisibleText" not in text


@pytest.mark.asyncio
async def test_executor_does_not_leak_envelope_dict(monkeypatch):
    """REGRESSION (2026-05-03): when payload extraction failed, the
    adapter used to fall back to ``str(data)``, dumping the entire
    `{'payloads':[...], 'meta':{...}}` envelope — including the
    bootstrap prompt, sessionId, agent harness metadata, full system
    prompt report, and tool schemas — into the canvas chat as the
    assistant's reply. The reply must NEVER include those fields."""
    payload = {
        "payloads": [{"text": "the only thing the user should see"}],
        "meta": {
            "sessionId": "secret-session-id",
            "systemPromptReport": {"chars": 26027, "tools": ["leaky"]},
            "agentMeta": {"provider": "custom-api-minimax-io"},
        },
    }
    fake_exec, _ = _make_subprocess_factory(stdout=json.dumps(payload).encode())
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("hi"), queue)

    text = _event_text(queue.events[0])
    # Exact equality: the reply must be ONLY the payload text — no
    # envelope spillover, no field-name labels, no surrounding dict
    # punctuation.
    assert text == "the only thing the user should see"
    for leaked_field in ("sessionId", "systemPromptReport", "agentMeta",
                         "secret-session-id", "custom-api-minimax-io"):
        assert leaked_field not in text, \
            f"envelope field {leaked_field!r} leaked into the assistant reply"


@pytest.mark.asyncio
async def test_executor_falls_back_to_raw_output_on_invalid_json(monkeypatch):
    """If openclaw prints a non-JSON line (early-exit warning, etc.),
    surface the raw text rather than crash."""
    fake_exec, _ = _make_subprocess_factory(stdout=b"not json at all")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("hi"), queue)

    assert _event_text(queue.events[0]) == "not json at all"


# ---- executor: error paths ------------------------------------------


@pytest.mark.asyncio
async def test_executor_handles_nonzero_exit_with_stderr(monkeypatch):
    fake_exec, _ = _make_subprocess_factory(
        stdout=b"", stderr=b"openclaw boom", returncode=1
    )
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("trigger error"), queue)

    text = repr(queue.events[0])
    assert "OpenClaw error" in text
    assert "boom" in text


@pytest.mark.asyncio
async def test_executor_handles_nonzero_exit_no_stderr(monkeypatch):
    """Same path as above but without stderr — confirm fallback message
    mentions the return code."""
    fake_exec, _ = _make_subprocess_factory(
        stdout=b"", stderr=b"", returncode=2
    )
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("trigger error"), queue)

    text = repr(queue.events[0])
    # When there's no stderr, code falls through the truthy stderr check
    # to print returncode in the alternate branch.
    assert "OpenClaw" in text
    # Either the "code 2" or "OpenClaw error: " path; both are acceptable.
    assert ("code 2" in text) or ("OpenClaw error" in text)


@pytest.mark.asyncio
async def test_executor_handles_subprocess_timeout(monkeypatch):
    """asyncio.wait_for raises TimeoutError when the subprocess exceeds
    130s — executor must emit a timeout message rather than propagate."""

    async def slow_exec(*args, **kwargs):
        proc = MagicMock()
        proc.returncode = 0

        async def slow_communicate():
            await asyncio.sleep(10)  # would exceed our patched timeout
            return b"", b""

        proc.communicate = slow_communicate
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", slow_exec)
    real_wait_for = asyncio.wait_for

    async def fast_wait_for(coro, timeout):
        # Hand back a fast timeout regardless of caller's value.
        return await real_wait_for(coro, timeout=0.05)

    monkeypatch.setattr("asyncio.wait_for", fast_wait_for)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("slow query"), queue)

    text = repr(queue.events[0])
    assert "timed out" in text or "TimeoutError" in text or "OpenClaw" in text


@pytest.mark.asyncio
async def test_executor_handles_generic_exception(monkeypatch):
    async def exploding_exec(*args, **kwargs):
        raise RuntimeError("subprocess.create exploded")

    monkeypatch.setattr("asyncio.create_subprocess_exec", exploding_exec)

    executor = OpenClawA2AExecutor(heartbeat=None)
    queue = _CapturingQueue()
    await executor.execute(_ctx("hi"), queue)

    text = repr(queue.events[0])
    assert "OpenClaw error" in text
    assert "exploded" in text


# ---- executor: heartbeat side-effects -------------------------------


@pytest.mark.asyncio
async def test_executor_clears_current_task_on_completion(monkeypatch):
    """set_current_task should be called twice: once with the brief
    summary at the start, once with the empty string in the finally."""

    payload = {"payloads": [{"text": "ok"}]}
    fake_exec, _ = _make_subprocess_factory(stdout=json.dumps(payload).encode())
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    set_calls: List[str] = []

    async def fake_set_current_task(heartbeat, value):
        set_calls.append(value)

    monkeypatch.setattr("adapter.set_current_task", fake_set_current_task)

    fake_heartbeat = MagicMock()
    executor = OpenClawA2AExecutor(heartbeat=fake_heartbeat)
    queue = _CapturingQueue()
    await executor.execute(_ctx("ping"), queue)

    # First call sets the brief, second call clears.
    assert len(set_calls) == 2
    assert set_calls[0]  # non-empty
    assert set_calls[1] == ""


@pytest.mark.asyncio
async def test_executor_clears_current_task_on_exception(monkeypatch):
    """Even if execute() crashes inside the subprocess block, the
    finally-clause must reset the heartbeat task to ''."""

    async def exploding_exec(*args, **kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr("asyncio.create_subprocess_exec", exploding_exec)

    set_calls: List[str] = []

    async def fake_set_current_task(heartbeat, value):
        set_calls.append(value)

    monkeypatch.setattr("adapter.set_current_task", fake_set_current_task)

    executor = OpenClawA2AExecutor(heartbeat=MagicMock())
    queue = _CapturingQueue()
    await executor.execute(_ctx("crash"), queue)

    assert "" in set_calls
    assert len(set_calls) == 2  # set + clear


# ---- executor: cancel -----------------------------------------------


@pytest.mark.asyncio
async def test_cancel_is_noop():
    """OpenClaw doesn't expose an interrupt API today; cancel() is a
    no-op and must not raise."""
    executor = OpenClawA2AExecutor(heartbeat=None)
    result = await executor.cancel(MagicMock(), _CapturingQueue())
    assert result is None
