"""Regression test for the RFC #600 session-id derivation fix.

Before the fix, OpenClawA2AExecutor.execute() always passed
``context.task_id or "default"`` as the openclaw CLI ``--session-id``.

In a2a-sdk v1, ``task_id`` changes per task (each inbound canvas turn is
typically a fresh task), while ``context_id`` is the stable cross-turn
conversation key. Using ``task_id`` means openclaw's native
``SessionManager`` (pi-embedded-runner/run/attempt.ts:1655) opens a
NEW session file every turn — defeating the whole point of having a
persistent session store.

Per RFC #600
(https://git.moleculesai.app/molecule-ai/internal/issues/600): the
platform stops shipping ``messages_history``; the agent owns its own
chat persistence. For that to work, the inbound dispatch MUST key the
agent's session store on a stable identifier — context_id.

These tests are import-level only (no openclaw CLI subprocess, no
network) so they run cheap in CI on any runner.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

if "adapter" not in sys.modules:
    pytest.skip(
        "adapter.py not importable (molecule_runtime missing) — "
        "matches canonical validator soft-skip",
        allow_module_level=True,
    )
adapter = sys.modules["adapter"]


def _build_context(*, task_id: str | None, context_id: str | None):
    """Build a minimal RequestContext-shaped mock with the two id
    attributes the execute() session-id derivation reads. message text
    is irrelevant to this test — we patch the CLI invocation."""
    ctx = MagicMock()
    ctx.task_id = task_id
    ctx.context_id = context_id
    msg = MagicMock()
    text_part = MagicMock()
    text_part.text = "hi"
    text_part.kind = "text"
    msg.parts = [text_part]
    msg.metadata = None
    ctx.message = msg
    return ctx


@pytest.mark.asyncio
async def test_session_id_prefers_context_id_over_task_id(monkeypatch):
    """RFC #600: context_id is the stable cross-turn key — execute() must
    pass it to the openclaw CLI so the native SessionManager resumes the
    same session file across turns."""
    captured = []

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (
                b'{"result": {"payloads": [{"text": "ok"}]}}',
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return _FakeProc()

    monkeypatch.setattr(
        adapter.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        adapter, "set_current_task", AsyncMock(return_value=None)
    )

    ex = adapter.OpenClawA2AExecutor(heartbeat=None)
    ctx = _build_context(task_id="task-changes-per-turn", context_id="chat-stable")
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()

    await ex.execute(ctx, queue)

    # The CLI args were captured — find --session-id and assert it's
    # context_id, not task_id.
    assert len(captured) == 1
    args = captured[0]
    sid_idx = args.index("--session-id")
    assert args[sid_idx + 1] == "chat-stable"
    assert args[sid_idx + 1] != "task-changes-per-turn"


@pytest.mark.asyncio
async def test_session_id_falls_back_to_task_id_when_context_id_unset(monkeypatch):
    """Backwards-compat: legacy clients that don't set context_id still
    get a deterministic session id (task_id) rather than crashing or
    silently sharing 'default' across unrelated chats."""
    captured = []

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (
                b'{"result": {"payloads": [{"text": "ok"}]}}',
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return _FakeProc()

    monkeypatch.setattr(
        adapter.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        adapter, "set_current_task", AsyncMock(return_value=None)
    )

    ex = adapter.OpenClawA2AExecutor(heartbeat=None)
    ctx = _build_context(task_id="task-only", context_id=None)
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()

    await ex.execute(ctx, queue)

    args = captured[0]
    sid_idx = args.index("--session-id")
    assert args[sid_idx + 1] == "task-only"


@pytest.mark.asyncio
async def test_session_id_falls_back_to_default_when_both_unset(monkeypatch):
    """Final fallback when neither context_id nor task_id is provided —
    matches legacy behavior so totally-anonymous inbound messages still
    invoke openclaw successfully (single shared session, not a crash)."""
    captured = []

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (
                b'{"result": {"payloads": [{"text": "ok"}]}}',
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return _FakeProc()

    monkeypatch.setattr(
        adapter.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        adapter, "set_current_task", AsyncMock(return_value=None)
    )

    ex = adapter.OpenClawA2AExecutor(heartbeat=None)
    ctx = _build_context(task_id=None, context_id=None)
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()

    await ex.execute(ctx, queue)

    args = captured[0]
    sid_idx = args.index("--session-id")
    assert args[sid_idx + 1] == "default"
