"""Pin the openclaw adapter's provider-routing logic.

Why this exists
---------------
openclaw is OpenAI-compat only — its ``--custom-compatibility`` CLI
flag is hard-set to ``openai``. Model strings arrive in
LangChain-style ``<provider>:<id>`` form because the wheel's
config.py default is ``anthropic:claude-opus-4-7`` (so langchain /
crewai consumers get a uniform model string out of the box).

The pre-fix routing in ``adapter.py:setup()`` fell through
``anthropic:claude-X`` to the OpenAI key + ``api.openai.com`` path
with the bare model id ``claude-X``, which OpenAI doesn't host.
Every inference call failed silently — the workspace looked online
but was structurally broken on every turn.

These tests exercise ``_resolve_provider_routing`` (the pure
extraction of the routing decision) so the eight branches it
encodes are pinned without spinning up the npm install + onboard
side effects of the real ``setup()``.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest


# ---- Stubs ----
#
# adapter.py imports a chain of platform modules at module load:
#
#   - molecule_runtime.adapters.base     (BaseAdapter, AdapterConfig)
#   - molecule_runtime.adapters.shared_runtime (helper re-exports)
#   - a2a.server.agent_execution        (AgentExecutor base class)
#
# Stub the minimum surface so the import succeeds in a CI environment
# where the runtime wheel isn't pip-installable. Routing logic itself
# touches none of these, so the stubs only need to satisfy `from X
# import Y` at module load.


def _ensure_module(dotted: str) -> types.ModuleType:
    if dotted not in sys.modules:
        sys.modules[dotted] = types.ModuleType(dotted)
    return sys.modules[dotted]


def _ensure_attr(mod: types.ModuleType, name: str, value: object) -> None:
    if not hasattr(mod, name):
        setattr(mod, name, value)


def _install_stubs() -> None:
    _ensure_module("molecule_runtime")
    _ensure_module("molecule_runtime.adapters")
    base = _ensure_module("molecule_runtime.adapters.base")
    _ensure_attr(base, "BaseAdapter", type("BaseAdapter", (), {}))
    _ensure_attr(base, "AdapterConfig", type("AdapterConfig", (), {}))
    shared = _ensure_module("molecule_runtime.adapters.shared_runtime")
    _ensure_attr(shared, "brief_task", lambda *a, **kw: "")
    _ensure_attr(shared, "extract_message_text", lambda *a, **kw: "")
    _ensure_attr(shared, "set_current_task", lambda *a, **kw: None)
    _ensure_module("a2a")
    _ensure_module("a2a.server")
    a2a_exec = _ensure_module("a2a.server.agent_execution")
    _ensure_attr(a2a_exec, "AgentExecutor", type("AgentExecutor", (), {}))


def _load_adapter():
    _install_stubs()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    sys.modules.pop("adapter", None)
    import adapter  # noqa: WPS433
    return adapter


@pytest.fixture
def resolve():
    """Bind the routing helper after stubbed import."""
    return _load_adapter()._resolve_provider_routing


# ---- Branch coverage on `_resolve_provider_routing` -------------------------


def test_bare_model_id_routes_to_openai(resolve):
    """No ``:`` prefix → falls to the legacy OpenAI default."""
    prefix, model, url, key = resolve(
        "gpt-4o-mini",
        env={"OPENAI_API_KEY": "sk-test"},
        runtime_config={},
    )
    assert prefix == "openai"
    assert model == "gpt-4o-mini"
    assert url == "https://api.openai.com/v1"
    assert key == "sk-test"


def test_openai_prefix_strips_and_routes(resolve):
    prefix, model, url, key = resolve(
        "openai:gpt-4o",
        env={"OPENAI_API_KEY": "sk-test"},
        runtime_config={},
    )
    assert prefix == "openai"
    assert model == "gpt-4o"
    assert url == "https://api.openai.com/v1"


def test_groq_prefix_uses_groq_key(resolve):
    prefix, model, url, key = resolve(
        "groq:llama-3.1-70b",
        env={"GROQ_API_KEY": "gsk-test", "OPENAI_API_KEY": "sk-also"},
        runtime_config={},
    )
    assert prefix == "groq"
    assert model == "llama-3.1-70b"
    assert url == "https://api.groq.com/openai/v1"
    # GROQ_API_KEY wins because it's listed first in the per-prefix tuple
    # — operator with both keys set doesn't accidentally shadow groq with
    # the openai fallback.
    assert key == "gsk-test"


def test_groq_prefix_falls_back_to_openai_key(resolve):
    """Some operators only set OPENAI_API_KEY for a groq endpoint —
    the per-prefix tuple lists ``OPENAI_API_KEY`` as the second-choice
    fallback for groq specifically. Pin the fallback so removing it
    breaks loudly."""
    prefix, model, url, key = resolve(
        "groq:llama-3.1-70b",
        env={"OPENAI_API_KEY": "sk-only"},
        runtime_config={},
    )
    assert prefix == "groq"
    assert key == "sk-only"


def test_qianfan_prefix_with_aistudio_fallback(resolve):
    prefix, model, url, key = resolve(
        "qianfan:ernie-4.0",
        env={"AISTUDIO_API_KEY": "aist-test"},
        runtime_config={},
    )
    assert prefix == "qianfan"
    assert model == "ernie-4.0"
    assert url == "https://qianfan.baidubce.com/v2"
    assert key == "aist-test"


def test_anthropic_prefix_reroutes_via_openrouter(resolve):
    """``anthropic:<id>`` → ``openrouter`` with model rewritten to
    ``anthropic/<id>`` slash-form. This is the load-bearing fix:
    OpenRouter exposes Claude under the OpenAI-compat API at the
    slash-form id, and openclaw is OpenAI-compat only, so this is
    the one path that actually returns Claude tokens."""
    prefix, model, url, key = resolve(
        "anthropic:claude-opus-4-7",
        env={"OPENROUTER_API_KEY": "or-test"},
        runtime_config={},
    )
    assert prefix == "openrouter"
    assert model == "anthropic/claude-opus-4-7"
    assert url == "https://openrouter.ai/api/v1"
    assert key == "or-test"


def test_claude_prefix_reroutes_via_openrouter(resolve):
    """``claude:<id>`` is treated as an alias for ``anthropic:<id>``
    so wheel/CrewAI/langchain users typing either spelling land on
    the same OpenRouter path."""
    prefix, model, url, _ = resolve(
        "claude:claude-sonnet-4",
        env={"OPENROUTER_API_KEY": "or-test"},
        runtime_config={},
    )
    assert prefix == "openrouter"
    assert model == "anthropic/claude-sonnet-4"


def test_anthropic_prefix_without_openrouter_key_raises(resolve):
    """Fail fast with actionable guidance — silently routing to
    OPENAI_API_KEY + api.openai.com was the original bug shape."""
    with pytest.raises(RuntimeError) as exc:
        resolve(
            "anthropic:claude-opus-4-7",
            env={"OPENAI_API_KEY": "sk-only"},
            runtime_config={},
        )
    msg = str(exc.value)
    assert "OPENROUTER_API_KEY" in msg
    assert "openclaw is OpenAI-" in msg


def test_no_api_key_for_resolved_prefix_raises(resolve):
    """No key set at all → RuntimeError that names which env vars
    were searched, so an operator can paste it straight into Railway
    or 1Password."""
    with pytest.raises(RuntimeError) as exc:
        resolve(
            "openai:gpt-4o",
            env={},
            runtime_config={},
        )
    msg = str(exc.value)
    assert "no API key found" in msg
    assert "OPENAI_API_KEY" in msg


def test_runtime_config_provider_url_overrides_default(resolve):
    """An operator can pin a self-hosted OpenAI-compat gateway via
    ``runtime_config.provider_url`` (e.g. an Azure-OpenAI deployment)
    — pin that the runtime_config override wins over the per-prefix
    default."""
    _, _, url, _ = resolve(
        "openai:gpt-4o",
        env={"OPENAI_API_KEY": "sk-test"},
        runtime_config={"provider_url": "https://my-gateway.internal/v1"},
    )
    assert url == "https://my-gateway.internal/v1"


def test_runtime_config_none_uses_default(resolve):
    """Defensive: setup() always passes ``config.runtime_config`` (a
    dict) but the helper accepts ``None`` so unit tests / ad-hoc
    callers don't have to fabricate one."""
    _, _, url, _ = resolve(
        "openai:gpt-4o",
        env={"OPENAI_API_KEY": "sk-test"},
        runtime_config=None,
    )
    assert url == "https://api.openai.com/v1"


def test_unknown_prefix_falls_back_to_openai_key(resolve):
    """Unknown prefix → falls back to the OPENAI key/url path so
    operator-supplied prefixes that genuinely *are* OpenAI-compat
    pass through. anthropic/claude is the only explicit re-route."""
    prefix, model, url, key = resolve(
        "togetherai:meta-llama-3.1-70b",
        env={"OPENAI_API_KEY": "sk-test"},
        runtime_config={},
    )
    assert prefix == "togetherai"
    assert model == "meta-llama-3.1-70b"
    assert url == "https://api.openai.com/v1"
    assert key == "sk-test"
