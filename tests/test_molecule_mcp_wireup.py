"""Pin the molecule a2a MCP wire-up emitted into ``~/.openclaw/openclaw.json``.

Why this exists
---------------
``molecule_runtime`` ships ``a2a_mcp_server.py`` exposing platform A2A
primitives — ``list_peers``, ``delegate_task``, ``commit_memory``,
``recall_memory``, etc. — as MCP tools. claude-code gets this for free
via ``claude_sdk_executor._build_options`` injecting an "a2a" entry
into ``ClaudeAgentOptions.mcp_servers``. openclaw needs an equivalent
hand-off because its discovery surface is its own ``mcp.servers``
config block (cf. ``openclaw mcp --help``).

These tests pin the wire-up's externally-observable shape so a future
refactor doesn't silently drop:

  - the env-passthrough (WORKSPACE_ID, PLATFORM_URL, MOLECULE_ORG_ID,
    CONFIGS_DIR, A2A_MCP_SERVER_PATH) — without these the MCP server
    starts but every /registry/peers call 400s on missing ID,
  - the ``sys.executable`` interpreter selection (matching claude-
    code's pattern so the same Python that loaded molecule_runtime
    also runs the MCP server),
  - the soft-fail behaviour in ``_register_molecule_mcp`` (missing
    WORKSPACE_ID, missing a2a_mcp_server.py, openclaw CLI failure all
    log + return False rather than crash setup() — MCP wire-up
    failure is degraded behaviour, not a fatal provision failure).
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock

import pytest


# ---- Stubs ----
# Match the shape used by tests/test_model_routing.py so adapter.py can
# import in CI environments without the real molecule_runtime wheel.


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
def adapter_mod():
    return _load_adapter()


# ---- _build_molecule_mcp_config — pure shape ----


def test_build_mcp_config_minimum_shape(adapter_mod):
    """No env passthrough vars set → just command + args, no env block.

    The MCP server is happy to fall back to its module-level defaults
    (PLATFORM_URL='http://platform:8080', WORKSPACE_ID='') when env is
    silent — every call will then 400 in production, but locally /
    in dev that's still better than refusing to register.
    """
    cfg = adapter_mod._build_molecule_mcp_config(
        env={},
        mcp_server_path="/path/to/a2a_mcp_server.py",
        python_executable="/usr/bin/python3",
    )
    assert cfg == {
        "command": "/usr/bin/python3",
        "args": ["/path/to/a2a_mcp_server.py"],
    }
    assert "env" not in cfg


def test_build_mcp_config_passes_through_workspace_id(adapter_mod):
    cfg = adapter_mod._build_molecule_mcp_config(
        env={"WORKSPACE_ID": "ws-123", "PLATFORM_URL": "http://platform:8080"},
        mcp_server_path="/x.py",
        python_executable="/py",
    )
    assert cfg["env"] == {
        "WORKSPACE_ID": "ws-123",
        "PLATFORM_URL": "http://platform:8080",
    }


def test_build_mcp_config_passes_through_org_id_and_configs_dir(adapter_mod):
    """SaaS deployments need MOLECULE_ORG_ID for TenantGuard and
    CONFIGS_DIR for the on-disk auth token. Both flow through."""
    cfg = adapter_mod._build_molecule_mcp_config(
        env={
            "WORKSPACE_ID": "ws-1",
            "MOLECULE_ORG_ID": "org-uuid",
            "CONFIGS_DIR": "/configs",
        },
        mcp_server_path="/x.py",
        python_executable="/py",
    )
    assert cfg["env"]["MOLECULE_ORG_ID"] == "org-uuid"
    assert cfg["env"]["CONFIGS_DIR"] == "/configs"


def test_build_mcp_config_passes_through_a2a_mcp_server_path_override(adapter_mod):
    """Templates in non-default layouts can pin the script path via
    A2A_MCP_SERVER_PATH — the wire-up forwards it so the MCP server
    inherits the same override on respawn."""
    cfg = adapter_mod._build_molecule_mcp_config(
        env={"WORKSPACE_ID": "ws-1", "A2A_MCP_SERVER_PATH": "/custom/mcp.py"},
        mcp_server_path="/whatever.py",
        python_executable="/py",
    )
    assert cfg["env"]["A2A_MCP_SERVER_PATH"] == "/custom/mcp.py"


def test_build_mcp_config_skips_unset_env_vars(adapter_mod):
    """Empty-string env vars are dropped — the helper never emits
    ``WORKSPACE_ID=""`` because that would mask the actual missing-
    config diagnostic the MCP server prints when WORKSPACE_ID is
    absent."""
    cfg = adapter_mod._build_molecule_mcp_config(
        env={"WORKSPACE_ID": "", "PLATFORM_URL": "http://x"},
        mcp_server_path="/x.py",
        python_executable="/py",
    )
    assert "WORKSPACE_ID" not in cfg.get("env", {})
    assert cfg["env"]["PLATFORM_URL"] == "http://x"


def test_build_mcp_config_does_not_pass_through_unrelated_env(adapter_mod):
    """Defence-in-depth: the helper only forwards the documented
    passthrough list. A leaked OPENAI_API_KEY in the openclaw config
    file would be a real security regression — pin against accidental
    blanket-forward."""
    cfg = adapter_mod._build_molecule_mcp_config(
        env={
            "WORKSPACE_ID": "ws-1",
            "OPENAI_API_KEY": "sk-secret",
            "MINIMAX_API_KEY": "mm-secret",
            "ANTHROPIC_API_KEY": "ant-secret",
        },
        mcp_server_path="/x.py",
        python_executable="/py",
    )
    forwarded = cfg.get("env", {})
    assert "OPENAI_API_KEY" not in forwarded
    assert "MINIMAX_API_KEY" not in forwarded
    assert "ANTHROPIC_API_KEY" not in forwarded


# ---- _register_molecule_mcp — soft-fail / happy-path ----


def test_register_skips_without_workspace_id(adapter_mod, caplog):
    """No WORKSPACE_ID → return False, log info, do NOT shell out to
    openclaw. Dev/smoke runs land here and must not fail loudly."""
    import logging

    caplog.set_level(logging.INFO)
    assert adapter_mod._register_molecule_mcp(env={}) is False
    assert any("WORKSPACE_ID not set" in rec.message for rec in caplog.records)


def test_register_skips_when_mcp_server_unresolvable(adapter_mod, monkeypatch, caplog):
    """No a2a_mcp_server.py on disk (e.g. older runtime wheel without
    the helper) → return False, log warning, do NOT shell out."""
    import logging

    monkeypatch.setattr(adapter_mod, "_resolve_mcp_server_path", lambda: None)
    caplog.set_level(logging.WARNING)
    assert adapter_mod._register_molecule_mcp(env={"WORKSPACE_ID": "w"}) is False
    assert any("a2a_mcp_server.py not found" in rec.message for rec in caplog.records)


def test_register_happy_path_invokes_openclaw_mcp_set(adapter_mod, monkeypatch, tmp_path):
    """All preconditions met → call ``openclaw mcp set molecule '<json>'``
    and return True. The recorded subprocess argv must:

      - end with a parseable JSON blob,
      - encode the resolved python interpreter (sys.executable),
      - encode env passthrough for WORKSPACE_ID + PLATFORM_URL.
    """
    fake_mcp_path = tmp_path / "a2a_mcp_server.py"
    fake_mcp_path.write_text("# stub")
    monkeypatch.setattr(
        adapter_mod, "_resolve_mcp_server_path", lambda: str(fake_mcp_path)
    )

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    monkeypatch.setattr(adapter_mod.subprocess, "run", fake_run)

    env = {
        "WORKSPACE_ID": "ws-abc",
        "PLATFORM_URL": "http://platform:8080",
        "MOLECULE_ORG_ID": "org-1",
    }
    assert adapter_mod._register_molecule_mcp(env=env) is True

    argv = captured["argv"]
    assert argv[0:4] == ["openclaw", "mcp", "set", "molecule"]
    payload = json.loads(argv[4])
    assert payload["command"] == sys.executable
    assert payload["args"] == [str(fake_mcp_path)]
    assert payload["env"] == {
        "WORKSPACE_ID": "ws-abc",
        "PLATFORM_URL": "http://platform:8080",
        "MOLECULE_ORG_ID": "org-1",
    }


def test_register_returns_false_on_openclaw_nonzero_exit(adapter_mod, monkeypatch, tmp_path, caplog):
    """openclaw rejected the entry (e.g. schema drift between npm
    versions) → log warning + return False. setup() continues."""
    import logging

    fake_mcp_path = tmp_path / "a2a_mcp_server.py"
    fake_mcp_path.write_text("# stub")
    monkeypatch.setattr(
        adapter_mod, "_resolve_mcp_server_path", lambda: str(fake_mcp_path)
    )

    def fake_run(argv, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stderr = "openclaw: unknown subcommand"
        return result

    monkeypatch.setattr(adapter_mod.subprocess, "run", fake_run)

    caplog.set_level(logging.WARNING)
    assert adapter_mod._register_molecule_mcp(env={"WORKSPACE_ID": "w"}) is False
    assert any("openclaw mcp set failed" in rec.message for rec in caplog.records)


def test_register_returns_false_when_subprocess_raises(adapter_mod, monkeypatch, tmp_path, caplog):
    """openclaw binary not on PATH (FileNotFoundError) → soft-fail with
    a warning. This is the realistic dev path where the npm install
    step inside setup() either ran in a container we're not exercising
    here or simply isn't installed."""
    import logging
    import subprocess as _subprocess

    fake_mcp_path = tmp_path / "a2a_mcp_server.py"
    fake_mcp_path.write_text("# stub")
    monkeypatch.setattr(
        adapter_mod, "_resolve_mcp_server_path", lambda: str(fake_mcp_path)
    )

    def fake_run(argv, **kwargs):
        raise FileNotFoundError("openclaw not on PATH")

    monkeypatch.setattr(adapter_mod.subprocess, "run", fake_run)

    caplog.set_level(logging.WARNING)
    assert adapter_mod._register_molecule_mcp(env={"WORKSPACE_ID": "w"}) is False
    assert any("openclaw mcp set raised" in rec.message for rec in caplog.records)


def test_register_uses_os_environ_by_default(adapter_mod, monkeypatch):
    """Calling _register_molecule_mcp() with no env arg falls back to
    os.environ — the real setup() code path."""
    fake_mcp_path = "/tmp/nonexistent_mcp.py"
    # Force the resolver to return the same path the os.path.isfile guard
    # will reject — so the function returns False without shelling out.
    monkeypatch.setattr(
        adapter_mod, "_resolve_mcp_server_path", lambda: fake_mcp_path
    )
    monkeypatch.delenv("WORKSPACE_ID", raising=False)
    # No explicit env kwarg — must default to os.environ which lacks WS_ID.
    assert adapter_mod._register_molecule_mcp() is False


# ---- name / passthrough constants -----------------------------------


def test_molecule_mcp_server_name_constant(adapter_mod):
    """Pin the name 'molecule' so the system prompt + docs (which
    reference ``mcp__molecule__list_peers`` etc.) stay in sync with
    the wire-up."""
    assert adapter_mod._MOLECULE_MCP_NAME == "molecule"


def test_passthrough_env_vars_include_required_set(adapter_mod):
    """Pin the documented passthrough list. Removing one would silently
    break either workspace identity (WORKSPACE_ID), routing
    (PLATFORM_URL), SaaS auth (MOLECULE_ORG_ID), or token discovery
    (CONFIGS_DIR)."""
    required = {"WORKSPACE_ID", "PLATFORM_URL", "MOLECULE_ORG_ID", "CONFIGS_DIR"}
    assert required.issubset(set(adapter_mod._MCP_PASSTHROUGH_ENV_VARS))
