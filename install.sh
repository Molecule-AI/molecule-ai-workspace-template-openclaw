#!/usr/bin/env bash
# install.sh — host-install hook for the openclaw runtime.
#
# Why this exists
# ---------------
# openclaw's npm package (openclaw@2026.4.29 at time of writing) pins
# `engines.node >= 22.14.0`. The default Ubuntu 24.04 workspace AMI
# ships Node 18.19.1 from apt. When adapter.py:setup() runs
# `npm install --prefix ~/.local -g openclaw` against Node 18, npm
# either fails outright or surfaces a successful install whose
# postinstall scripts crash on first invocation — both end as
# `RuntimeError: Failed to install OpenClaw` from adapter.py:71, the
# molecule-runtime never binds port 8000, and the workspace flips to
# status=failed within ~2 minutes of provision.
#
# Diagnosed 2026-05-01 from /var/log/molecule-runtime.log on a failed
# provision:
#
#     RuntimeError: Failed to install OpenClaw: npm WARN EBADENGINE
#       package: 'openclaw@2026.4.29',
#       required: { node: '>=22.14.0' },
#       current: { node: 'v18.19.1', npm: '9.2.0' }
#
# What this does
# --------------
# Installs Node 22 LTS from nodesource into the system PATH. apt is
# the standard way to install Node on Ubuntu and matches openclaw's
# own README. The runtime user has passwordless sudo on the AMI, so
# we don't need to fall back to a user-local tarball install. Node 22
# is the LTS line that openclaw upstream targets.
#
# Idempotent — if Node ≥22 is already on PATH (e.g. AMI gets bumped
# in a future cycle and this hook becomes a no-op), early-exits 0
# without touching apt.
#
# Tracked: this is template-local "system deps" workaround. The
# durable fix is bumping the workspace AMI to ship Node 22 directly,
# which would also benefit gemini-cli and any future Node-based
# runtimes. Until then, this hook keeps openclaw provisioning green.

set -euo pipefail

current_major() {
  command -v node >/dev/null 2>&1 || { echo 0; return; }
  node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1
}

if [ "$(current_major)" -ge 22 ]; then
  echo "Node $(node --version) already satisfies >=22 — skipping install"
  exit 0
fi

echo "Installing Node 22 from nodesource (had: $(node --version 2>/dev/null || echo 'none'))"

# nodesource setup_22.x: configures /etc/apt/sources.list.d/nodesource.list
# and refreshes apt cache. Re-running on an already-configured host is a
# no-op, so this stays idempotent across reboots / re-runs.
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -

# `apt-get install -y nodejs` from nodesource provides Node 22 + npm 10.
# --no-install-recommends keeps the install slim (no recommended-but-unused
# build toolchain pulled in transitively).
sudo apt-get install -y --no-install-recommends nodejs

echo "Node $(node --version) installed; npm $(npm --version)"

# --- MiniMax routing ---------------------------------------------------
# Same gap codex template hits: the provisioner has a MODEL_PROVIDER
# env→config.yaml pass-through (ec2.go:1923) but never exports
# MODEL_PROVIDER from user-data. Result: openclaw's adapter sees the
# molecule_runtime library default `anthropic:claude-opus-4-7` and
# fails fast with the "model requires Anthropic/Claude routing but
# openclaw is OpenAI-compatible only" RuntimeError, /registry/register
# never fires, the workspace flips to status=failed inside the
# provisioning timeout window. Caught live during the 4-runtime A2A
# E2E (2026-05-03).
#
# When the operator's only LLM key is MINIMAX_API_KEY, route the
# `openai` prefix at MiniMax's OpenAI-compat endpoint and pin the
# default model to a MiniMax one. The adapter's `_resolve_provider_routing`
# already honors `OPENAI_BASE_URL` (cf. adapter.py:111), so this hooks
# into the supported override surface — no adapter change needed.
if [ -n "${MINIMAX_API_KEY:-}" ] && [ -z "${MODEL_PROVIDER:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  WORKSPACE_CONFIG_DIR="${WORKSPACE_CONFIG_PATH:-/configs}"
  WORKSPACE_CONFIG="${WORKSPACE_CONFIG_DIR}/config.yaml"
  OPENCLAW_MINIMAX_MODEL="${OPENCLAW_MINIMAX_MODEL:-MiniMax-M2.1}"
  if [ -f "$WORKSPACE_CONFIG" ] && [ -w "$WORKSPACE_CONFIG_DIR" ]; then
    if grep -qE '^model:' "$WORKSPACE_CONFIG"; then
      sed -i.bak "s|^model: .*|model: 'openai:${OPENCLAW_MINIMAX_MODEL}'|" "$WORKSPACE_CONFIG" && rm -f "${WORKSPACE_CONFIG}.bak"
    else
      printf "model: 'openai:%s'\n" "$OPENCLAW_MINIMAX_MODEL" >> "$WORKSPACE_CONFIG"
    fi
    echo "[install.sh] patched ${WORKSPACE_CONFIG} model=openai:${OPENCLAW_MINIMAX_MODEL} (MiniMax routing via OpenAI-compat)"
  elif [ -f "$WORKSPACE_CONFIG" ]; then
    echo "[install.sh] WARN: ${WORKSPACE_CONFIG} not writable; runtime may fall back to default model" >&2
  fi
  # Persist OPENAI_API_KEY + OPENAI_BASE_URL so the adapter's setup()
  # picks them up on its first env read. /etc/environment is the
  # standard place for system-wide env on Ubuntu cloud-init AMIs.
  if [ -w /etc/environment ] || sudo -n true 2>/dev/null; then
    sudo bash -c "{
      echo 'OPENAI_API_KEY=${MINIMAX_API_KEY}'
      echo 'OPENAI_BASE_URL=${MINIMAX_API_BASE:-https://api.minimax.io/v1}'
    } >> /etc/environment"
    echo "[install.sh] exported OPENAI_API_KEY=<MINIMAX_API_KEY> OPENAI_BASE_URL=${MINIMAX_API_BASE:-https://api.minimax.io/v1} → /etc/environment"
  fi
fi
