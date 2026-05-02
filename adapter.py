"""OpenClaw adapter — bridges OpenClaw's Node.js gateway with our A2A protocol.

OpenClaw is a Node.js agent runtime with its own gateway (port 18789).
This adapter:
1. Installs OpenClaw CLI (npm) and missing deps in the container
2. Runs non-interactive onboard with the configured model provider
3. Copies workspace files (SOUL.md, BOOTSTRAP.md, etc.) to OpenClaw's workspace dir
4. Starts the OpenClaw gateway as a background process
5. Proxies A2A messages via `openclaw agent --json` CLI subprocess
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess

from molecule_runtime.adapters.base import BaseAdapter, AdapterConfig
from molecule_runtime.adapters.shared_runtime import brief_task, extract_message_text, set_current_task
from a2a.server.agent_execution import AgentExecutor

logger = logging.getLogger(__name__)

OPENCLAW_WORKSPACE = os.path.expanduser("~/.openclaw/workspace-dev/main")
OPENCLAW_PORT = 18789

# Known missing optional deps in OpenClaw's npm package
OPENCLAW_MISSING_DEPS = ["@buape/carbon", "@larksuiteoapi/node-sdk", "@slack/web-api", "grammy"]

# Per-prefix API-key env var lookup. Each tuple is searched in order and
# the first env var present wins, so an operator running multiple
# providers can keep both keys set without one accidentally shadowing
# the other.
_API_KEY_BY_PREFIX = {
    "openai":     ("OPENAI_API_KEY",),
    "groq":       ("GROQ_API_KEY", "OPENAI_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "qianfan":    ("QIANFAN_API_KEY", "AISTUDIO_API_KEY"),
}

# OpenAI-compat base URL for each routed provider.
_PROVIDER_URLS = {
    "openai":     "https://api.openai.com/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "qianfan":    "https://qianfan.baidubce.com/v2",
}


def _resolve_provider_routing(model_str, env, runtime_config=None):
    """Translate a LangChain-style ``<provider>:<id>`` model string into
    the (prefix, model_id, provider_url, api_key) tuple openclaw needs.

    OpenClaw is OpenAI-compatible only — its ``--custom-compatibility``
    flag is hard-set to ``openai`` in setup() below. Model strings
    arrive in LangChain-style ``<provider>:<id>`` form (the wheel's
    config.py default is ``anthropic:claude-opus-4-7`` so
    langchain/crewai consumers get a uniform string out of the box).

    Routing rules:

    * openai/groq/openrouter/qianfan → existing per-prefix routing,
      each provider exposes an OpenAI-compat endpoint directly.
    * anthropic/claude → re-route through OpenRouter, which exposes
      Claude under the OpenAI-compat API at ``anthropic/<id>``
      slash-form. Caught 2026-05-01: without this re-route,
      ``anthropic:claude-X`` landed on the OPENAI_API_KEY +
      api.openai.com path with ``claude-X`` as the model id, which
      OpenAI doesn't host — every inference call failed silently and
      the workspace looked online but was structurally broken.
    * bare model id (no ``:``) → openai (legacy default, unchanged).
    * unknown prefix → falls back to OPENAI_API_KEY/api.openai.com so
      operator-supplied prefixes that genuinely *are* OpenAI-compat
      pass through; explicit anthropic/claude is the only re-route.

    Pure: takes ``env`` (a Mapping) and ``runtime_config`` (a Mapping
    or None) so the test suite can exercise every branch without
    monkeypatching ``os.environ``.
    """
    if ":" in model_str:
        prefix, model = model_str.split(":", 1)
    else:
        prefix, model = "openai", model_str

    if prefix in ("anthropic", "claude"):
        if not env.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                f"openclaw adapter: model={model_str!r} requires "
                "Anthropic/Claude routing but openclaw is OpenAI-"
                "compatible only. Either: (a) set OPENROUTER_API_KEY in "
                "workspace secrets so the adapter can route via "
                "OpenRouter (which exposes Claude under OpenAI-compat "
                "API), or (b) pick a model from the supported provider "
                "list (openai/groq/openrouter/qianfan) and set "
                "MODEL_PROVIDER on the workspace, e.g. "
                "openrouter:anthropic/claude-sonnet-4."
            )
        # OpenRouter exposes Claude under `anthropic/<id>` slash form.
        model = f"anthropic/{model}"
        prefix = "openrouter"

    env_vars = _API_KEY_BY_PREFIX.get(prefix, ("OPENAI_API_KEY",))
    api_key = next((env[v] for v in env_vars if env.get(v)), "")
    if not api_key:
        raise RuntimeError(
            f"openclaw adapter: no API key found for prefix={prefix!r} "
            f"(checked: {', '.join(env_vars)}). Set one of those env "
            f"vars in workspace secrets."
        )

    default_url = _PROVIDER_URLS.get(prefix, _PROVIDER_URLS["openai"])
    if runtime_config is not None:
        provider_url = runtime_config.get("provider_url", default_url)
    else:
        provider_url = default_url

    return prefix, model, provider_url, api_key


class OpenClawAdapter(BaseAdapter):

    def __init__(self):
        self._gateway_process = None

    @staticmethod
    def name() -> str:
        return "openclaw"

    @staticmethod
    def display_name() -> str:
        return "OpenClaw"

    @staticmethod
    def description() -> str:
        return "OpenClaw agent runtime — Node.js gateway with SOUL/BOOTSTRAP/AGENTS workspace convention"

    @staticmethod
    def get_config_schema() -> dict:
        return {
            "model": {"type": "string", "description": "Model ID (e.g. google/gemini-2.5-flash)"},
            "provider_url": {"type": "string", "description": "LLM provider base URL", "default": "https://openrouter.ai/api/v1"},
            "gateway_port": {"type": "integer", "description": "OpenClaw gateway port", "default": 18789},
        }

    async def setup(self, config: AdapterConfig) -> None:  # pragma: no cover
        """Install OpenClaw, run onboard, copy workspace files, start gateway."""
        # Boot-smoke contract (molecule-core#2275): the publish-image gate
        # invokes us with stub creds + no network so it can exercise lazy
        # imports inside execute(). Real gateway spawn would fail here
        # (no valid api_key, no `openclaw` binary on PATH yet), so skip
        # the heavy setup path entirely. The runtime's smoke_mode short-
        # circuit fires immediately after create_executor() returns.
        if os.environ.get("MOLECULE_SMOKE_MODE") == "1":
            logger.info("MOLECULE_SMOKE_MODE=1 — skipping OpenClaw gateway spawn")
            return

        npm_prefix = os.path.expanduser("~/.local")
        os.environ["PATH"] = f"{npm_prefix}/bin:{os.environ.get('PATH', '')}"

        # 1. Install OpenClaw CLI if not present
        if not shutil.which("openclaw"):
            logger.info("Installing OpenClaw CLI...")
            result = subprocess.run(
                ["npm", "install", "--prefix", npm_prefix, "-g", "openclaw"],
                capture_output=True, text=True, timeout=300,
                env={**os.environ, "npm_config_prefix": npm_prefix}
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install OpenClaw: {result.stderr[:500]}")

            # Install known missing optional deps
            oc_dir = os.path.join(npm_prefix, "lib/node_modules/openclaw")
            if os.path.exists(oc_dir):
                logger.info("Installing OpenClaw optional deps...")
                subprocess.run(
                    ["npm", "install"] + OPENCLAW_MISSING_DEPS,
                    capture_output=True, text=True, timeout=120, cwd=oc_dir
                )
            logger.info("OpenClaw CLI installed")

        # 2. Resolve API key and model via the pure routing helper.
        prefix, model, provider_url, api_key = _resolve_provider_routing(
            config.model, os.environ, config.runtime_config
        )
        if prefix == "openrouter" and config.model.split(":", 1)[0] in ("anthropic", "claude"):
            logger.info(
                "openclaw adapter: rerouting anthropic-prefixed model via OpenRouter (model=%s)",
                model,
            )

        # 3. Run non-interactive onboard
        if not os.path.exists(os.path.expanduser("~/.openclaw/openclaw.json")):
            logger.info(f"Running OpenClaw onboard (model: {model})...")
            subprocess.run(
                ["openclaw", "onboard", "--non-interactive",
                 "--auth-choice", "custom-api-key",
                 "--custom-base-url", provider_url,
                 "--custom-model-id", model,
                 "--custom-api-key", api_key,
                 "--custom-compatibility", "openai",
                 "--secret-input-mode", "plaintext",
                 "--accept-risk", "--skip-health"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "NODE_NO_WARNINGS": "1"}
            )
            logger.info("OpenClaw onboard complete")

        # 3b. Fix context window (OpenClaw defaults to 16K, but modern models have much more)
        oc_config_path = os.path.expanduser("~/.openclaw/openclaw.json")
        if os.path.exists(oc_config_path):
            try:
                import json as json_mod
                oc_cfg = json_mod.load(open(oc_config_path))
                provider_name = "custom-" + provider_url.split("//")[1].split("/")[0].replace(".", "-")
                providers = oc_cfg.get("models", {}).get("providers", {})
                if provider_name in providers:
                    for m in providers[provider_name].get("models", []):
                        m["contextWindow"] = 1000000  # 1M tokens for modern models
                        m["maxTokens"] = 16384
                    json_mod.dump(oc_cfg, open(oc_config_path, "w"), indent=2)
                    logger.info(f"Fixed context window for {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to fix context window: {e}")

        # 3c. Always write auth-profiles.json
        # (key may have been set via secrets API after first boot)
        if api_key:
            auth_dir = os.path.expanduser("~/.openclaw/agents/main/agent")
            os.makedirs(auth_dir, exist_ok=True)
            auth_file = os.path.join(auth_dir, "auth-profiles.json")
            import json as json_mod
            provider_name = "custom-" + provider_url.split("//")[1].split("/")[0].replace(".", "-")
            auth_data = {provider_name: {"type": "api-key", "key": api_key}}
            with open(auth_file, "w") as f:
                json_mod.dump(auth_data, f, indent=2)
            logger.info(f"Wrote auth-profiles.json for {provider_name}")

        # 4. Copy workspace files from /configs to OpenClaw's workspace dir
        os.makedirs(OPENCLAW_WORKSPACE, exist_ok=True)
        for fname in os.listdir(config.config_path):
            src = os.path.join(config.config_path, fname)
            if os.path.isfile(src) and fname.endswith(".md"):
                shutil.copy2(src, os.path.join(OPENCLAW_WORKSPACE, fname))
                logger.debug(f"Copied {fname} to OpenClaw workspace")

        # 5. Start the gateway as a background process
        gateway_port = config.runtime_config.get("gateway_port", OPENCLAW_PORT)
        logger.info(f"Starting OpenClaw gateway on port {gateway_port}...")
        env = os.environ.copy()
        env["NODE_NO_WARNINGS"] = "1"
        self._gateway_process = subprocess.Popen(
            ["openclaw", "gateway", "--dev", "--port", str(gateway_port), "--bind", "loopback"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env,
        )
        # Wait for gateway to become healthy (max 30s)
        for attempt in range(15):
            await asyncio.sleep(2)
            if self._gateway_process.poll() is not None:
                raise RuntimeError("OpenClaw gateway process exited")
            try:
                health = subprocess.run(
                    ["openclaw", "gateway", "health"],
                    capture_output=True, text=True, timeout=10,
                    env=os.environ.copy()
                )
                if health.returncode == 0:
                    logger.info(f"OpenClaw gateway healthy (PID: {self._gateway_process.pid})")
                    break
            except subprocess.TimeoutExpired:
                logger.debug(f"Gateway health check timeout (attempt {attempt+1}/15)")
        else:
            raise RuntimeError("OpenClaw gateway did not become healthy within 30s")

    async def create_executor(self, config: AdapterConfig) -> AgentExecutor:
        return OpenClawA2AExecutor(heartbeat=config.heartbeat)


class OpenClawA2AExecutor(AgentExecutor):
    """Proxies A2A messages to OpenClaw via `openclaw agent` CLI subprocess."""

    def __init__(self, heartbeat=None):
        self._heartbeat = heartbeat

    async def execute(self, context, event_queue):
        from a2a.helpers import new_text_message

        user_message = extract_message_text(context)

        if not user_message:
            await event_queue.enqueue_event(new_text_message("No message provided"))
            return

        await set_current_task(self._heartbeat, brief_task(user_message))

        # Call OpenClaw agent via CLI
        try:
            proc = await asyncio.create_subprocess_exec(
                "openclaw", "agent",
                "--session-id", context.task_id or "default",
                "--message", user_message,
                "--json", "--timeout", "120",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PATH": f"{os.path.expanduser('~/.local/bin')}:{os.environ.get('PATH', '')}"}
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=130)
            output = stdout.decode().strip()

            if proc.returncode == 0 and output:
                try:
                    data = json.loads(output)
                    payloads = data.get("result", {}).get("payloads", [])
                    if payloads:
                        reply = payloads[0].get("text", "")
                    else:
                        reply = str(data)
                except json.JSONDecodeError:
                    reply = output
            else:
                reply = f"OpenClaw error: {stderr.decode()[:300]}" if stderr else f"OpenClaw returned code {proc.returncode}"

        except asyncio.TimeoutError:
            reply = "OpenClaw timed out after 120s"
        except Exception as e:
            reply = f"OpenClaw error: {e}"
        finally:
            await set_current_task(self._heartbeat, "")

        await event_queue.enqueue_event(new_text_message(reply))

    async def cancel(self, context, event_queue):  # pragma: no cover
        pass


Adapter = OpenClawAdapter
