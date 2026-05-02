# Known Issues

The following issues are tracked for the openclaw workspace template. Each entry describes the symptom, root cause, affected components, and current workaround.

---

## Issue 1 — openclaw version pinned to old release

**Severity:** High  
**First introduced:** v0.1.0 (initial release)  
**Components affected:** `requirements.txt`

### Symptom

After installing `requirements.txt`, the agent runs with an outdated version of the openclaw runtime that does not support token streaming or the latest tool-call schema. Tasks that require streaming output appear to hang; tools defined in `config.yaml` with the newer `strict: true` flag are rejected by the runtime.

### Root cause

`requirements.txt` pins `openclaw-runtime==0.9.3`, which was released before the streaming and strict-tool-schema features landed. No upper bound prevents pip from upgrading, but the template has not been updated to point to the latest release (`0.11.2`).

### Current workaround

Override the pinned version when installing or at build time:

```bash
pip install openclaw-runtime==0.11.2
```

Or update `requirements.txt` locally until the template is patched:

```
openclaw-runtime==0.11.2
```

### Tracking

- Filed: 2025-01-08
- Ticket: OWC-201

---

## Issue 2 — adapter.py not forwarding context to platform

**Severity:** High  
**First introduced:** v0.2.0  
**Components affected:** `adapter.py`

### Symptom

When the agent completes a task, the platform receives the final output message but all intermediate `context` fields (tool call history, token usage, reasoning steps) are omitted. The platform task history shows only the final assistant message. This makes it impossible to audit tool-call chains or calculate token costs post-hoc.

### Root cause

`adapter.py`'s `build_callback_payload()` method copies `output.text` into the platform payload but discards the `context` dict returned by the openclaw runtime. The method signature accepted the context but it was not serialized into the JSON sent to the `tasks.complete` endpoint.

### Current workaround

Set `OPENCLAW_FORWARD_CONTEXT=1` before running the container:

```bash
export OPENCLAW_FORWARD_CONTEXT=1
docker run --rm --env-file .env openclaw-local:latest
```

When this flag is set, `adapter.py` serializes the full context dict into the callback payload under the `extended_metadata` key.

### Tracking

- Filed: 2025-03-05
- Ticket: OWC-223

---

## Issue 3 — system-prompt.md injected twice in some conditions

**Severity:** Medium  
**First introduced:** v0.2.1  
**Components affected:** `adapter.py`, `system-prompt.md`

### Symptom

In long-running sessions where the platform sends a `session.resume` event (e.g., after a platform-side reconnect), the agent's system prompt appears doubled: the base prompt from `system-prompt.md` is prepended again on top of the accumulated context, causing the agent to re-read its own instructions and produce confused or repetitive output.

### Root cause

`adapter.py` injects `system-prompt.md` unconditionally inside the `handle_session_resume()` branch, without checking whether the prompt has already been injected in the current session. The accumulated context already contains the original prompt; the second injection doubles it.

### Current workaround

Disable automatic resume prompt injection by setting `OPENCLAW_NO_RESUME_REINJECT=1`:

```bash
export OPENCLAW_NO_RESUME_REINJECT=1
```

The session will resume with the accumulated context as-is, without re-injecting the base system prompt. Operators should verify that `system-prompt.md` contains no stateful instructions that would be needed at resume time.

### Tracking

- Filed: 2025-03-19
- Ticket: OWC-241

---

## Issue 4 — config.yaml skill list not merged with platform defaults

**Severity:** Medium  
**First introduced:** v0.2.0  
**Components affected:** `config.yaml`, `adapter.py`

### Symptom

The operator declares skills in `config.yaml` under the `skills:` key. When the workspace boots, only the skills from the platform defaults are active; the skills listed in the local `config.yaml` are silently ignored. The agent does not receive the relevant skill prompt fragments.

### Root cause

`adapter.py` reads the platform's default skill list and assigns it directly to the session, then overwrites it with the locally parsed `config.yaml` skills list using a direct assignment instead of a merge operation. This results in the local list replacing (not augmenting) the platform defaults. The intended behaviour is a additive merge.

### Current workaround

Set `OPENCLAW_SKILL_LIST` as an environment variable instead of (or in addition to) `config.yaml`. Environment-variable skill lists are additive and take precedence over platform defaults:

```bash
export OPENCLAW_SKILL_LIST=code-review,security-scan,docs-generate
```

Remove the `skills:` block from `config.yaml` to avoid confusion while this issue is unresolved.

### Tracking

- Filed: 2025-04-10
- Ticket: OWC-258

---

## Issue 5 — `anthropic:` / `claude:` model strings silently routed to OpenAI

**Severity:** High
**First introduced:** v0.2.0 (initial routing table)
**Components affected:** `adapter.py`

### Symptom

A workspace booted with the wheel-default model string `anthropic:claude-opus-4-7` (or any `anthropic:<id>` / `claude:<id>` configured by langchain/crewai consumers) reaches `running` status, the gateway becomes healthy, and the A2A handshake succeeds — but every subsequent inference call returns 401/404 from `api.openai.com`. The workspace looks fully online while being structurally unable to answer any question.

### Root cause

OpenClaw is OpenAI-compatible only — `--custom-compatibility` is hard-set to `openai`. The pre-fix prefix-routing table in `adapter.py` mapped every prefix it didn't recognise (including `anthropic` and `claude`) to `OPENAI_API_KEY` + `https://api.openai.com/v1`, then forwarded the bare model id (`claude-opus-4-7`) to the OpenAI endpoint. OpenAI doesn't host Claude models, so every call failed — but the failure surfaced as a per-message 401/404 instead of a boot-time error, so monitoring-by-status missed it.

### Fix (PR landing on `fix/anthropic-prefix-route-via-openrouter`)

`_resolve_provider_routing` (extracted as a pure helper in `adapter.py`) now detects `anthropic` / `claude` prefixes and re-routes through OpenRouter, which exposes Claude under the OpenAI-compat API at the slash-form id `anthropic/<id>`. The helper is exercised by `tests/test_model_routing.py` across all twelve routing branches. Per-prefix API-key lookup also lands so `groq:` / `openrouter:` / `qianfan:` use their proper env vars instead of the legacy fallthrough to `OPENAI_API_KEY`.

### Current workaround (pre-fix workspaces)

If you are pinned to a pre-fix template tag, set both env vars on the workspace:

```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENCLAW_MODEL=openrouter:anthropic/claude-opus-4-7
```

The explicit `openrouter:` prefix avoids the broken anthropic-fallthrough path and matches the slash-form id OpenRouter expects.

### Tracking

- Filed: 2026-05-01
- Ticket: OWC-272

---
