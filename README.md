# template-openclaw

Molecule AI workspace template for the **openclaw** runtime.

## Usage

### In Molecule AI canvas
Select this template when creating a new workspace — it appears in the template picker automatically.

### From a URL (community install)
Paste this URL when creating a workspace:
```
github://Molecule-AI/template-openclaw
```

## Files
- `config.yaml` — workspace configuration (runtime, model, skills, etc.)
- `system-prompt.md` — agent system prompt (if present)

## Schema version
`template_schema_version: 1` — compatible with Molecule AI platform v1.x.

## API keys / model routing
For `anthropic:` or `claude:` model prefixes, set `OPENROUTER_API_KEY` — OpenClaw is OpenAI-compatible only and does not speak the Anthropic API natively, so Claude models are routed through OpenRouter (which exposes them under the OpenAI-compat surface). See `known-issues.md` Issue 5 for the routing table and pre-fix workaround.

## License
Business Source License 1.1 — © Molecule AI.
