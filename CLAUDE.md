# molecule-ai-workspace-template-openclaw

A Molecule AI workspace template for the **openclaw** single-agent runtime. It provisions a self-contained Docker environment for one agent that integrates with the Molecule platform, runs one-off or long-running tasks, and supports skill injection via prompt fragments.

## Purpose

`openclaw` is a lightweight single-agent runtime. Unlike deepagents (which runs an orchestrator plus multiple task agents), openclaw runs a single agent container that receives task assignments from the platform, executes them, and returns results. The template provides:

- A `config.yaml` that specifies the agent's system configuration, model, and skill requirements
- An `adapter.py` that bridges Molecule platform events to the openclaw agent and surfaces results
- A `system-prompt.md` that defines the agent's persona, tool conventions, and Molecule integration behaviour
- A `requirements.txt` pinning runtime dependencies
- A `Dockerfile` that bundles everything into a runnable image

## Key Files and Their Roles

| File | Role |
|------|------|
| `config.yaml` | Declarative agent config: schema version, model, default timeout, skill list, platform integration flags |
| `adapter.py` | Translates Molecule platform task events into agent input; formats agent output for the platform callback |
| `system-prompt.md` | Static system prompt injected at session start; defines agent persona, tool namespace, and escalation path |
| `requirements.txt` | Runtime Python dependencies |
| `Dockerfile` | Single-stage build: installs deps, copies workspace files, runs the agent entrypoint |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MOLECULE_PLATFORM_URL` | Yes | — | Base URL of the Molecule platform API |
| `MOLECULE_WORKSPACE_ID` | Yes | — | Workspace instance identifier |
| `MOLECULE_API_KEY` | Yes | — | API key for platform authentication |
| `MOLECULE_TASK_ID` | Yes | — | Current task identifier |
| `OPENCLAW_MODEL` | No | `claude-sonnet-4-20250514` | Model used by the openclaw agent |
| `OPENCLAW_MAX_TOKENS` | No | `4096` | Maximum output tokens per response |
| `OPENCLAW_TIMEOUT_SEC` | No | `300` | Task execution timeout |
| `OPENCLAW_TEMPERATURE` | No | `0.7` | Sampling temperature |
| `OPENCLAW_SYSTEM_PROMPT_PATH` | No | `/workspace/system-prompt.md` | Path to the system prompt file |
| `MOLECULE_SKILLS_DIR` | No | `/workspace/skills` | Directory containing skill prompt fragments |
| `OPENCLAW_SKILL_LIST` | No | — | Comma-separated list of skill names to activate |
| `OPENCLAW_TOOLS_ENABLED` | No | `true` | Enable tool use for this agent session |

## Skill Loading

Skills are loaded by `adapter.py` at session startup. The directory specified by `MOLECULE_SKILLS_DIR` is scanned for `.md` files matching the names listed in `OPENCLAW_SKILL_LIST`. Each matching file is read and prepended to the system prompt so the agent has immediate context for its task domain.

If `OPENCLAW_SKILL_LIST` is empty, no skill fragments are loaded.

Skill files are not baked into the image — they are injected at runtime as a volume mount, enabling hot-reload without rebuilding.

## Development Setup

```bash
# 1. Clone the template
git clone https://github.com/your-org/molecule-ai-workspace-template-openclaw.git
cd molecule-ai-workspace-template-openclaw

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set required env vars
export MOLECULE_PLATFORM_URL=https://platform.molecule.ai
export MOLECULE_WORKSPACE_ID=ws-dev-local
export MOLECULE_API_KEY=your-dev-key-here
export MOLECULE_TASK_ID=task-local-test-001
export OPENCLAW_MODEL=claude-sonnet-4-20250514

# 5. Build the Docker image locally
docker build -t openclaw-local:latest .

# 6. Run a smoke test
docker run --rm \
  --env-file .env \
  -v "$(pwd)/skills:/workspace/skills:ro" \
  openclaw-local:latest python -m openclaw_smoke_test
```

## Testing

### Unit tests

```bash
pytest tests/unit/ -v
```

### Integration tests

Integration tests require a running Molecule platform and valid credentials.

```bash
export MOLECULE_PLATFORM_URL=https://platform.molecule.ai
export MOLECULE_WORKSPACE_ID=ws-integration-test
export MOLECULE_API_KEY=your-integration-key
export MOLECULE_TASK_ID=task-integration-test-001
export OPENCLAW_TIMEOUT_SEC=60
pytest tests/integration/ -v
```

### Smoke test (no platform required)

```bash
python -m openclaw_smoke_test
```

This runs `adapter.py` in mock mode, verifies that system prompts are assembled correctly, and checks that the agent produces a valid result structure.

## Release Process

1. **Version bump** — Update `__version__` in `__init__.py` and tag: `git tag -a v0.x.y -m "Release v0.x.y"`.
2. **CI pipeline** — Push the tag to trigger CI: lint (`ruff`), type check (`mypy`), unit tests, Docker build and push.
3. **Changelog** — Add entries to `CHANGELOG.md`.
4. **Notify** — Open a PR against `molecule-ai/workspace-registry` updating the openclaw template entry with the new tag and SHA.

## Repository Structure

```
molecule-ai-workspace-template-openclaw/
├── Dockerfile
├── config.yaml
├── requirements.txt
├── adapter.py
├── __init__.py
├── system-prompt.md
├── skills/               # optional; mounted at runtime
├── CLAUDE.md
├── known-issues.md
└── runbooks/
    └── local-dev-setup.md
```
