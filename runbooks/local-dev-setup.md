# Local Development Setup — openclaw

This runbook covers cloning, installing dependencies, building the Docker image, and resolving common local development issues for the openclaw workspace template.

## Prerequisites

- Python 3.11+
- Docker 24.0+
- `git`
- A Molecule platform account with API key, workspace ID, and a task ID for testing

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/molecule-ai-workspace-template-openclaw.git
cd molecule-ai-workspace-template-openclaw
```

If working on a fork:

```bash
git remote add upstream https://github.com/your-org/molecule-ai-workspace-template-openclaw.git
```

## 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For linting and type checking:

```bash
pip install ruff mypy pytest
```

## 3. Docker Build

Build the image locally and verify the entrypoint is set correctly:

```bash
docker build -t openclaw-local:latest .
docker run --rm openclaw-local:latest --version
# Expected: prints the version from __init__.py
```

Always rebuild without cache when syncing from upstream to avoid stale layers:

```bash
docker build --no-cache -t openclaw-local:latest .
```

## 4. Configure Environment for Development

Create a `.env` file in the repo root (never commit this file):

```
MOLECULE_PLATFORM_URL=https://platform.molecule.ai
MOLECULE_WORKSPACE_ID=ws-dev-local
MOLECULE_API_KEY=your-dev-key-here
MOLECULE_TASK_ID=task-local-dev-001
OPENCLAW_MODEL=claude-sonnet-4-20250514
OPENCLAW_TIMEOUT_SEC=60
OPENCLAW_TEMPERATURE=0.7
OPENCLAW_MAX_TOKENS=4096
MOLECULE_SKILLS_DIR=/workspace/skills
OPENCLAW_SKILL_LIST=code-review,security-scan
OPENCLAW_FORWARD_CONTEXT=1
OPENCLAW_NO_RESUME_REINJECT=1
```

Override environment variables at runtime without editing the file:

```bash
docker run --rm \
  --env-file .env \
  -e OPENCLAW_TIMEOUT_SEC=30 \
  -v "$(pwd)/skills:/workspace/skills:ro" \
  openclaw-local:latest python -m adapter
```

### Dev-only Overrides

| Variable | Dev default | Production default | Purpose |
|----------|-------------|-------------------|---------|
| `OPENCLAW_TIMEOUT_SEC` | `60` | `300` | Shorter timeout for faster dev loops |
| `OPENCLAW_TEMPERATURE` | `0.7` | `0.3` | More random output for exploratory testing |
| `OPENCLAW_FORWARD_CONTEXT` | `1` | `0` | Useful in dev to inspect full context |
| `OPENCLAW_NO_RESUME_REINJECT` | `1` | `0` | Prevents double system-prompt injection during session resume testing |

## 5. Run the Smoke Test

The smoke test runs without a platform connection:

```bash
source .venv/bin/activate
python -m openclaw_smoke_test
# Exit code 0 = pass
```

To test against a mock platform server:

```bash
# Start mock server
python -m mock_platform_server &
MOCK_PORT=$!

# Run adapter integration test
export MOLECULE_PLATFORM_URL=http://localhost:${MOCK_PORT}
export MOLECULE_WORKSPACE_ID=ws-test
export MOLECULE_API_KEY=test-key
export MOLECULE_TASK_ID=task-test-001
pytest tests/integration/test_adapter.py -v
```

## 6. Common Issues

| Symptom | Likely cause | Resolution |
|---------|--------------|------------|
| Agent appears to hang; no output | `requirements.txt` pins old `openclaw-runtime==0.9.3` | Override: `pip install openclaw-runtime==0.11.2` before build |
| No tool call history in platform UI | `OPENCLAW_FORWARD_CONTEXT` not set | Set `OPENCLAW_FORWARD_CONTEXT=1` before running container |
| Doubled system prompt after reconnect | `handle_session_resume()` re-injects prompt unconditionally | Set `OPENCLAW_NO_RESUME_REINJECT=1` as interim workaround |
| Skills from `config.yaml` not loaded | Direct assignment overwrites platform defaults instead of merging | Use `OPENCLAW_SKILL_LIST` env var instead of `config.yaml` skills block |
| `docker build` fails with `pip install` error | Python version older than 3.11 or pip not upgraded | Use Python 3.11+; run `pip install --upgrade pip` first |
| Skills directory not found at runtime | Volume not mounted; `MOLECULE_SKILLS_DIR` points to empty path | Mount: `-v "$(pwd)/skills:/workspace/skills:ro"` |
| Session resume produces confused output | `system-prompt.md` injected twice on resume | See known-issues.md Issue 3; set `OPENCLAW_NO_RESUME_REINJECT=1` |
| Adapter callback returns empty metadata | Context field not forwarded in `build_callback_payload()` | Set `OPENCLAW_FORWARD_CONTEXT=1` or patch `adapter.py` to serialize context |

## 7. Hot-Reloading Skills

Skill prompt files are loaded from `MOLECULE_SKILLS_DIR` at runtime and do not require a container restart. To update a skill:

```bash
# Edit skill file
vim skills/code-review.md

# No restart needed — adapter reloads on next task
```

For immediate reload during development, send `SIGHUP`:

```bash
docker kill --signal HUP <container_id>
```
