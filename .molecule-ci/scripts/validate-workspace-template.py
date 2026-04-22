#!/usr/bin/env python3
"""Validate a Molecule AI workspace template repo."""
import os, sys, yaml

errors = []

if not os.path.isfile("config.yaml"):
    print("::error::config.yaml not found at repo root")
    sys.exit(1)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

if not config.get("name"):
    errors.append("Missing required field: name")
if not config.get("runtime"):
    errors.append("Missing required field: runtime")

known = {"langgraph", "claude-code", "crewai", "autogen", "deepagents", "hermes", "gemini-cli", "openclaw"}
runtime = config.get("runtime", "")
if runtime and runtime not in known:
    print(f"::warning::Runtime '{runtime}' is not in the known set. OK for custom runtimes.")

# Check for legacy imports
if os.path.isfile("adapter.py"):
    with open("adapter.py") as f:
        content = f.read()
        if "molecule_runtime" in content:
            print("::warning::adapter.py imports 'molecule_runtime' — legacy import, use 'molecule_ai' or platform SDK")

# Check for missing molecule-ai-workspace-runtime dependency hint
if os.path.isfile("Dockerfile"):
    with open("Dockerfile") as f:
        content = f.read()
        if "molecule-ai-workspace-runtime" not in content:
            print("::warning::Dockerfile does not reference 'molecule-ai-workspace-runtime' — may need base runtime package")

sv = config.get("template_schema_version")
if sv is None:
    errors.append("Missing template_schema_version (add: template_schema_version: 1)")

if errors:
    for e in errors:
        print(f"::error::{e}")
    sys.exit(1)

print(f"✓ config.yaml valid: {config['name']} (runtime: {config.get('runtime')})")
