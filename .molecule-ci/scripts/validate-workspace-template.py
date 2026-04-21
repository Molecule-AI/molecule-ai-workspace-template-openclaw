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
sv = config.get("template_schema_version")
if sv is None:
    errors.append("Missing template_schema_version (add: template_schema_version: 1)")
if errors:
    for e in errors:
        print(f"::error::{e}")
    sys.exit(1)
print(f"✓ config.yaml valid: {config['name']} (runtime: {config.get('runtime')})")
