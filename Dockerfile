# Pin by digest, not tag — avoids surprise base-bumps. Pairs with the
# Trivy gate in molecule-ci/.github/workflows/publish-template-image.yml
# (PR #35): pin avoids unexpected upgrade, Trivy catches if the pinned
# digest accumulates fixable HIGH/CRITICAL vulns over time. Bump
# deliberately by re-resolving:
#   docker pull --platform linux/amd64 python:3.11-slim
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim
# Last resolved: 2026-05-03 (RFC #388 PR-2b).
FROM python:3.11-slim@sha256:6d85378d88a19cd4d76079817532d62232be95757cb45945a99fec8e8084b9c2

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gosu ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install OpenClaw CLI
RUN npm install -g openclaw 2>/dev/null || true

RUN useradd -u 1000 -m -s /bin/bash agent
WORKDIR /app

# RUNTIME_VERSION is forwarded from molecule-ci's reusable publish
# workflow as a docker build-arg. Cascade-triggered builds set it to
# the exact runtime version PyPI just published. Including it as an
# ARG changes the cache key for the pip install layer below — the
# fix for the cascade cache trap that bit us 5x on 2026-04-27.
ARG RUNTIME_VERSION=

# Bump pip + setuptools + wheel BEFORE installing project deps — the
# python:3.11-slim base ships old transitives (jaraco.context, wheel,
# setuptools) Trivy flags as fixable HIGH CVEs. Bumping here resolves
# them at the metadata layer; subsequent pip installs use the upgraded
# resolvers. molecule-ci#38 Phase-1.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ -n "${RUNTIME_VERSION}" ]; then \
      pip install --no-cache-dir --upgrade "molecule-ai-workspace-runtime==${RUNTIME_VERSION}"; \
    fi

COPY adapter.py .
COPY __init__.py .

ENV ADAPTER_MODULE=adapter

ENTRYPOINT ["molecule-runtime"]
