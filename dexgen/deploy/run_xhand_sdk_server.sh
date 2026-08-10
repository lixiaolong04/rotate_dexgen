#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

exec docker run -it --rm \
  --network host \
  --privileged \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  xhand-deploy:22.04 \
  python3 -m dexgen.deploy.xhand_sdk_server "$@"
