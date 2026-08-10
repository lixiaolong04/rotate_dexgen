#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

exec python -m dexgen.deploy.manus_auto_real "$@"
