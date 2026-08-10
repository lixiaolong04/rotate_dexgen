#!/usr/bin/env bash
set -euo pipefail

# Deploy the current UniLab V1 mask student snapshots on the real Xhand.
# Usage:
#   bash scripts/deploy_unilab_v1_student.sh 8103
#   bash scripts/deploy_unilab_v1_student.sh 8102 --log v1_8102_real.npz
#
# Seeds available in ckpts/unilab_v1_students_final_20260630:
#   8103 is the current distillation-reward leader, then 8102/8100, then 8101.

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "xhand_deploy" ]]; then
  # Non-interactive shells do not load conda automatically.
  # Keep this best-effort so the script still works inside a prepared Docker/env.
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate xhand_deploy
  fi
fi

seed="${1:-8103}"
if [[ $# -gt 0 ]]; then
  shift
fi

ckpt="ckpts/unilab_v1_students_final_20260630/student_${seed}_best.pth"
if [[ ! -f "$ckpt" ]]; then
  echo "missing checkpoint: $ckpt" >&2
  exit 2
fi

python rl_isaaclab/deploy_xhand/deploy_xhand.py \
  --load_path "$ckpt" \
  --mode 4 --kp 10 --kd 0.3 --speed 0 \
  --action_scale_mul 1.0 \
  --min-fingers 2 \
  --disable-tactile-ids 1,4 \
  "$@"
