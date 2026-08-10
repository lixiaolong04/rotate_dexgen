#!/usr/bin/env bash
set -euo pipefail

# Resumable active-recovery rollout collection.
# Usage: ./dexgen/pipelines/collect_recovery_dataset.sh [cw|ccw|all] [first_shard] [last_shard]

mode_request="${1:-all}"
first_shard="${2:-0}"
last_shard="${3:-9}"

case "${mode_request}" in
  cw|ccw|all) ;;
  *) printf 'mode must be cw, ccw, or all\n' >&2; exit 2 ;;
esac

dexgen_python_bin="${DEXGEN_PYTHON_BIN:-/home/user/miniconda3/envs/env_isaaclab/bin/python}"
dexgen_device="${DEXGEN_DEVICE:-cuda:0}"
dexgen_num_envs="${DEXGEN_NUM_ENVS:-50}"
dexgen_episodes_per_env="${DEXGEN_EPISODES_PER_ENV:-20}"
dexgen_output_dir="${DEXGEN_RECOVERY_OUTPUT_DIR:-dexgen/data/recovery_modes_v6}"
mkdir -p "${dexgen_output_dir}"

collect_mode() {
  local mode="$1"
  local checkpoint
  local seed_base
  if [[ "${mode}" == "cw" ]]; then
    checkpoint="ckpts/overnight_pose5/student_9402.pth"
    seed_base=10000
  else
    checkpoint="ckpts/overnight_ccw/student_9503.pth"
    seed_base=20000
  fi

  local shard
  for ((shard=first_shard; shard<=last_shard; shard++)); do
    local tag
    local output
    local seed
    tag="$(printf '%02d' "${shard}")"
    output="${dexgen_output_dir}/${mode}_recovery_${tag}.pt"
    local progress_file="${dexgen_output_dir}/${mode}_recovery_${tag}.progress.json"
    seed=$((seed_base + shard))
    if [[ -e "${output}" ]]; then
      printf '[recovery-dataset] audit existing %s\n' "${output}"
      "${dexgen_python_bin}" dexgen/commands/data/audit_recovery_data.py --data "${output}"
      continue
    fi
    printf '[recovery-dataset] mode=%s shard=%s seed=%s output=%s\n' \
      "${mode}" "${tag}" "${seed}" "${output}"
    "${dexgen_python_bin}" dexgen/commands/data/collect_rl_data.py \
      --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
      --num_envs "${dexgen_num_envs}" \
      --episodes_per_env "${dexgen_episodes_per_env}" \
      --algorithm ProprioAdapt \
      --load_path "${checkpoint}" \
      --output "${output}" \
      --progress_file "${progress_file}" \
      --mode "${mode}" \
      --seed "${seed}" \
      --sensor_profile raw \
      --domain_randomization full \
      --so3 \
      --recovery_perturbations \
      --headless \
      --device "${dexgen_device}"
    "${dexgen_python_bin}" dexgen/commands/data/audit_recovery_data.py --data "${output}"
  done
}

if [[ "${mode_request}" == "all" ]]; then
  collect_mode cw
  collect_mode ccw
else
  collect_mode "${mode_request}"
fi
