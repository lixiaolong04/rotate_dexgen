#!/usr/bin/env bash
set -euo pipefail

# Collect full-DR expert rollouts carrying the per-frame 9-D `priv_info` RMA label.
# Usage: ./dexgen/pipelines/collect_rma_privileged_dataset.sh [cw|ccw] [first_shard] [last_shard]

mode="${1:?mode must be cw or ccw}"
first_shard="${2:-0}"
last_shard="${3:-4}"

case "${mode}" in
  cw)
    checkpoint="ckpts/overnight_pose5/student_9402.pth"
    seed_base=51000
    ;;
  ccw)
    checkpoint="ckpts/overnight_ccw/student_9503.pth"
    seed_base=52000
    ;;
  *) printf 'mode must be cw or ccw\n' >&2; exit 2 ;;
esac

dexgen_python_bin="${DEXGEN_PYTHON_BIN:-/home/ray/anaconda3/envs/env_isaaclab/bin/python}"
dexgen_device="${DEXGEN_DEVICE:-cuda:0}"
dexgen_num_envs="${DEXGEN_NUM_ENVS:-100}"
dexgen_episodes_per_env="${DEXGEN_EPISODES_PER_ENV:-5}"
dexgen_output_dir="${DEXGEN_RMA_OUTPUT_DIR:-dexgen/data/rma_privileged_v1}"
dexgen_log_dir="${DEXGEN_RMA_LOG_DIR:-dexgen/logs/rma_privileged_v1}"
mkdir -p "${dexgen_output_dir}" "${dexgen_log_dir}"

for ((shard=first_shard; shard<=last_shard; shard++)); do
  tag="$(printf '%02d' "${shard}")"
  output="${dexgen_output_dir}/${mode}_rma_${tag}.pt"
  progress="${dexgen_output_dir}/${mode}_rma_${tag}.progress.json"
  log="${dexgen_log_dir}/${mode}_rma_${tag}.log"
  seed=$((seed_base + shard))
  if [[ -e "${output}" ]]; then
    printf '[rma-data] existing shard: %s\n' "${output}"
    continue
  fi
  printf '[rma-data] mode=%s shard=%s seed=%s device=%s\n' \
    "${mode}" "${tag}" "${seed}" "${dexgen_device}"
  "${dexgen_python_bin}" dexgen/commands/data/collect_rl_data.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
    --num_envs "${dexgen_num_envs}" \
    --episodes_per_env "${dexgen_episodes_per_env}" \
    --algorithm ProprioAdapt \
    --load_path "${checkpoint}" \
    --output "${output}" \
    --progress_file "${progress}" \
    --mode "${mode}" \
    --seed "${seed}" \
    --sensor_profile raw \
    --domain_randomization full \
    --so3 \
    --recovery_perturbations \
    --headless \
    --device "${dexgen_device}" 2>&1 | tee "${log}"
done
