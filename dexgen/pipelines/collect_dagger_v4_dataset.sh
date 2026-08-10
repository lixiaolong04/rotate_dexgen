#!/usr/bin/env bash
set -euo pipefail

# Resumable v4 intervention-DAgger collection.
# Usage:
#   ./dexgen/pipelines/collect_dagger_v4_dataset.sh cw|ccw natural|controlled first_shard last_shard

mode="${1:?mode must be cw or ccw}"
kind="${2:?kind must be natural or controlled}"
first_shard="${3:-0}"
last_shard="${4:-4}"

case "${mode}" in
  cw)
    expert="ckpts/overnight_pose5/student_9402.pth"
    seed_base=31000
    ;;
  ccw)
    expert="ckpts/overnight_ccw/student_9503.pth"
    seed_base=41000
    ;;
  *) printf 'mode must be cw or ccw\n' >&2; exit 2 ;;
esac
case "${kind}" in
  natural) seed_base=$((seed_base + 1000)) ;;
  controlled) seed_base=$((seed_base + 2000)) ;;
  *) printf 'kind must be natural or controlled\n' >&2; exit 2 ;;
esac

dexgen_python_bin="${DEXGEN_PYTHON_BIN:-/home/ray/anaconda3/envs/env_isaaclab/bin/python}"
dexgen_device="${DEXGEN_DEVICE:-cuda:0}"
dexgen_num_envs="${DEXGEN_NUM_ENVS:-100}"
dexgen_episodes_per_env="${DEXGEN_EPISODES_PER_ENV:-5}"
dexgen_output_dir="${DEXGEN_DAGGER_OUTPUT_DIR:-dexgen/data/dagger_recovery_v4}"
dexgen_log_dir="${DEXGEN_DAGGER_LOG_DIR:-dexgen/logs/dagger_v4}"
mkdir -p "${dexgen_output_dir}" "${dexgen_log_dir}"

for ((shard=first_shard; shard<=last_shard; shard++)); do
  tag="$(printf '%02d' "${shard}")"
  output="${dexgen_output_dir}/${mode}_${kind}_${tag}.pt"
  progress="${dexgen_output_dir}/${mode}_${kind}_${tag}.progress.json"
  log="${dexgen_log_dir}/${mode}_${kind}_${tag}.log"
  lock="${output}.lock"
  seed=$((seed_base + shard))
  if [[ -e "${output}" ]]; then
    printf '[dagger-v4] audit existing %s\n' "${output}"
    "${dexgen_python_bin}" dexgen/commands/data/audit_recovery_data.py --data "${output}"
    continue
  fi
  if ! mkdir "${lock}" 2>/dev/null; then
    printf '[dagger-v4] waiting for another collector: %s\n' "${output}"
    while [[ -d "${lock}" && ! -e "${output}" ]]; do
      sleep 30
    done
    if [[ -e "${output}" ]]; then
      "${dexgen_python_bin}" dexgen/commands/data/audit_recovery_data.py --data "${output}"
      continue
    fi
    printf 'collector lock disappeared without output: %s\n' "${output}" >&2
    exit 1
  fi

  extra_args=()
  if [[ "${kind}" == "natural" ]]; then
    extra_args+=(--dagger_natural_triggers)
  else
    extra_args+=(
      --recovery_perturbations
      --perturbation_interval_min 2
      --perturbation_interval_max 4
      --perturbation_type_weights 0.2 0.6 0.2
      --finger_release_subtype_weights 0.5 0.2 0.2 0.1
    )
  fi

  printf '[dagger-v4] mode=%s kind=%s shard=%s device=%s\n' \
    "${mode}" "${kind}" "${tag}" "${dexgen_device}"
  "${dexgen_python_bin}" dexgen/commands/data/collect_rl_data.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
    --num_envs "${dexgen_num_envs}" \
    --episodes_per_env "${dexgen_episodes_per_env}" \
    --algorithm ProprioAdapt \
    --load_path "${expert}" \
    --output "${output}" \
    --progress_file "${progress}" \
    --mode "${mode}" \
    --seed "${seed}" \
    --sensor_profile raw \
    --domain_randomization full \
    --so3 \
    --dagger_diffusion \
      dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
    --dagger_idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
    --dagger_ddim_steps 8 \
    --direction_check_seconds 0 \
    --headless \
    --device "${dexgen_device}" \
    "${extra_args[@]}" 2>&1 | tee "${log}"
  "${dexgen_python_bin}" dexgen/commands/data/audit_recovery_data.py --data "${output}" \
    | tee -a "${log}"
  rmdir "${lock}"
done
