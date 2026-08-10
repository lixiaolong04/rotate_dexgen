#!/usr/bin/env bash
set -euo pipefail

# Fixed v3/v4 active-recovery comparison for one rotation mode.
# Usage: ./dexgen/pipelines/eval_dagger_v4_compare.sh cw|ccw [cuda:0] [standard|stress]

mode="${1:?mode must be cw or ccw}"
dexgen_device="${2:-cuda:0}"
profile="${3:-standard}"
case "${mode}" in cw|ccw) ;; *) printf 'mode must be cw or ccw\n' >&2; exit 2 ;; esac
case "${profile}" in standard|stress) ;; *) printf 'profile must be standard or stress\n' >&2; exit 2 ;; esac

dexgen_python_bin="${DEXGEN_PYTHON_BIN:-/home/ray/anaconda3/envs/env_isaaclab/bin/python}"
if [[ "${profile}" == "standard" ]]; then
  log_dir="${DEXGEN_DAGGER_LOG_DIR:-dexgen/logs/dagger_v4/compare}"
  profile_args=()
else
  log_dir="${DEXGEN_DAGGER_LOG_DIR:-dexgen/logs/dagger_v4/compare_stress}"
  profile_args=(
    --perturbation_type_weights 0.2 0.6 0.2
    --finger_release_subtype_weights 0.5 0.2 0.2 0.1
  )
fi
mkdir -p "${log_dir}"

for version in v3 v4; do
  if [[ "${version}" == "v3" ]]; then
    diffusion="dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt"
    idm="dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt"
  else
    diffusion="dexgen/checkpoints/diffusion_modes_full_dr_dagger_v4_v2b_best.pt"
    idm="dexgen/checkpoints/idm_modes_full_dr_dagger_v4_best.pt"
  fi
  for seed in 123 456 789; do
    log="${log_dir}/${version}_${mode}_seed${seed}.log"
    "${dexgen_python_bin}" dexgen/commands/eval/eval_dexgen.py \
      --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
      --num_envs 20 \
      --mode "${mode}" \
      --seed "${seed}" \
      --diffusion "${diffusion}" \
      --idm "${idm}" \
      --steps 2400 \
      --ddim_steps 8 \
      --sensor_profile raw \
      --so3 \
      --recovery_perturbations \
      "${profile_args[@]}" \
      --headless \
      --device "${dexgen_device}" \
      2>&1 | tee "${log}"
  done
done
