#!/usr/bin/env bash
set -euo pipefail

# Train the 83-D v2b IDM first, then the Diffusion model, from the audited
# full-DR + v3 recovery + v4 intervention-DAgger mixture.

dexgen_python_bin="${DEXGEN_PYTHON_BIN:-/home/ray/anaconda3/envs/env_isaaclab/bin/python}"
dexgen_device="${DEXGEN_DEVICE:-cuda:0}"
log_dir="${DEXGEN_DAGGER_LOG_DIR:-dexgen/logs/dagger_v4}"
mkdir -p "${log_dir}"

shopt -s nullglob
steady=(dexgen/data/large_modes_v5/cw_full_*.pt dexgen/data/large_modes_v5/ccw_full_*.pt)
recovery=(dexgen/data/recovery_modes_v6/cw_recovery_*.pt dexgen/data/recovery_modes_v6/ccw_recovery_*.pt)
dagger=(dexgen/data/dagger_recovery_v4/cw_*.pt dexgen/data/dagger_recovery_v4/ccw_*.pt)
if (( ${#steady[@]} != 20 || ${#recovery[@]} != 20 || ${#dagger[@]} != 20 )); then
  printf 'expected 20 steady, 20 v3 recovery and 20 v4 DAgger shards; got %d/%d/%d\n' \
    "${#steady[@]}" "${#recovery[@]}" "${#dagger[@]}" >&2
  exit 1
fi
all_data=("${steady[@]}" "${recovery[@]}" "${dagger[@]}")

"${dexgen_python_bin}" dexgen/commands/data/audit_recovery_data.py --data "${dagger[@]}" \
  | tee "${log_dir}/formal_dagger_audit.log"

"${dexgen_python_bin}" dexgen/commands/train/train_idm.py \
  --data "${all_data[@]}" \
  --include_tactile \
  --tactile_arch v2b \
  --explicit_recovery_only \
  --recovery_fraction 0.35 \
  --recovery_type_weights 0.1 0.5 0.1 0.3 \
  --tactile_aux_weight 0.005 \
  --batch_size 512 \
  --epochs 15 \
  --num_workers 16 \
  --output dexgen/checkpoints/idm_modes_full_dr_dagger_v4_best.pt \
  --latest_output dexgen/checkpoints/idm_modes_full_dr_dagger_v4_latest.pt \
  --device "${dexgen_device}" \
  2>&1 | tee "${log_dir}/train_idm_v4.log"

"${dexgen_python_bin}" dexgen/commands/eval/eval_offline.py \
  --data "${dagger[@]}" \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_dagger_v4_best.pt \
  --max_samples 100000 \
  --ddim_steps 8 \
  --device "${dexgen_device}" \
  2>&1 | tee "${log_dir}/offline_idm_v4_with_v3_diffusion.log"

"${dexgen_python_bin}" dexgen/commands/train/train_diffusion.py \
  --data "${all_data[@]}" \
  --include_tactile \
  --tactile_arch v2b \
  --explicit_recovery_only \
  --recovery_fraction 0.35 \
  --recovery_type_weights 0.1 0.5 0.1 0.3 \
  --tactile_aux_weight 0.005 \
  --prediction_type epsilon \
  --noise_schedule cosine \
  --batch_size 512 \
  --epochs 15 \
  --num_workers 16 \
  --output dexgen/checkpoints/diffusion_modes_full_dr_dagger_v4_v2b_best.pt \
  --latest_output dexgen/checkpoints/diffusion_modes_full_dr_dagger_v4_v2b_latest.pt \
  --device "${dexgen_device}" \
  2>&1 | tee "${log_dir}/train_diffusion_v4.log"

"${dexgen_python_bin}" dexgen/commands/eval/eval_offline.py \
  --data "${dagger[@]}" \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_dagger_v4_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_dagger_v4_best.pt \
  --max_samples 100000 \
  --ddim_steps 8 \
  --device "${dexgen_device}" \
  2>&1 | tee "${log_dir}/offline_v4.log"
