#!/usr/bin/env bash
set -euo pipefail

pipeline_script="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$(dirname "${pipeline_script}")/../.."
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

variant="${DEXGEN_VARIANT:-rma_v6}"
log_dir="${DEXGEN_VALIDATION_LOG_DIR:-dexgen/logs/rma_v6_validation}"
mkdir -p "${log_dir}"
diffusion="${DEXGEN_DIFFUSION:-dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_rma_init_v6_best.pt}"
idm="${DEXGEN_IDM:-dexgen/checkpoints/idm_modes_full_dr_active_recovery_rma_init_v6_best.pt}"
diffusion_log="${DEXGEN_DIFFUSION_LOG:-dexgen/logs/rma_privileged_v1/train_diffusion_rma_v6.log}"
idm_log="${DEXGEN_IDM_LOG:-dexgen/logs/rma_privileged_v1/train_idm_rma_v6.log}"

while ! grep -q '\[diffusion\] best val=' "${diffusion_log}" 2>/dev/null ||
      ! grep -q '\[idm\] best normalized' "${idm_log}" 2>/dev/null; do
  sleep 30
done
test -s "${diffusion}"
test -s "${idm}"
touch "${log_dir}/TRAINING_COMPLETE"

python -m pytest tests/test_dexgen_pipeline.py -q \
  > "${log_dir}/pytest.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python dexgen/commands/eval/eval_offline.py \
  --data \
    dexgen/data/large_modes_v5/cw_full_*.pt \
    dexgen/data/large_modes_v5/ccw_full_*.pt \
    dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
    dexgen/data/recovery_modes_v6/ccw_recovery_*.pt \
  --diffusion "${diffusion}" --idm "${idm}" \
  --split val --batch_size 512 --ddim_steps 8 --max_samples 100000 \
  --num_workers 8 --device cuda:0 > "${log_dir}/offline.log" 2>&1
touch "${log_dir}/OFFLINE_COMPLETE"

evaluate_direction() {
  local mode="$1"
  local visible_gpu="$2"
  local seed
  for seed in 123 456 789; do
    CUDA_VISIBLE_DEVICES="${visible_gpu}" python dexgen/commands/eval/eval_dexgen.py \
      --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
      --num_envs 20 --mode "${mode}" \
      --diffusion "${diffusion}" --idm "${idm}" \
      --steps 2400 --ddim_steps 8 --so3 --recovery_perturbations \
      --sensor_profile raw --device cuda:0 --headless --seed "${seed}" \
      > "${log_dir}/${variant}_${mode}_seed${seed}.log" 2>&1
  done
  touch "${log_dir}/${mode^^}_COMPLETE"
}

evaluate_direction cw 0 &
cw_pid=$!
evaluate_direction ccw 1 &
ccw_pid=$!
wait "${cw_pid}"
wait "${ccw_pid}"
touch "${log_dir}/PIPELINE_COMPLETE"
