#!/usr/bin/env bash
set -euo pipefail

cd /root/rotate_dexgen
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

train_log_dir="dexgen/logs/teacher_student_v8b_mixed"
log_dir="dexgen/logs/teacher_student_v8b_validation"
diffusion="dexgen/checkpoints/diffusion_modes_teacher_student_v8b_mixed_best.pt"
idm="dexgen/checkpoints/idm_modes_teacher_student_v8b_mixed_best.pt"
mkdir -p "${log_dir}"

while ! test -f "${train_log_dir}/TRAINING_COMPLETE"; do
  sleep 30
done
test -s "${diffusion}"
test -s "${idm}"
touch "${log_dir}/TRAINING_COMPLETE"

python -m pytest tests/test_dexgen_pipeline.py -q \
  > "${log_dir}/pytest.log" 2>&1
touch "${log_dir}/PYTEST_COMPLETE"

student_data=(
  dexgen/data/large_modes_v5/cw_full_*.pt
  dexgen/data/large_modes_v5/ccw_full_*.pt
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt
)
rma_data=(
  dexgen/data/rma_privileged_v1/cw_rma_*.pt
  dexgen/data/rma_privileged_v1/ccw_rma_*.pt
)

CUDA_VISIBLE_DEVICES=0 python dexgen/commands/eval/eval_offline.py \
  --data "${student_data[@]}" \
  --diffusion "${diffusion}" --idm "${idm}" \
  --latent_source student --split val --batch_size 1024 --ddim_steps 8 \
  --max_samples 100000 --num_workers 8 --device cuda:0 \
  > "${log_dir}/offline_100k_student_standard.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python dexgen/commands/eval/eval_offline.py \
  --data "${rma_data[@]}" \
  --diffusion "${diffusion}" --idm "${idm}" \
  --latent_source student --split val --batch_size 1024 --ddim_steps 8 \
  --max_samples 100000 --num_workers 8 --device cuda:0 \
  > "${log_dir}/offline_rma_student.log" 2>&1 &
student_rma_pid=$!

CUDA_VISIBLE_DEVICES=1 python dexgen/commands/eval/eval_offline.py \
  --data "${rma_data[@]}" \
  --diffusion "${diffusion}" --idm "${idm}" \
  --latent_source teacher --split val --batch_size 1024 --ddim_steps 8 \
  --max_samples 100000 --num_workers 8 --device cuda:0 \
  > "${log_dir}/offline_rma_teacher.log" 2>&1 &
teacher_rma_pid=$!

wait "${student_rma_pid}"
wait "${teacher_rma_pid}"
touch "${log_dir}/OFFLINE_COMPLETE"

if ! python -c '
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
motion = float(re.search(r"motion MAE:\s+([0-9.]+) mm", text).group(1))
target = float(re.search(r"Diffusion -> IDM\s+MAE=([0-9.]+)", text).group(1))
print(f"[v8b-gate] motion_mae_mm={motion:.4f} target_mae_rad={target:.5f}")
raise SystemExit(0 if motion <= 1.00 and target <= 0.018 else 1)
' "${log_dir}/offline_100k_student_standard.log" \
  > "${log_dir}/offline_gate.log" 2>&1; then
  touch "${log_dir}/OFFLINE_REJECTED"
  exit 0
fi
touch "${log_dir}/OFFLINE_ACCEPTED"

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
      > "${log_dir}/v8b_${mode}_seed${seed}.log" 2>&1
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
