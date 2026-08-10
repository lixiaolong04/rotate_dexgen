#!/usr/bin/env bash
set -euo pipefail

cd /root/rotate_dexgen
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

log_dir="dexgen/logs/teacher_student_v8_h4"
checkpoint_dir="dexgen/checkpoints"
data_dir="dexgen/data/rma_privileged_v1"
data_files=("${data_dir}"/cw_rma_*.pt "${data_dir}"/ccw_rma_*.pt)
adapter="${checkpoint_dir}/teacher_student_v8_h4_best.pt"
diffusion="${checkpoint_dir}/diffusion_modes_teacher_student_v8_h4_best.pt"
idm="${checkpoint_dir}/idm_modes_teacher_student_v8_h4_best.pt"

mkdir -p "${log_dir}" "${checkpoint_dir}"

python dexgen/commands/train/train_v8_adapter.py \
  --data "${data_files[@]}" \
  --output "${adapter}" \
  --latest_output "${checkpoint_dir}/teacher_student_v8_h4_latest.pt" \
  --priv_indices 3 4 8 \
  --history_steps 4 \
  --history_stride 2 \
  --sample_stride 2 \
  --latent_dim 8 \
  --batch_size 512 \
  --epochs 20 \
  --learning_rate 3e-4 \
  --normalizer_samples 200000 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --log_every 200 \
  --device cuda:0 \
  2>&1 | tee "${log_dir}/adapter.log"

touch "${log_dir}/ADAPTER_COMPLETE"

python dexgen/commands/train/train_v8_policy.py \
  --component diffusion \
  --data "${data_files[@]}" \
  --adapter "${adapter}" \
  --output "${diffusion}" \
  --latest_output "${checkpoint_dir}/diffusion_modes_teacher_student_v8_h4_latest.pt" \
  --history_steps 4 \
  --history_stride 2 \
  --sample_stride 2 \
  --prediction_steps 2 \
  --future_stride 2 \
  --recovery_fraction 0.25 \
  --batch_size 512 \
  --epochs 15 \
  --learning_rate 1e-4 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --log_every 200 \
  --device cuda:0 \
  2>&1 | tee "${log_dir}/diffusion.log" &
diffusion_pid=$!

python dexgen/commands/train/train_v8_policy.py \
  --component idm \
  --data "${data_files[@]}" \
  --adapter "${adapter}" \
  --output "${idm}" \
  --latest_output "${checkpoint_dir}/idm_modes_teacher_student_v8_h4_latest.pt" \
  --history_steps 4 \
  --history_stride 2 \
  --sample_stride 2 \
  --prediction_steps 2 \
  --future_stride 2 \
  --recovery_fraction 0.25 \
  --batch_size 512 \
  --epochs 15 \
  --learning_rate 3e-4 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --log_every 200 \
  --device cuda:1 \
  2>&1 | tee "${log_dir}/idm.log" &
idm_pid=$!

wait "${diffusion_pid}"
touch "${log_dir}/DIFFUSION_COMPLETE"
wait "${idm_pid}"
touch "${log_dir}/IDM_COMPLETE"
touch "${log_dir}/TRAINING_COMPLETE"
