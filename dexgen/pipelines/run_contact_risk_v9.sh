#!/usr/bin/env bash
set -euo pipefail

cd /root/rotate_dexgen
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

profile="${1:?usage: run_contact_risk_v9.sh <full5|mask14> <cuda-device>}"
device="${2:?usage: run_contact_risk_v9.sh <full5|mask14> <cuda-device>}"
case "${profile}" in
  full5)
    mask_args=()
    ;;
  mask14)
    mask_args=(--disable_tactile_ids 1 4)
    ;;
  *)
    echo "unknown tactile profile: ${profile}" >&2
    exit 2
    ;;
esac

base="dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt"
data_files=(
  dexgen/data/rma_privileged_v1/cw_rma_*.pt
  dexgen/data/rma_privileged_v1/ccw_rma_*.pt
)
log_dir="dexgen/logs/contact_risk_v9_${profile}"
teacher="dexgen/checkpoints/contact_risk_v9_${profile}_teacher_best.pt"
student="dexgen/checkpoints/contact_risk_v9_${profile}_student_best.pt"
mkdir -p "${log_dir}"

python dexgen/commands/train/train_contact_risk.py \
  --stage teacher \
  --data "${data_files[@]}" \
  --base_diffusion "${base}" \
  --output "${teacher}" \
  --history_steps 30 --history_stride 1 \
  --prediction_steps 2 --future_stride 2 --sample_stride 1 \
  --risk_horizon_seconds 1.0 \
  "${mask_args[@]}" \
  --latent_dim 16 --batch_size 1024 --epochs 15 \
  --learning_rate 3e-4 --num_workers 12 --prefetch_factor 4 \
  --log_every 200 --device "${device}" \
  2>&1 | tee "${log_dir}/teacher.log"
touch "${log_dir}/TEACHER_COMPLETE"

python - "${teacher}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
metrics = checkpoint["metrics"]
print(
    f"[teacher-gate] auc={metrics['teacher_auc']:.6f} "
    f"zero_auc={metrics['zero_latent_auc']:.6f} "
    f"uplift={metrics['teacher_auc'] - metrics['zero_latent_auc']:.6f}"
)
if metrics["teacher_auc"] < 0.60:
    raise SystemExit("teacher AUC gate failed")
PY
touch "${log_dir}/TEACHER_ACCEPTED"

python dexgen/commands/train/train_contact_risk.py \
  --stage student \
  --data "${data_files[@]}" \
  --base_diffusion "${base}" \
  --teacher_checkpoint "${teacher}" \
  --output "${student}" \
  --history_steps 30 --history_stride 1 \
  --prediction_steps 2 --future_stride 2 --sample_stride 1 \
  --risk_horizon_seconds 1.0 \
  "${mask_args[@]}" \
  --latent_dim 16 --batch_size 1024 --epochs 20 \
  --learning_rate 3e-4 --num_workers 12 --prefetch_factor 4 \
  --log_every 200 --device "${device}" \
  2>&1 | tee "${log_dir}/student.log"
touch "${log_dir}/STUDENT_COMPLETE"
touch "${log_dir}/TRAINING_COMPLETE"
