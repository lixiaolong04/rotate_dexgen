#!/usr/bin/env bash
set -euo pipefail

cd /root/rotate_dexgen
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

profile="${1:?usage: run_contact_risk_v9c_local.sh <full5|mask14> <cuda-device>}"
device="${2:?usage: run_contact_risk_v9c_local.sh <full5|mask14> <cuda-device>}"
case "${profile}" in
  full5) mask_args=() ;;
  mask14) mask_args=(--disable_tactile_ids 1 4) ;;
  *) echo "unknown tactile profile: ${profile}" >&2; exit 2 ;;
esac

while pgrep -f "run_contact_risk_v9b_dagger.sh ${profile} " >/dev/null; do
  sleep 15
done

base="dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt"
data_files=(
  dexgen/data/rma_privileged_v1/cw_rma_*.pt
  dexgen/data/rma_privileged_v1/ccw_rma_*.pt
  dexgen/data/dagger_recovery_v4/cw_controlled_*.pt
  dexgen/data/dagger_recovery_v4/ccw_controlled_*.pt
  dexgen/data/dagger_recovery_v4/cw_natural_*.pt
  dexgen/data/dagger_recovery_v4/ccw_natural_*.pt
)
log_dir="dexgen/logs/contact_risk_v9c_local_${profile}"
teacher="dexgen/checkpoints/contact_risk_v9c_local_${profile}_teacher_best.pt"
student="dexgen/checkpoints/contact_risk_v9c_local_${profile}_student_best.pt"
mkdir -p "${log_dir}"

common=(
  --data "${data_files[@]}"
  --base_diffusion "${base}"
  --history_steps 30 --history_stride 1
  --prediction_steps 2 --future_stride 2 --sample_stride 1
  --risk_horizon_seconds 0.5 --object_dynamics_only
  --recovery_label_mode local
  "${mask_args[@]}"
  --latent_dim 16 --batch_size 4096
  --learning_rate 3e-4 --num_workers 24 --prefetch_factor 4
  --log_every 100 --device "${device}"
)

if ! test -f "${log_dir}/TEACHER_ACCEPTED"; then
  python dexgen/commands/train/train_contact_risk.py \
    --stage teacher "${common[@]}" \
    --output "${teacher}" --epochs 8 --early_stopping_patience 2 \
    2>&1 | tee "${log_dir}/teacher.log"
  touch "${log_dir}/TEACHER_COMPLETE"

  python - "${teacher}" <<'PY'
import sys
import torch

metrics = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["metrics"]
print(
    f"[teacher-gate] recovery_auc={metrics['teacher_recovery_auc']:.6f} "
    f"overall_auc={metrics['teacher_auc']:.6f}"
)
if metrics["teacher_recovery_auc"] < 0.65:
    raise SystemExit("local-risk teacher recovery-AUC gate failed")
PY
  touch "${log_dir}/TEACHER_ACCEPTED"
fi

python dexgen/commands/train/train_contact_risk.py \
  --stage student "${common[@]}" \
  --teacher_checkpoint "${teacher}" \
  --output "${student}" --epochs 12 --early_stopping_patience 3 \
  2>&1 | tee "${log_dir}/student.log"
touch "${log_dir}/STUDENT_COMPLETE"
touch "${log_dir}/TRAINING_COMPLETE"
