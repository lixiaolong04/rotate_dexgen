#!/usr/bin/env bash
set -euo pipefail

# Frozen-v3 baseline versus tactile contact-risk candidate reranking.
# Usage:
#   bash dexgen/pipelines/eval_contact_risk_v9c_rerank.sh smoke [candidates] [margin] [IDM-weight]
#   bash dexgen/pipelines/eval_contact_risk_v9c_rerank.sh formal [candidates] [margin] [IDM-weight]

cd /root/rotate_dexgen

profile="${1:-formal}"
risk_candidates="${2:-4}"
risk_margin="${3:-0.25}"
idm_consistency_weight="${4:-0}"
case "${profile}" in
  smoke)
    steps=800
    seeds=(123)
    ;;
  formal)
    steps=2400
    seeds=(123 456 789)
    ;;
  *)
    printf 'profile must be smoke or formal\n' >&2
    exit 2
    ;;
esac

python_bin="${DEXGEN_PYTHON_BIN:-/home/ray/anaconda3/envs/env_isaaclab/bin/python}"
diffusion="dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt"
idm="dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt"
risk="dexgen/checkpoints/contact_risk_v9c_local_full5_student_best.pt"
log_dir="dexgen/logs/contact_risk_v9c_rerank_${profile}_c${risk_candidates}_m${risk_margin}_i${idm_consistency_weight}"
mkdir -p "${log_dir}"

evaluate_direction() {
  local mode="$1"
  local visible_gpu="$2"
  local version seed
  for version in baseline rerank; do
    for seed in "${seeds[@]}"; do
      extra_args=()
      if [[ "${version}" == "rerank" ]]; then
        extra_args=(
          --contact_risk "${risk}"
          --risk_candidates "${risk_candidates}"
          --risk_min_logit_gain "${risk_margin}"
          --risk_idm_consistency_weight "${idm_consistency_weight}"
        )
      fi
      CUDA_VISIBLE_DEVICES="${visible_gpu}" "${python_bin}" \
        dexgen/commands/eval/eval_dexgen.py \
        --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
        --num_envs 20 \
        --mode "${mode}" \
        --seed "${seed}" \
        --diffusion "${diffusion}" \
        --idm "${idm}" \
        --steps "${steps}" \
        --ddim_steps 8 \
        --sensor_profile raw \
        --so3 \
        --recovery_perturbations \
        --headless \
        --device cuda:0 \
        "${extra_args[@]}" \
        > "${log_dir}/${version}_${mode}_seed${seed}.log" 2>&1
    done
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
