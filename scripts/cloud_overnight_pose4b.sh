#!/bin/bash
# pose4b 重训 (seed 9300-9303, 2026-06-24, 中指弯版 + scale [0.4,0.7] Ø32-56 去掉抓不住的最小档):
#   stop old -> gen_grasp(_0.4-0.7-10, pose4) -> teacher 1000M -> student 1000M -> eval
# Launch: cd /root/era_dexhand_lab && setsid bash scripts/cloud_overnight_pose4b.sh >logs/overnight_pose4b_boot.log 2>&1 </dev/null &
# Monitor: tail -f logs/overnight_pose4b.log
set -u
cd /root/era_dexhand_lab || exit 1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cloud_lib.sh"
cloud_activate_conda
M=logs/overnight_pose4b.log
echo "[p4b $(date)] ===== START pose4b seed 9300-9303 (scale 0.4-0.7) =====" > "$M"

# ---- 1. stop old ----
pkill -9 -f "[t]rain.py.*Isaac-Inhand" 2>/dev/null || true
sleep 8
cloud_kill_gen_grasp 2
echo "[p4b] stopped: train残留=$(pgrep -fc '[t]rain.py.*Isaac-Inhand') gg残留=$(cloud_count_gen_grasp)" >> "$M"

# ---- 2. gen_grasp -> pose4b cache (scale 0.4-0.7) ----
CACHE=cache/xhand_grasp_linspace_0.4-0.7-10.npy
rm -f "$CACHE"
echo "[p4b $(date)] gen_grasp START -> $CACHE" >> "$M"
CUDA_VISIBLE_DEVICES=0 setsid python -u rl_isaaclab/scripts/gen_grasp.py \
  --task Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0 --num_envs 20480 --headless > logs/overnight_pose4b_gen_grasp.log 2>&1 &
for k in $(seq 1 80); do
  sleep 30
  if [ -f "$CACHE" ]; then s1=$(stat -c %s "$CACHE" 2>/dev/null); sleep 10; s2=$(stat -c %s "$CACHE" 2>/dev/null)
    if [ "${s1:-0}" = "${s2:-1}" ] && [ "${s1:-0}" -gt 10000 ]; then break; fi; fi
done
cloud_kill_gen_grasp 3
if [ ! -f "$CACHE" ]; then echo "[p4b $(date)] FATAL: cache MISSING. ABORT." >> "$M"; exit 1; fi
echo "[p4b $(date)] gen_grasp DONE: $(stat -c '%s bytes' "$CACHE")" >> "$M"

# ---- 3. teacher 9300-3 (1000M) ----
for i in 0 1 2 3; do
  SEED=$((9300+i))
  CUDA_VISIBLE_DEVICES=$i nohup python -u rl_isaaclab/scripts/train.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 20480 \
    --max_agent_steps 1000000000 --seed "$SEED" > logs/diag_teacher_${SEED}.log 2>&1 &
  echo "[p4b] teacher seed=$SEED gpu=$i pid=$!" >> "$M"; sleep 5
done
echo "[p4b $(date)] 4 TEACHERS LAUNCHED (1000M ~10h), waiting..." >> "$M"
wait
echo "[p4b $(date)] ===== ALL TEACHERS DONE =====" >> "$M"

# ---- 4. student 9300-3 (1000M) ----
for i in 0 1 2 3; do
  SEED=$((9300+i))
  dir=$(grep -aoE "logs/debug/2026[0-9_-]+" "logs/diag_teacher_${SEED}.log" 2>/dev/null | head -1)
  best="${dir}/stage1_nn/best.pth"
  if [ -z "$dir" ] || [ ! -f "$best" ]; then echo "[p4b] SKIP student $SEED: best MISSING (dir=$dir)" >> "$M"; continue; fi
  CUDA_VISIBLE_DEVICES=$i nohup python -u rl_isaaclab/scripts/train.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 20480 \
    --max_agent_steps 1000000000 --algorithm ProprioAdapt --load_path "$best" --seed "$SEED" \
    > logs/student_${SEED}.log 2>&1 &
  echo "[p4b] student seed=$SEED gpu=$i pid=$! <- $best" >> "$M"; sleep 5
done
echo "[p4b $(date)] STUDENTS LAUNCHED (1000M ~10h), waiting..." >> "$M"
wait
echo "[p4b $(date)] ===== ALL STUDENTS DONE =====" >> "$M"

# ---- 5. collect + eval ----
mkdir -p ckpts/overnight_pose4b
for i in 0 1 2 3; do
  SEED=$((9300+i))
  dir=$(grep -aoE "logs/debug/2026[0-9_-]+" "logs/diag_teacher_${SEED}.log" 2>/dev/null | head -1)
  cp "${dir}/stage2_nn/best.pth" "ckpts/overnight_pose4b/student_${SEED}.pth" 2>/dev/null \
    && echo "[p4b] collected student_${SEED}.pth" >> "$M" || echo "[p4b] WARN student_${SEED} missing" >> "$M"
done
echo "[p4b $(date)] eval START" >> "$M"
CUDA_VISIBLE_DEVICES=0 python -u rl_isaaclab/scripts/eval_batch.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 200 \
  --algorithm ProprioAdapt --so3 --ckpt_glob 'ckpts/overnight_pose4b/student_*.pth' \
  > logs/overnight_pose4b_eval.log 2>&1
echo "[p4b $(date)] ===== eval DONE -- PIPELINE COMPLETE =====" >> "$M"
