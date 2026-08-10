#!/bin/bash
# pose5 重训 (seed 9400-9403, 2026-06-24, pose2基础+各指更弯抓紧Ø40, scale [0.4,0.7]):
#   stop old -> gen_grasp(_0.4-0.7-10, pose5) -> teacher 1000M -> student 1000M -> eval
# Launch: cd /root/era_dexhand_lab && setsid bash scripts/cloud_overnight_pose5.sh >logs/overnight_pose5_boot.log 2>&1 </dev/null &
# Monitor: tail -f logs/overnight_pose5.log
set -u
cd /root/era_dexhand_lab || exit 1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cloud_lib.sh"
cloud_activate_conda
M=logs/overnight_pose5.log
echo "[p5 $(date)] ===== START pose5 seed 9400-9403 (pose2基+各指更弯, scale 0.4-0.7) =====" > "$M"

# ---- 1. stop old ----
pkill -9 -f "[t]rain.py.*Isaac-Inhand" 2>/dev/null || true
sleep 8
cloud_kill_gen_grasp 2
echo "[p5] stopped: train残留=$(pgrep -fc '[t]rain.py.*Isaac-Inhand') gg残留=$(cloud_count_gen_grasp)" >> "$M"

# ---- 2. gen_grasp -> pose5 cache (scale 0.4-0.7, 新 pose) ----
CACHE=cache/xhand_grasp_linspace_0.4-0.7-10.npy
rm -f "$CACHE"
echo "[p5 $(date)] gen_grasp START -> $CACHE" >> "$M"
CUDA_VISIBLE_DEVICES=0 setsid python -u rl_isaaclab/scripts/gen_grasp.py \
  --task Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0 --num_envs 20480 --headless > logs/overnight_pose5_gen_grasp.log 2>&1 &
for k in $(seq 1 80); do
  sleep 30
  if [ -f "$CACHE" ]; then s1=$(stat -c %s "$CACHE" 2>/dev/null); sleep 10; s2=$(stat -c %s "$CACHE" 2>/dev/null)
    if [ "${s1:-0}" = "${s2:-1}" ] && [ "${s1:-0}" -gt 10000 ]; then break; fi; fi
done
cloud_kill_gen_grasp 3
if [ ! -f "$CACHE" ]; then echo "[p5 $(date)] FATAL: cache MISSING. ABORT." >> "$M"; exit 1; fi
echo "[p5 $(date)] gen_grasp DONE: $(stat -c '%s bytes' "$CACHE")" >> "$M"

# ---- 3. teacher 9400-3 (1000M) ----
for i in 0 1 2 3; do
  SEED=$((9400+i))
  CUDA_VISIBLE_DEVICES=$i nohup python -u rl_isaaclab/scripts/train.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 20480 \
    --max_agent_steps 1000000000 --seed "$SEED" > logs/diag_teacher_${SEED}.log 2>&1 &
  echo "[p5] teacher seed=$SEED gpu=$i pid=$!" >> "$M"; sleep 5
done
echo "[p5 $(date)] 4 TEACHERS LAUNCHED (1000M ~10h), waiting..." >> "$M"
wait
echo "[p5 $(date)] ===== ALL TEACHERS DONE =====" >> "$M"

# ---- 4. student 9400-3 (1000M) ----
for i in 0 1 2 3; do
  SEED=$((9400+i))
  dir=$(grep -aoE "logs/debug/2026[0-9_-]+" "logs/diag_teacher_${SEED}.log" 2>/dev/null | head -1)
  best="${dir}/stage1_nn/best.pth"
  if [ -z "$dir" ] || [ ! -f "$best" ]; then echo "[p5] SKIP student $SEED: best MISSING (dir=$dir)" >> "$M"; continue; fi
  CUDA_VISIBLE_DEVICES=$i nohup python -u rl_isaaclab/scripts/train.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 20480 \
    --max_agent_steps 1000000000 --algorithm ProprioAdapt --load_path "$best" --seed "$SEED" \
    > logs/student_${SEED}.log 2>&1 &
  echo "[p5] student seed=$SEED gpu=$i pid=$! <- $best" >> "$M"; sleep 5
done
echo "[p5 $(date)] STUDENTS LAUNCHED (1000M ~10h), waiting..." >> "$M"
wait
echo "[p5 $(date)] ===== ALL STUDENTS DONE =====" >> "$M"

# ---- 5. collect + eval ----
mkdir -p ckpts/overnight_pose5
for i in 0 1 2 3; do
  SEED=$((9400+i))
  dir=$(grep -aoE "logs/debug/2026[0-9_-]+" "logs/diag_teacher_${SEED}.log" 2>/dev/null | head -1)
  cp "${dir}/stage2_nn/best.pth" "ckpts/overnight_pose5/student_${SEED}.pth" 2>/dev/null \
    && echo "[p5] collected student_${SEED}.pth" >> "$M" || echo "[p5] WARN student_${SEED} missing" >> "$M"
done
echo "[p5 $(date)] eval START" >> "$M"
CUDA_VISIBLE_DEVICES=0 python -u rl_isaaclab/scripts/eval_batch.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 200 \
  --algorithm ProprioAdapt --so3 --ckpt_glob 'ckpts/overnight_pose5/student_*.pth' \
  > logs/overnight_pose5_eval.log 2>&1
echo "[p5 $(date)] ===== eval DONE -- PIPELINE COMPLETE =====" >> "$M"
