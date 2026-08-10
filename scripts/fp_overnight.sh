#!/bin/bash
# frontpad-tactile 过夜 pipeline (tmux 里跑, 断线不影响):
#   teachers(8500-8503) 跑 TH 秒 → SIGKILL → 等GPU释放 → 各从自己 teacher best.pth 蒸 1 student → 跑 SH 秒 → SIGKILL。
# 用时间触发(不解析步数), 避开 v1 master 的 \r 解析坑。num_envs=4000(75体在20480卡PhysX, 4000总body≈V1)。
# 用法: bash scripts/fp_overnight.sh [teacher秒=28800] [student秒=18000]
cd /root/era_dexhand_lab || exit 1
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
M=logs/fp_master.log
SEEDS="8500 8501 8502 8503"
N=20480          # option C 回合并USD → 无 init 墙, 满血 20480 = V1
TH=${1:-28800}   # teacher 秒 (默认 8h; 20480 FPS高, 足够到 ~500M+)
SH=${2:-18000}   # student 秒 (默认 5h)
log(){ echo "[fp_master $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$M"; }

log "START. teachers 已在跑, 等 ${TH}s 后 kill 转 student"
sleep "$TH"
log "SIGKILL teachers"
pkill -9 -f "[t]rain.py.*Isaac-Inhand-Rotate-Xhand-Wave"
sleep 10
log "等 GPU 释放 (<2GB)"
for w in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  [ "${used:-9999}" -lt 2000 ] && { log "GPU freed (max=${used}MiB)"; break; }
  sleep 5
done
log "起 4 student(从各 teacher best.pth 蒸)"
i=0
for s in $SEEDS; do
  dir=$(grep -oiE "logs/debug/[0-9_-]+" logs/fp_teacher_$s.log 2>/dev/null | head -1)
  best="${dir}/stage1_nn/best.pth"
  if [ -n "$dir" ] && [ -f "$best" ]; then
    log "student $s <- $best (GPU $i)"
    CUDA_VISIBLE_DEVICES=$i setsid python -u rl_isaaclab/scripts/train.py \
      --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs $N \
      --algorithm ProprioAdapt --load_path "$best" --seed "$s" \
      > logs/fp_student_$s.log 2>&1 < /dev/null &
  else
    log "跳过 student $s: best.pth 没找到 (dir=$dir)"
  fi
  i=$((i+1)); sleep 12
done
log "students 起完, 等 ${SH}s 后 kill"
sleep "$SH"
log "SIGKILL students"
pkill -9 -f "[t]rain.py.*ProprioAdapt"
log "DONE. 4 student 在 <run>/stage2_nn/best.pth, 可 eval。"
