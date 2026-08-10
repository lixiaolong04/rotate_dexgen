#!/bin/bash
# V1 student 重蒸(修过夜误杀bug: 原master用步数判停, student日志\r分隔→解析出多值→[ -lt ]报错→误判完成)。
# 本脚本改【时间触发停止】(sleep 5h 后 kill), 不解析步数, 彻底避开。teacher best.pth 已训到500M。
# tmux 跑(断线不影响)。日志: logs/v1_student_rerun.log; 各student: logs/v1_student2_<seed>.log
cd /root/era_dexhand_lab || exit 1
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
M=logs/v1_student_rerun.log
log(){ echo "[rerun $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$M"; }
# seed -> teacher run dir (从 master 日志确认)
tdir(){ case "$1" in 8100) echo 2026-06-29_02-50-57;; 8101) echo 2026-06-29_02-51-03;; 8102) echo 2026-06-29_02-51-09;; 8103) echo 2026-06-29_02-51-15;; esac; }

log "起 4 student(从 teacher best.pth 蒸馏, GPU0-3)"
i=0
for s in 8100 8101 8102 8103; do
  best="logs/debug/$(tdir $s)/stage1_nn/best.pth"
  if [ -f "$best" ]; then
    log "student $s <- $best (GPU $i)"
    CUDA_VISIBLE_DEVICES=$i setsid python -u rl_isaaclab/scripts/train.py \
      --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 20480 \
      --algorithm ProprioAdapt --load_path "$best" --seed "$s" \
      > logs/v1_student2_$s.log 2>&1 < /dev/null &
  else
    log "缺 $best, 跳过 $s"
  fi
  i=$((i+1)); sleep 12
done
log "4 student 已起。sleep 5h(~480-500M步)后 SIGKILL。best.pth 持续存峰值, 中途也可随时取。"
sleep 18000
log "5h 到, SIGKILL students"
pkill -9 -f "[t]rain.py.*ProprioAdapt"
log "DONE. 4 student 在 <run>/stage2_nn/best.pth, 可 eval。"
