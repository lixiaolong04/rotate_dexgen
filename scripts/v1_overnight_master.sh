#!/bin/bash
# V1 过夜全自动 pipeline(云端 tmux 跑, 断线不影响):
#   teachers(8100-8103) 训到 TT(M步) -> SIGKILL -> 等GPU显存释放 -> 各蒸1 student(从该teacher best.pth) -> student到ST -> SIGKILL.
# 日志: logs/v1_master.log。各 teacher/student: logs/v1_teacher_<seed>.log / logs/v1_student_<seed>.log。
# 早上 eval: python rl_isaaclab/scripts/eval_batch.py ... --algorithm ProprioAdapt --ckpt_glob '<student dirs>/stage2_nn/best.pth'
cd /root/era_dexhand_lab || exit 1
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
M=logs/v1_master.log
SEEDS="8100 8101 8102 8103"
TT=500   # teacher 目标 M 步 (~plateau, 到顶; best.pth 持续存峰值)
ST=400   # student 目标 M 步
log(){ echo "[master $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$M"; }
steps(){ tr '\r' '\n' < "$1" 2>/dev/null | grep "Agent Steps" | tail -1 | grep -oP "Agent Steps: \K[0-9]+" | tail -1; }  # 2026-06-29 修: student日志\r分隔→必须 tr \r \n 再取, 否则解析出多值 [ -lt ] 报错→误判完成

log "START. 等 teachers 全部 >= ${TT}M (上限8h)"
for w in $(seq 1 480); do
  ok=1
  for s in $SEEDS; do m=$(steps logs/v1_teacher_$s.log); [ "${m:-0}" -lt "$TT" ] && ok=0; done
  [ "$ok" = 1 ] && { log "teachers 全部 >= ${TT}M"; break; }
  [ $((w % 30)) -eq 0 ] && log "等teacher(poll $w): $(for s in $SEEDS; do echo -n "$s=$(steps logs/v1_teacher_$s.log)M "; done)"
  sleep 60
done

log "SIGKILL teachers"
pkill -9 -f "[t]rain.py.*Isaac-Inhand-Rotate-Xhand-Wave"
sleep 10
log "等 GPU 显存释放 (<2GB)"
for w in $(seq 1 36); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  [ "${used:-9999}" -lt 2000 ] && { log "GPU 已释放 (max=${used}MiB)"; break; }
  sleep 5
done

log "起 4 个 student"
i=0
for s in $SEEDS; do
  dir=$(grep -oiE "logs/debug/[0-9_-]+" logs/v1_teacher_$s.log 2>/dev/null | head -1)
  best="${dir}/stage1_nn/best.pth"
  if [ -n "$dir" ] && [ -f "$best" ]; then
    log "student $s <- $best (GPU $i)"
    CUDA_VISIBLE_DEVICES=$i setsid python -u rl_isaaclab/scripts/train.py \
      --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 20480 \
      --algorithm ProprioAdapt --load_path "$best" --seed "$s" \
      > logs/v1_student_$s.log 2>&1 < /dev/null &
  else
    log "跳过 student $s: best.pth 没找到 (dir=$dir)"
  fi
  i=$((i+1)); sleep 12
done

log "等 students 全部 >= ${ST}M (上限8h; student不自停)"
for w in $(seq 1 480); do
  ok=1
  for s in $SEEDS; do
    [ -f logs/v1_student_$s.log ] || continue
    m=$(steps logs/v1_student_$s.log); [ "${m:-0}" -lt "$ST" ] && ok=0
  done
  [ "$ok" = 1 ] && { log "students 全部 >= ${ST}M"; break; }
  [ $((w % 30)) -eq 0 ] && log "等student(poll $w): $(for s in $SEEDS; do echo -n "$s=$(steps logs/v1_student_$s.log)M "; done)"
  sleep 60
done
log "SIGKILL students"
pkill -9 -f "[t]rain.py.*ProprioAdapt"
log "DONE. 4 student 在 <run>/stage2_nn/best.pth, 可 eval 挑权重。"
