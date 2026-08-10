#!/bin/bash
# frontpad TensorBoard 启动器 (tmux 里跑)。自动把 4 个 teacher(+ student, 若已起)的
# run dir 软链到 logs/tb_frontpad/ 下, 再开 TB。重复运行会刷新软链(student 起来后再跑一次即可纳入)。
# 用法: bash scripts/fp_tb.sh [port=16010]
cd /root/era_dexhand_lab || exit 1
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
PORT=${1:-16010}
rm -rf logs/tb_frontpad; mkdir -p logs/tb_frontpad
for s in 8500 8501 8502 8503; do
  td=$(grep -oE "logs/debug/[0-9_-]+" logs/fp_teacher_$s.log 2>/dev/null | head -1)
  [ -n "$td" ] && [ -d "$td" ] && ln -sf "$PWD/$td" "logs/tb_frontpad/teacher_$s"
  sd=$(grep -oE "logs/debug/[0-9_-]+" logs/fp_student_$s.log 2>/dev/null | head -1)
  [ -n "$sd" ] && [ -d "$sd" ] && ln -sf "$PWD/$sd" "logs/tb_frontpad/student_$s"
done
echo "[fp_tb] linked: $(ls logs/tb_frontpad)"
tensorboard --logdir logs/tb_frontpad --port "$PORT" --host 0.0.0.0
