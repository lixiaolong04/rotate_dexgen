#!/bin/bash
# 可视化 frontpad(option C 几何过滤)student —— 打开 Isaac Sim GUI 看它转物体。
# 用法: bash scripts/play_frontpad.sh [seed=8500] [num_envs=20]
#   seed: 8500(sim最优) / 8501 / 8502 / 8503
#   ⚠️ num_envs 必须能被 10 整除(scale 10 桶); 默认 20。
cd /media/user/Data/Robotera/era_dexhand_lab || exit 1
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
SEED=${1:-8500}
N=${2:-20}
CKPT="ckpts/frontpad_students_20260701/student_${SEED}.pth"
[ -f "$CKPT" ] || { echo "缺权重: $CKPT"; exit 2; }
echo "[play] seed=$SEED num_envs=$N  ckpt=$CKPT"
python rl_isaaclab/scripts/play.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs "$N" \
  --load_path "$CKPT" \
  --algorithm ProprioAdapt
