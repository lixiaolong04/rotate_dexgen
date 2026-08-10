#!/bin/bash
# frontpad-tactile teacher 启动器 (tmux 里跑, 断线不影响)。
# 用法: bash scripts/fp_teacher.sh <seed> <gpu> [num_envs]
#   前pad触觉实验 teacher(PPO看priv), 其余超参=V1(pose5/kp10kd0.3/reward/DR/scale, 复用V1 grasp缓存)。
cd /root/era_dexhand_lab || exit 1
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
SEED=${1:-8500}; GPU=${2:-0}; N=${3:-20480}
mkdir -p logs
echo "[fp_teacher] seed=$SEED gpu=$GPU num_envs=$N  $(date)" >> logs/fp_teacher_$SEED.log
CUDA_VISIBLE_DEVICES=$GPU python -u rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs "$N" --seed "$SEED" \
  >> logs/fp_teacher_$SEED.log 2>&1
