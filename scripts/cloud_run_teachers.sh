#!/bin/bash
# 云端编排: 起 4 teacher (research_unilab_rotation = UniLab 全SO(3)复刻).
# 由 orchestrate 触发(git pull 在外层做, 本脚本只杀残留+验缓存+起 teacher).
# seed 1000-1003 / GPU 0-3 / num_envs 16384 / max_agent_steps 1000M / 错峰 5s.
# 启动: cd /root/era_dexhand_lab && bash scripts/cloud_run_teachers.sh
# 监控: tail -f logs/teacher_orchestrate.log ; 各卡 logs/diag_teacher_100X.log
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud_lib.sh
source "${SCRIPT_DIR}/cloud_lib.sh"

cd /root/era_dexhand_lab || exit 1
mkdir -p logs
M=logs/teacher_orchestrate.log
echo "[orch $(date)] START head=$(git log --oneline -1)" >> "$M"
cloud_activate_conda
# 杀掉任何残留 gen_grasp(Isaac shutdown 挂住的); 见 cloud_lib.sh 说明
cloud_kill_gen_grasp 2
echo "[orch] gen_grasp残留=$(cloud_count_gen_grasp)" >> "$M"
echo "[orch] cache: $(stat -c '%s bytes mtime=%y' cache/xhand_grasp_linspace_0.4-0.6-8.npy 2>/dev/null || echo MISSING!!!)" >> "$M"
echo "[orch] 训练摩擦(应为2.0): $(grep -E 'elastomer_base_friction|object_base_friction' rl_isaaclab/tasks/inhand_rotate/xhand_wave_env_cfg.py | head -2 | tr '\n' ' ')" >> "$M"
# 起 4 teacher
for i in 0 1 2 3; do
  SEED=$((1000+i))
  CUDA_VISIBLE_DEVICES=$i nohup python -u rl_isaaclab/scripts/train.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 16384 \
    --max_agent_steps 1000000000 --seed "$SEED" \
    > logs/diag_teacher_${SEED}.log 2>&1 &
  echo "[orch] launched seed=$SEED gpu=$i pid=$!" >> "$M"
  sleep 5
done
echo "[orch $(date)] ALL 4 TEACHERS LAUNCHED" >> "$M"
