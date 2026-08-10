#!/usr/bin/env bash
# Post-training validation for the independent tactile-longhistory v5 experiment.
# It deliberately never writes to the frozen v3 checkpoints or datasets.
set -euo pipefail

cd /root/rotate_dexgen
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

log_dir=dexgen/logs/v5_longhistory_validation
mkdir -p "$log_dir"

# These are the two parent training processes.  Their DataLoader workers are
# children, so waiting for the parents is sufficient and avoids racing GPUs.
for pid in 3264548 3264549; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
done

diffusion=dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v5_tactile_longhistory_best.pt
idm=dexgen/checkpoints/idm_modes_full_dr_active_recovery_v5_tactile_longhistory_best.pt
data=(
    dexgen/data/large_modes_v5/cw_full_*.pt
    dexgen/data/large_modes_v5/ccw_full_*.pt
    dexgen/data/recovery_modes_v6/cw_recovery_*.pt
    dexgen/data/recovery_modes_v6/ccw_recovery_*.pt
)

test -s "$diffusion"
test -s "$idm"

python -m pytest tests/test_dexgen_pipeline.py -q \
    > "$log_dir/pytest.log" 2>&1

python dexgen/commands/eval/eval_offline.py \
    --data "${data[@]}" \
    --diffusion "$diffusion" \
    --idm "$idm" \
    --split val --ddim_steps 25 --max_samples 100000 --device cuda:0 \
    > "$log_dir/offline.log" 2>&1

for mode in cw ccw; do
    for seed in 123 456 789; do
        python dexgen/commands/eval/eval_guided.py \
            --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
            --num_envs 20 --mode "$mode" --seed "$seed" \
            --diffusion "$diffusion" --idm "$idm" \
            --sensor_profile raw --controller guided \
            --guidance_strength 1000000 --noise_alpha 0.1 \
            --command_time_mode separate \
            --command_horizon_steps 1 --command_second_horizon_steps 2 \
            --steps 1200 --ddim_steps 25 --so3 --device cuda:0 --headless \
            > "$log_dir/guided_${mode}_seed${seed}.log" 2>&1
    done
done

touch "$log_dir/COMPLETE"
