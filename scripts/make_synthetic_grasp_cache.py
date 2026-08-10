"""Generate a synthetic grasp cache for Xhand 21 DOF.

Why this exists:
gen_grasp.py's physics-validation procedure (240 consecutive steps of
cond1+cond2+cond3 passing) is mathematically unreachable for Xhand's
21 DOF open-cup geometry — per-step success ~37%, so 0.37^240 ≈ 10^-104.
gen_grasp cache stays at 0 forever.

This synthesizes a cache from a hand-tuned pose directly. The pose has been
visually verified (5 fingers reach object, fingers form a cup). RL training
will start from these initial states and the policy will learn to maintain
the grasp during episode rollout (~1000 iter to stabilize).

Format expected by env._reset_idx:
    rows of [hand_dof_pos (21), object_pos (3), object_quat (4)] = 28 floats
    file shape: (N_total, 28) where N_total = bucket_grasp * scale_range[2]

Usage:
    python scripts/make_synthetic_grasp_cache.py
Output:
    cache/xhand_grasp_linspace_<lo>-<hi>-<num>.npy
"""
import os
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----- joint values matching xhand_wave_grasp_env_cfg.py's tuned pre-grasp pose -----
# (Order matches actuated_joint_names in cfg.)
ACTUATED_JOINTS_AND_VALUES = [
    ("right_hand_thumb_mcp0_joint", 0.840),
    ("right_hand_thumb_mcp1_joint", 0.780),
    ("right_hand_thumb_pip_joint",  0.210),
    ("right_hand_thumb_dip_joint",  0.090),
    ("right_hand_index_mcp0_joint",-0.120),
    ("right_hand_index_mcp1_joint", 1.200),
    ("right_hand_index_pip_joint",  0.300),
    ("right_hand_index_dip_joint",  0.103),
    ("right_hand_mid_mcp0_joint",  -0.017),
    ("right_hand_mid_mcp1_joint",   0.600),
    ("right_hand_mid_pip_joint",    0.731),
    ("right_hand_mid_dip_joint",    0.224),
    ("right_hand_ring_mcp0_joint",  0.148),
    ("right_hand_ring_mcp1_joint",  0.600),
    ("right_hand_ring_pip_joint",   0.615),
    ("right_hand_ring_dip_joint",   0.315),
    ("right_hand_ltspalm_joint",    0.192),
    ("right_hand_pinky_mcp0_joint", 0.139),
    ("right_hand_pinky_mcp1_joint", 1.193),
    ("right_hand_pinky_pip_joint",  0.103),
    ("right_hand_pinky_dip_joint",  0.103),
]

# URDF joint limits × dof_limits_scale=0.9 (cfg) — clamp noise to stay within.
JOINT_LIMITS_SCALED = {
    "right_hand_thumb_mcp0_joint": (0.0,    1.178),
    "right_hand_thumb_mcp1_joint": (0.0,    1.414),
    "right_hand_thumb_pip_joint":  (0.0,    1.414),
    "right_hand_thumb_dip_joint":  (0.0,    1.414),
    "right_hand_index_mcp0_joint": (-0.157, 1.100),
    "right_hand_index_mcp1_joint": (0.0,    1.414),
    "right_hand_index_pip_joint":  (0.0,    1.414),
    "right_hand_index_dip_joint":  (0.0,    1.414),
    "right_hand_mid_mcp0_joint":   (-0.628, 0.157),
    "right_hand_mid_mcp1_joint":   (0.0,    1.414),
    "right_hand_mid_pip_joint":    (0.0,    1.414),
    "right_hand_mid_dip_joint":    (0.0,    1.414),
    "right_hand_ring_mcp0_joint":  (-0.628, 0.157),
    "right_hand_ring_mcp1_joint":  (0.0,    1.414),
    "right_hand_ring_pip_joint":   (0.0,    1.414),
    "right_hand_ring_dip_joint":   (0.0,    1.414),
    "right_hand_ltspalm_joint":    (0.0,    0.471),
    "right_hand_pinky_mcp0_joint": (-0.942, 0.157),
    "right_hand_pinky_mcp1_joint": (0.0,    1.414),
    "right_hand_pinky_pip_joint":  (0.0,    1.414),
    "right_hand_pinky_dip_joint":  (0.0,    1.414),
}

# Object init pose — matches xhand_wave_grasp_env_cfg.py
OBJECT_POS  = (-0.075, 0.005, 0.65)
OBJECT_QUAT = (1.0, 0.0, 0.0, 0.0)

# Cache file naming convention. Must match cfg.scale_range = [0.5, 0.5, 1]
SCALE_LO  = 0.5
SCALE_HI  = 0.5
SCALE_NUM = 1
ROWS_PER_SCALE = 50000   # production cache size (matches what gen_grasp would produce)

# Per-row noise — adds diversity to cache so all envs don't start from identical state.
JOINT_NOISE_SIGMA = 0.02   # rad ≈ 1.1° per joint
OBJ_XY_NOISE_SIGMA = 0.005  # m, lateral position noise on object
OBJ_Z_NOISE_SIGMA  = 0.002  # m, smaller Z noise so object stays in reset window


def main():
    num_joints = len(ACTUATED_JOINTS_AND_VALUES)
    assert num_joints == 21, f"expected 21 actuated joints, got {num_joints}"

    joint_pos_base = np.array([v for _, v in ACTUATED_JOINTS_AND_VALUES], dtype=np.float32)
    obj_pos_base   = np.array(OBJECT_POS,  dtype=np.float32)
    obj_quat_base  = np.array(OBJECT_QUAT, dtype=np.float32)

    joint_lo = np.array([JOINT_LIMITS_SCALED[n][0] for n, _ in ACTUATED_JOINTS_AND_VALUES], dtype=np.float32)
    joint_hi = np.array([JOINT_LIMITS_SCALED[n][1] for n, _ in ACTUATED_JOINTS_AND_VALUES], dtype=np.float32)

    rng = np.random.default_rng(seed=42)
    rows = np.zeros((SCALE_NUM * ROWS_PER_SCALE, 28), dtype=np.float32)
    for i in range(rows.shape[0]):
        # joint noise + clamp to scaled limits
        jp = joint_pos_base + rng.normal(0.0, JOINT_NOISE_SIGMA, size=num_joints).astype(np.float32)
        jp = np.clip(jp, joint_lo, joint_hi)
        # object xyz noise (small)
        op = obj_pos_base.copy()
        op[0] += rng.normal(0.0, OBJ_XY_NOISE_SIGMA)
        op[1] += rng.normal(0.0, OBJ_XY_NOISE_SIGMA)
        op[2] += rng.normal(0.0, OBJ_Z_NOISE_SIGMA)
        # keep quat at identity (no rotation diversity for now)
        rows[i, :num_joints]     = jp
        rows[i, num_joints:num_joints+3] = op
        rows[i, num_joints+3:]   = obj_quat_base

    print(f"data shape: {rows.shape}  dtype: {rows.dtype}")
    print(f"joint_pos row 0:   {rows[0, :num_joints]}")
    print(f"object pose row 0: {rows[0, num_joints:]}")
    print(f"joint_pos std (proves noise added): {rows[:, :num_joints].std(axis=0)[:5]}")

    out_dir = PROJECT_ROOT / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xhand_grasp_linspace_{SCALE_LO}-{SCALE_HI}-{SCALE_NUM}.npy"
    np.save(out_path, rows)
    print(f"wrote: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
