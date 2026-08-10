"""Launch MANUS tabletop pickup followed by autonomous DexGen rotation.

This is intentionally separate from ``eval_manus_guided.py``.  It selects the
single-hand tabletop scene while reusing the established direct-MANUS and
guided DexGen control implementation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "dexgen" / "commands" / "eval" / "eval_guided.py"


def has_option(arguments: list[str], name: str) -> bool:
    return name in arguments or any(value.startswith(f"{name}=") for value in arguments)


def main() -> None:
    user_args = sys.argv[1:]
    defaults = (
        ("--num_envs", "1"),
        ("--controller", "guided"),
        ("--mode", "cw"),
        (
            "--diffusion",
            "dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt",
        ),
        ("--idm", "dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt"),
        ("--sensor_profile", "raw"),
        ("--command_time_mode", "broadcast"),
        ("--guidance_strength", "50000"),
        ("--noise_alpha", "0"),
        ("--manus_command_address", "tcp://127.0.0.1:8020"),
        ("--manus_command_gain", "0.55"),
        ("--manus_stale_timeout_ms", "200"),
        ("--manus_rl_noise_alpha", "0.1"),
        ("--manus_direct_max_joint_delta_rad", "0.15"),
        ("--max_command_motion_m", "0.03"),
        ("--ddim_steps", "25"),
        ("--object_scale", "0.6"),
        ("--tabletop_surface_z", "0.48"),
        ("--tabletop_size_xy", "0.50"),
        ("--tabletop_thickness", "0.06"),
        ("--tabletop_object_clearance", "0.002"),
        ("--tabletop_hand_height_m", "0.16"),
        ("--tabletop_lift_height", "0.12"),
        ("--tabletop_lift_seconds", "2.0"),
        ("--tabletop_min_contact_fingers", "3"),
        ("--tabletop_min_total_contact_force_n", "0.8"),
        ("--tabletop_grasp_assist_rad", "0.05"),
        ("--tabletop_nudge_step_m", "0.005"),
        ("--steps", "6000"),
        ("--seed", "123"),
        ("--diffusion_seed", "2026"),
    )
    effective_args = [
        "--tabletop_pickup",
        "--interactive_mode_switch",
    ]
    for option, value in defaults:
        if not has_option(user_args, option):
            effective_args.extend((option, value))
    effective_args.extend(user_args)
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, [sys.executable, str(EVALUATOR), *effective_args])


if __name__ == "__main__":
    main()
