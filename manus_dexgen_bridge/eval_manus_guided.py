"""Launch the isolated MANUS-guided DexGen evaluator.

This wrapper keeps all user-facing teleoperation defaults in the bridge folder.
The underlying evaluator's MANUS hook is default-off, so its historical RL,
noisy-RL and expert entry points retain their original behaviour.
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
        ("--guidance_strength", "100000"),
        ("--noise_alpha", "0"),
        ("--manus_command_address", "tcp://127.0.0.1:8020"),
        ("--manus_command_gain", "0.55"),
        ("--manus_stale_timeout_ms", "200"),
        ("--manus_rl_noise_alpha", "0.1"),
        ("--max_command_motion_m", "0.03"),
        ("--command_comparison_log", "manus_dexgen_bridge/logs/latest_command_comparison.pt"),
        ("--ddim_steps", "25"),
        ("--object_scale", "0.6"),
        ("--steps", "3000"),
        ("--seed", "123"),
        ("--diffusion_seed", "2026"),
    )
    effective_args = ["--dual_hand_visualization", "--manus_auto_rearm_after_reset"]
    if not has_option(user_args, "--headless") and not has_option(
        user_args, "--interactive_mode_switch"
    ):
        effective_args.append("--interactive_mode_switch")
    for option, value in defaults:
        if not has_option(user_args, option):
            effective_args.extend((option, value))
    effective_args.extend(user_args)
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, [sys.executable, str(EVALUATOR), *effective_args])


if __name__ == "__main__":
    main()
