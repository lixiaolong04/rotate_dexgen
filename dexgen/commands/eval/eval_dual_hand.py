"""Visualize the RL coarse command beside the object-hand DexGen execution.

The spatially isolated hand on the right displays the same noisy RL coarse target
that is converted by exact FK into the guidance command.  The physical hand
continues to execute Guided Diffusion -> IDM while rotating the cylinder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
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
            "dexgen/checkpoints/diffusion_modes_full_dr_epsilon_cosine_v5_best.pt",
        ),
        ("--idm", "dexgen/checkpoints/idm_modes_full_dr_v5_best.pt"),
        ("--sensor_profile", "raw"),
        ("--command_time_mode", "separate"),
        ("--command_horizon_steps", "1"),
        ("--command_second_horizon_steps", "2"),
        ("--guidance_strength", "100000"),
        ("--noise_alpha", "0.1"),
        ("--ddim_steps", "25"),
        ("--steps", "3000"),
        ("--seed", "123"),
        ("--diffusion_seed", "2026"),
    )
    effective_args = ["--dual_hand_visualization"]
    for option, value in defaults:
        if not has_option(user_args, option):
            effective_args.extend((option, value))
    effective_args.extend(user_args)

    os.chdir(REPO_ROOT)
    os.execv(
        sys.executable,
        [sys.executable, str(EVALUATOR), *effective_args],
    )


if __name__ == "__main__":
    main()
