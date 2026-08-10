"""Audit two-mode DexGen data and test whether the diffusion model uses mode."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from dexgen.data import DexGenDataset
from dexgen.commands.eval.eval_offline import load_models
from dexgen.modes import rotation_mode
from dexgen.v8_runtime import student_latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, nargs="+",
        default=[Path("dexgen/data/cylinder_rollouts_cw_v3.pt"), Path("dexgen/data/cylinder_rollouts_ccw_v3.pt")],
    )
    parser.add_argument("--diffusion", type=Path, default=Path("dexgen/checkpoints/diffusion_modes_v4_best.pt"))
    parser.add_argument("--idm", type=Path, default=Path("dexgen/checkpoints/idm_modes_v4_best.pt"))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add(stats: dict[int, dict[str, float]], mode: torch.Tensor, name: str, value: torch.Tensor) -> None:
    for mode_id in mode.unique().tolist():
        selected = mode == mode_id
        values = value[selected].flatten()
        bucket = stats[mode_id]
        bucket[f"{name}_sq"] += float(values.square().sum())
        bucket[f"{name}_abs"] += float(values.abs().sum())
        bucket[f"{name}_n"] += values.numel()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    diffusion, idm, diffusion_norm, idm_norm, dataset_cfg, v8_runtime = load_models(
        args.diffusion, args.idm, device
    )
    if diffusion.num_modes != 2:
        raise ValueError(f"Mode audit requires num_modes=2, got {diffusion.num_modes}")
    dataset = DexGenDataset(
        rollout_path=args.data,
        split="val",
        history_steps=dataset_cfg["history_steps"],
        prediction_steps=dataset_cfg["prediction_steps"],
        future_stride=dataset_cfg["future_stride"],
        history_stride=dataset_cfg["history_stride"],
        sample_stride=dataset_cfg["sample_stride"],
        validation_fraction=dataset_cfg.get("validation_fraction", 0.1),
        split_seed=dataset_cfg.get("split_seed", 42),
        drop_trim_seconds=dataset_cfg["drop_trim_seconds"],
        include_tactile=bool(dataset_cfg["include_tactile"]),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    episode_stats: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for episode in dataset.payload["episodes"]:
        mode_id = int(episode["mode_id"])
        episode_stats[mode_id]["episodes"] += 1
        episode_stats[mode_id]["drops"] += episode.get("termination") == "drop"
        episode_stats[mode_id]["transitions"] += int(episode["action"].shape[0])

    stats: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    motion_sum: dict[int, torch.Tensor] = {}
    motion_count: dict[int, int] = defaultdict(int)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        for batch in loader:
            state = batch["state"].to(device, non_blocking=True)
            true_motion = batch["motion"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)
            state_z = diffusion_norm.normalize_state(state)
            env_latent = student_latent(v8_runtime, state_z)
            noise = torch.randn(
                true_motion.shape, device=device, dtype=true_motion.dtype, generator=generator
            )
            correct_z = diffusion.sample(
                state_z, ddim_steps=args.ddim_steps, mode=mode, initial_noise=noise,
                env_latent=env_latent,
            )
            wrong_z = diffusion.sample(
                state_z, ddim_steps=args.ddim_steps, mode=1 - mode, initial_noise=noise,
                env_latent=env_latent,
            )
            correct = diffusion_norm.denormalize_motion(correct_z)
            wrong = diffusion_norm.denormalize_motion(wrong_z)

            add(stats, mode, "correct_motion_error_m", correct - true_motion)
            add(stats, mode, "wrong_motion_error_m", wrong - true_motion)
            add(stats, mode, "mode_effect_m", correct - wrong)

            idm_state = idm_norm.normalize_state(state)
            oracle_target = idm_norm.denormalize_target(
                idm.act(
                    idm_state, idm_norm.normalize_motion(true_motion),
                    env_latent=env_latent,
                )
            )
            generated_target = idm_norm.denormalize_target(
                idm.act(
                    idm_state, idm_norm.normalize_motion(correct),
                    env_latent=env_latent,
                )
            )
            add(stats, mode, "oracle_target_error_rad", oracle_target - target)
            add(stats, mode, "generated_target_error_rad", generated_target - target)

            for mode_id in mode.unique().tolist():
                selected_motion = true_motion[mode == mode_id]
                motion_sum[mode_id] = motion_sum.get(
                    mode_id, torch.zeros_like(selected_motion[0])
                ) + selected_motion.sum(0)
                motion_count[mode_id] += selected_motion.shape[0]

    print("\nTWO-MODE DATA AUDIT")
    for mode_id in sorted(episode_stats):
        name = rotation_mode(mode_id).name
        values = episode_stats[mode_id]
        print(
            f"{name}: episodes={int(values['episodes'])} drops={int(values['drops'])} "
            f"({100.0 * values['drops'] / values['episodes']:.1f}%) "
            f"transitions={int(values['transitions'])} "
            f"mean_len={values['transitions'] / values['episodes']:.1f}"
        )

    print("\nTWO-MODE MODEL AUDIT")
    for mode_id in sorted(stats):
        name = rotation_mode(mode_id).name
        values = stats[mode_id]

        def rmse(metric: str, scale: float = 1.0) -> float:
            return scale * (values[f"{metric}_sq"] / values[f"{metric}_n"]) ** 0.5

        def mae(metric: str, scale: float = 1.0) -> float:
            return scale * values[f"{metric}_abs"] / values[f"{metric}_n"]

        print(
            f"{name}: motion correct/wrong RMSE="
            f"{rmse('correct_motion_error_m', 1000):.4f}/{rmse('wrong_motion_error_m', 1000):.4f} mm "
            f"mode_effect={rmse('mode_effect_m', 1000):.4f} mm"
        )
        print(
            f"     IDM oracle/generated MAE="
            f"{mae('oracle_target_error_rad'):.5f}/{mae('generated_target_error_rad'):.5f} rad"
        )

    means = {mode_id: total / motion_count[mode_id] for mode_id, total in motion_sum.items()}
    if set(means) == {0, 1}:
        print(
            f"\ntrue-motion mode-mean separation: "
            f"{float((means[0] - means[1]).square().mean().sqrt()) * 1000:.4f} mm RMS"
        )


if __name__ == "__main__":
    main()
