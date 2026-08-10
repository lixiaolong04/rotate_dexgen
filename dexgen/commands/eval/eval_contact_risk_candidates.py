"""Diagnose risk-ranked Diffusion candidates after the frozen v3 IDM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.commands.eval.eval_offline import load_models  # noqa: E402
from dexgen.contact_risk_runtime import (  # noqa: E402
    load_contact_risk_runtime,
    sample_motion_candidates,
)
from dexgen.risk_data import RecoveryRiskDataset  # noqa: E402
from dexgen.data import FeatureNormalizer  # noqa: E402
from dexgen.training_utils import worker_loader_kwargs  # noqa: E402
from dexgen.xhand_keypoints import exact_keypoint_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact_risk", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--diffusion",
        type=Path,
        default=Path(
            "dexgen/checkpoints/"
            "diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt"
        ),
    )
    parser.add_argument(
        "--idm",
        type=Path,
        default=Path(
            "dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt"
        ),
    )
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--min_logit_gain", type=float, default=0.25)
    parser.add_argument("--ddim_steps", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--risk_seed", type=int, default=90_031)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    left = left - left.mean()
    right = right - right.mean()
    return float(
        (left * right).sum()
        / (left.square().sum().sqrt() * right.square().sum().sqrt()).clamp_min(
            1.0e-12
        )
    )


def main() -> None:
    args = parse_args()
    if args.candidates < 1 or args.min_logit_gain < 0.0:
        raise ValueError("candidates must be positive and margin non-negative")
    device = torch.device(args.device)
    risk_checkpoint = torch.load(
        args.contact_risk, map_location="cpu", weights_only=False
    )

    (
        diffusion,
        idm,
        diffusion_norm,
        idm_norm,
        dataset_config,
        v8_runtime,
    ) = load_models(args.diffusion, args.idm, device)
    if v8_runtime is not None:
        raise ValueError("This diagnostic targets the frozen legacy v3")
    runtime = load_contact_risk_runtime(
        args.contact_risk,
        device,
        base_normalizer=diffusion_norm,
        state_dim=diffusion.state_dim,
        history_steps=diffusion.history_steps,
        prediction_steps=diffusion.prediction_steps,
        keypoint_count=diffusion.keypoint_count,
        num_modes=diffusion.num_modes,
    )
    config = risk_checkpoint["dataset_config"]
    dataset_normalizer = FeatureNormalizer.from_state_dict(
        risk_checkpoint["normalizer"]
    )
    data = args.data or [
        Path(value) for value in risk_checkpoint["source_data"]
    ]
    dataset = RecoveryRiskDataset(
        data,
        split="val",
        normalizer=dataset_normalizer,
        history_steps=config["history_steps"],
        history_stride=config["history_stride"],
        prediction_steps=config["prediction_steps"],
        future_stride=config["future_stride"],
        sample_stride=config["sample_stride"],
        risk_horizon_seconds=config["risk_horizon_seconds"],
        disable_tactile_ids=config["disable_tactile_ids"],
        use_priv_info=not config.get("object_dynamics_only", False),
        recovery_label_mode=config.get("recovery_label_mode", "event"),
    )
    dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        **worker_loader_kwargs(args.num_workers, 4),
    )
    baseline_generator = torch.Generator(device=device).manual_seed(args.seed)
    extra_generator = torch.Generator(device=device).manual_seed(args.risk_seed)

    values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "risk",
            "first_error",
            "second_error",
            "motion_norm",
            "target_delta",
            "selected",
            "recovery",
        )
    }
    state_mean = diffusion_norm.state_mean.to(device)
    state_std = diffusion_norm.state_std.to(device)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            policy = batch["policy_state"].to(device, non_blocking=True)
            history = batch["history"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)
            candidates = sample_motion_candidates(
                diffusion,
                policy,
                mode,
                candidate_count=args.candidates,
                ddim_steps=args.ddim_steps,
                baseline_generator=baseline_generator,
                extra_generator=extra_generator,
            )
            batch_size = policy.shape[0]
            repeated_policy = policy[:, None].expand(
                batch_size, args.candidates, *policy.shape[1:]
            ).reshape(batch_size * args.candidates, *policy.shape[1:])
            repeated_mode = mode[:, None].expand(
                batch_size, args.candidates
            ).reshape(-1)
            latent = runtime["student"](history)
            repeated_latent = latent[:, None].expand(
                batch_size, args.candidates, latent.shape[-1]
            ).reshape(batch_size * args.candidates, latent.shape[-1])
            risk = runtime["head"](
                repeated_policy,
                candidates.flatten(0, 1),
                repeated_latent,
                repeated_mode,
            ).view(batch_size, args.candidates)

            idm_targets_z = idm.act(
                repeated_policy, candidates.flatten(0, 1)
            )
            idm_targets = idm_norm.denormalize_target(idm_targets_z).view(
                batch_size, args.candidates, -1
            )
            raw_policy = policy * state_std + state_mean
            current_joint = raw_policy[:, -1, 15:36]
            current_target = raw_policy[:, -1, 36:57]
            achieved = exact_keypoint_motion(
                current_joint[:, None].expand_as(idm_targets).flatten(0, 1),
                idm_targets.flatten(0, 1),
                max_motion_m=None,
            ).view(batch_size, args.candidates, 10, 3)
            physical_candidates = diffusion_norm.denormalize_motion(candidates)
            first_error = (
                achieved - physical_candidates[:, :, 0]
            ).norm(dim=-1).mean(dim=-1) * 1000.0
            second_error = (
                achieved - physical_candidates[:, :, 1]
            ).norm(dim=-1).mean(dim=-1) * 1000.0
            motion_norm = physical_candidates.norm(dim=-1).mean(dim=(-1, -2))
            target_delta = (
                idm_targets - current_target[:, None]
            ).abs().mean(dim=-1)

            selected = risk.argmax(dim=1)
            best_gain = risk.gather(1, selected[:, None]).squeeze(1) - risk[:, 0]
            selected = torch.where(
                best_gain >= args.min_logit_gain,
                selected,
                torch.zeros_like(selected),
            )
            values["risk"].append(risk.cpu())
            values["first_error"].append(first_error.cpu())
            values["second_error"].append(second_error.cpu())
            values["motion_norm"].append(motion_norm.cpu())
            values["target_delta"].append(target_delta.cpu())
            values["selected"].append(selected.cpu())
            values["recovery"].append(batch["is_recovery"].bool())
            if batch_index % 20 == 0:
                print(
                    f"[candidate-diagnostic] processed "
                    f"{batch_index * args.batch_size}",
                    flush=True,
                )

    merged = {name: torch.cat(chunks) for name, chunks in values.items()}
    rows = torch.arange(merged["selected"].shape[0])
    selected = merged["selected"].long()
    print("\nCONTACT-RISK CANDIDATE/IDM DIAGNOSTIC")
    print(f"samples:       {len(rows)}")
    print(f"candidates:    {args.candidates}")
    print(f"non-baseline:  {100.0 * float((selected != 0).float().mean()):.2f}%")
    print(
        f"risk vs -FK first-error correlation: "
        f"{correlation(merged['risk'], -merged['first_error']):+.4f}"
    )
    print(
        f"risk vs -IDM target-delta correlation: "
        f"{correlation(merged['risk'], -merged['target_delta']):+.4f}"
    )
    for name in ("first_error", "second_error", "motion_norm", "target_delta"):
        metric = merged[name]
        baseline = metric[:, 0]
        chosen = metric[rows, selected]
        unit = "mm" if "error" in name else ("m" if name == "motion_norm" else "rad")
        print(
            f"{name:<13} baseline={float(baseline.mean()):.5f} {unit} "
            f"selected={float(chosen.mean()):.5f} {unit} "
            f"delta={float((chosen - baseline).mean()):+.5f} {unit}"
        )
    better_fk = (
        merged["first_error"][rows, selected] <= merged["first_error"][:, 0]
    )
    print(f"selected FK error <= baseline: {100.0 * float(better_fk.float().mean()):.2f}%")
    for name, mask in (
        ("recovery", merged["recovery"].bool()),
        ("regular", ~merged["recovery"].bool()),
    ):
        print(
            f"{name:<9} samples={int(mask.sum())} "
            f"nonbaseline={100.0 * float((selected[mask] != 0).float().mean()):.2f}% "
            f"first_error_delta="
            f"{float((merged['first_error'][rows, selected] - merged['first_error'][:, 0])[mask].mean()):+.5f} mm"
        )
    print("\n[IDM-consistency penalty sweep: adjusted_logit = risk - weight * error_mm]")
    for weight in (0.0, 0.05, 0.10, 0.20, 0.50):
        adjusted = merged["risk"] - weight * merged["first_error"]
        choice = adjusted.argmax(dim=1)
        adjusted_gain = adjusted[rows, choice] - adjusted[:, 0]
        choice = torch.where(
            adjusted_gain >= args.min_logit_gain,
            choice,
            torch.zeros_like(choice),
        )
        raw_gain = (
            merged["risk"][rows, choice] - merged["risk"][:, 0]
        )
        error_delta = (
            merged["first_error"][rows, choice]
            - merged["first_error"][:, 0]
        )
        print(
            f"weight={weight:<4.2f} "
            f"nonbaseline={100.0 * float((choice != 0).float().mean()):6.2f}% "
            f"raw_logit_gain={float(raw_gain.mean()):+.5f} "
            f"first_error_delta={float(error_delta.mean()):+.5f} mm"
        )


if __name__ == "__main__":
    main()
