"""Evaluate contact-risk checkpoints with causal tactile ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.commands.train.train_contact_risk import binary_auc
from dexgen.data import FeatureNormalizer
from dexgen.risk_data import RecoveryRiskDataset
from dexgen.risk_models import CandidateContactRiskHead, TactileProprioTConv
from dexgen.training_utils import worker_loader_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "xhand_dexgen_contact_risk_student_v9":
        raise ValueError("Expected a trained v9 contact-risk Student checkpoint")
    device = torch.device(args.device)
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    config = checkpoint["dataset_config"]
    data = args.data or [Path(value) for value in checkpoint["source_data"]]
    dataset = RecoveryRiskDataset(
        data,
        split="val",
        normalizer=normalizer,
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        **worker_loader_kwargs(args.num_workers, args.prefetch_factor),
    )
    student = TactileProprioTConv(**checkpoint["student_config"]).to(device)
    student.load_state_dict(checkpoint["student"])
    student.eval()
    head = CandidateContactRiskHead(**checkpoint["risk_head_config"]).to(device)
    head.load_state_dict(checkpoint["risk_head"])
    head.eval()
    zero_touch = normalizer.normalize_state(
        torch.zeros_like(normalizer.state_mean)
    )[-5:].to(device)
    variants = (
        "normal",
        "motion_zero",
        "motion_shuffle",
        "adapter_zero",
        "policy_zero",
        "all_zero",
        "adapter_shuffle",
        "all_shuffle",
    )
    logits = {name: [] for name in variants}
    labels: list[torch.Tensor] = []
    recovery_masks: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device, non_blocking=True)
            policy = batch["policy_state"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)
            labels.append(batch["label"].cpu())
            recovery_masks.append(batch["is_recovery"].bool().cpu())
            permutation = torch.randperm(history.shape[0], generator=generator).to(device)

            history_zero = history.clone()
            history_zero[..., -5:] = zero_touch
            policy_zero = policy.clone()
            policy_zero[..., -5:] = zero_touch
            history_shuffle = history.clone()
            history_shuffle[..., -5:] = history[permutation, :, -5:]
            policy_shuffle = policy.clone()
            policy_shuffle[..., -5:] = policy[permutation, :, -5:]

            latent_normal = student(history)
            latent_zero = student(history_zero)
            latent_shuffle = student(history_shuffle)
            values = {
                "normal": head(policy, motion, latent_normal, mode),
                "motion_zero": head(
                    policy, torch.zeros_like(motion), latent_normal, mode
                ),
                "motion_shuffle": head(
                    policy, motion[permutation], latent_normal, mode
                ),
                "adapter_zero": head(policy, motion, latent_zero, mode),
                "policy_zero": head(policy_zero, motion, latent_normal, mode),
                "all_zero": head(policy_zero, motion, latent_zero, mode),
                "adapter_shuffle": head(policy, motion, latent_shuffle, mode),
                "all_shuffle": head(policy_shuffle, motion, latent_shuffle, mode),
            }
            for name, value in values.items():
                logits[name].append(value.cpu())

    label = torch.cat(labels)
    recovery = torch.cat(recovery_masks)
    print(
        f"[risk-ablation] checkpoint={args.checkpoint} samples={len(dataset)} "
        f"recovery_samples={int(recovery.sum())}",
        flush=True,
    )
    for name in variants:
        value = torch.cat(logits[name])
        print(
            f"{name:16s} overall_auc={binary_auc(value, label):.6f} "
            f"recovery_auc={binary_auc(value[recovery], label[recovery]):.6f} "
            f"regular_auc={binary_auc(value[~recovery], label[~recovery]):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
