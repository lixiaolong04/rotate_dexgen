"""Train a privileged candidate-risk teacher or its 30-frame tactile student."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.data import FeatureNormalizer
from dexgen.risk_data import RecoveryRiskDataset
from dexgen.risk_models import (
    CandidateContactRiskHead,
    PrivilegedContactEncoder,
    TactileProprioTConv,
)
from dexgen.training_utils import atomic_torch_save, worker_loader_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--base_diffusion", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latest_output", type=Path, default=None)
    parser.add_argument("--history_steps", type=int, default=30)
    parser.add_argument("--history_stride", type=int, default=1)
    parser.add_argument("--prediction_steps", type=int, default=2)
    parser.add_argument("--future_stride", type=int, default=2)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--risk_horizon_seconds", type=float, default=1.0)
    parser.add_argument("--disable_tactile_ids", type=int, nargs="*", default=[])
    parser.add_argument(
        "--object_dynamics_only",
        action="store_true",
        help="Use deploy-unavailable object pose/velocity as Teacher input without static priv_info.",
    )
    parser.add_argument(
        "--recovery_label_mode",
        choices=("event", "local"),
        default="event",
        help="Use whole-event success or candidate-local future contact/rotation labels.",
    )
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--teacher_hidden_dim", type=int, default=128)
    parser.add_argument("--student_hidden_dim", type=int, default=128)
    parser.add_argument("--risk_hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--reconstruction_weight", type=float, default=0.1)
    parser.add_argument("--student_risk_weight", type=float, default=0.25)
    parser.add_argument("--normalizer_samples", type=int, default=200_000)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; zero disables.",
    )
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def binary_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(logits)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, logits.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


@torch.no_grad()
def validate(
    teacher: PrivilegedContactEncoder,
    head: CandidateContactRiskHead,
    loader: DataLoader,
    priv_mean: torch.Tensor,
    priv_std: torch.Tensor,
    device: torch.device,
    student: TactileProprioTConv | None = None,
) -> dict[str, float]:
    teacher.eval()
    head.eval()
    if student is not None:
        student.eval()
    teacher_logits: list[torch.Tensor] = []
    student_logits: list[torch.Tensor] = []
    zero_logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    recovery_masks: list[torch.Tensor] = []
    latent_error = latent_count = 0.0
    for batch in loader:
        policy_state = batch["policy_state"].to(device, non_blocking=True)
        motion = batch["motion"].to(device, non_blocking=True)
        mode = batch["mode"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        privileged = batch["privileged"].to(device, non_blocking=True)
        teacher_latent = teacher((privileged - priv_mean) / priv_std)
        teacher_value = head(policy_state, motion, teacher_latent, mode)
        teacher_logits.append(teacher_value.cpu())
        zero_logits.append(head(policy_state, motion, torch.zeros_like(teacher_latent), mode).cpu())
        labels.append(label.cpu())
        recovery_masks.append(batch["is_recovery"].bool().cpu())
        if student is not None:
            student_latent = student(batch["history"].to(device, non_blocking=True))
            student_logits.append(head(policy_state, motion, student_latent, mode).cpu())
            latent_error += float((student_latent - teacher_latent).square().sum())
            latent_count += teacher_latent.numel()
    label = torch.cat(labels)
    recovery = torch.cat(recovery_masks)
    teacher_value = torch.cat(teacher_logits)
    zero_value = torch.cat(zero_logits)
    result = {
        "teacher_bce": float(F.binary_cross_entropy_with_logits(teacher_value, label)),
        "teacher_auc": binary_auc(teacher_value, label),
        "zero_latent_auc": binary_auc(zero_value, label),
        "positive_fraction": float(label.mean()),
        "teacher_recovery_auc": binary_auc(
            teacher_value[recovery], label[recovery]
        ),
        "teacher_regular_auc": binary_auc(
            teacher_value[~recovery], label[~recovery]
        ),
        "recovery_positive_fraction": float(label[recovery].mean()),
    }
    if student is not None:
        student_value = torch.cat(student_logits)
        result.update(
            {
                "student_bce": float(F.binary_cross_entropy_with_logits(student_value, label)),
                "student_auc": binary_auc(student_value, label),
                "student_recovery_auc": binary_auc(
                    student_value[recovery], label[recovery]
                ),
                "student_regular_auc": binary_auc(
                    student_value[~recovery], label[~recovery]
                ),
                "student_teacher_probability_mae": float(
                    (student_value.sigmoid() - teacher_value.sigmoid()).abs().mean()
                ),
                "student_teacher_latent_mse": latent_error / max(latent_count, 1.0),
            }
        )
    return result


def load_base_normalizer(path: Path) -> tuple[dict, FeatureNormalizer]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    if config.get("state_dim") != 83 or config.get("history_steps") != 4:
        raise ValueError("Contact risk training requires the proven v3 4x83 tactile checkpoint")
    return checkpoint, FeatureNormalizer.from_state_dict(checkpoint["normalizer"])


def build_datasets(args: argparse.Namespace, normalizer: FeatureNormalizer):
    kwargs = dict(
        rollout_path=args.data,
        normalizer=normalizer,
        history_steps=args.history_steps,
        history_stride=args.history_stride,
        prediction_steps=args.prediction_steps,
        future_stride=args.future_stride,
        sample_stride=args.sample_stride,
        risk_horizon_seconds=args.risk_horizon_seconds,
        disable_tactile_ids=args.disable_tactile_ids,
        use_priv_info=not args.object_dynamics_only,
        recovery_label_mode=args.recovery_label_mode,
    )
    return (
        RecoveryRiskDataset(split="train", **kwargs),
        RecoveryRiskDataset(split="val", **kwargs),
    )


def main() -> None:
    args = parse_args()
    if args.stage == "student" and args.teacher_checkpoint is None:
        raise ValueError("--teacher_checkpoint is required for student training")
    if min(args.history_steps, args.batch_size, args.epochs, args.latent_dim) < 1:
        raise ValueError("history, batch, epoch, and latent dimensions must be positive")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0.0:
        raise ValueError("early-stopping arguments must be non-negative")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    base_checkpoint, normalizer = load_base_normalizer(args.base_diffusion)
    train_set, val_set = build_datasets(args, normalizer)
    loader_kwargs = worker_loader_kwargs(args.num_workers, args.prefetch_factor)
    sampler = WeightedRandomSampler(
        train_set.balanced_sample_weights(), num_samples=len(train_set), replacement=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )
    latest_output = args.latest_output or args.output.with_name(
        args.output.stem.replace("_best", "") + "_latest" + args.output.suffix
    )

    if args.stage == "teacher":
        priv_mean, priv_std = train_set.fit_privileged_stats(args.normalizer_samples)
        teacher = PrivilegedContactEncoder(
            train_set.privileged_dim, args.latent_dim, args.teacher_hidden_dim
        ).to(device)
        head = CandidateContactRiskHead(
            state_dim=train_set.state_dim,
            keypoint_count=train_set.keypoint_count,
            prediction_steps=args.prediction_steps,
            latent_dim=args.latent_dim,
            num_modes=train_set.num_modes,
            hidden_dim=args.risk_hidden_dim,
        ).to(device)
        parameters = list(teacher.parameters()) + list(head.parameters())
    else:
        teacher_checkpoint = torch.load(
            args.teacher_checkpoint, map_location="cpu", weights_only=False
        )
        if teacher_checkpoint.get("format") != "xhand_dexgen_contact_risk_teacher_v9":
            raise ValueError("Unsupported contact-risk teacher checkpoint")
        expected_mask = tuple(teacher_checkpoint["dataset_config"]["disable_tactile_ids"])
        if expected_mask != tuple(sorted(set(args.disable_tactile_ids))):
            raise ValueError(
                f"Teacher tactile mask {expected_mask} differs from requested {args.disable_tactile_ids}"
            )
        expected_dynamics_only = bool(
            teacher_checkpoint["dataset_config"].get("object_dynamics_only", False)
        )
        if expected_dynamics_only != bool(args.object_dynamics_only):
            raise ValueError("Teacher privileged-state contract differs from the requested dataset")
        expected_label_mode = teacher_checkpoint["dataset_config"].get(
            "recovery_label_mode", "event"
        )
        if expected_label_mode != args.recovery_label_mode:
            raise ValueError("Teacher recovery-label contract differs from the requested dataset")
        teacher = PrivilegedContactEncoder(**teacher_checkpoint["teacher_config"]).to(device)
        teacher.load_state_dict(teacher_checkpoint["teacher"])
        head = CandidateContactRiskHead(**teacher_checkpoint["risk_head_config"]).to(device)
        head.load_state_dict(teacher_checkpoint["risk_head"])
        teacher.eval().requires_grad_(False)
        head.eval().requires_grad_(False)
        priv_mean = teacher_checkpoint["privileged_mean"]
        priv_std = teacher_checkpoint["privileged_std"]
        student = TactileProprioTConv(
            state_dim=train_set.state_dim,
            history_steps=args.history_steps,
            latent_dim=teacher.latent_dim,
            hidden_dim=args.student_hidden_dim,
        ).to(device)
        parameters = list(student.parameters())

    priv_mean = priv_mean.to(device)
    priv_std = priv_std.to(device)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = float("inf")
    stale_epochs = 0
    print(
        f"[risk-{args.stage}] train={len(train_set)} val={len(val_set)} "
        f"classes={train_set.class_summary()} mask={args.disable_tactile_ids} "
        f"batch={args.batch_size} updates/epoch={len(train_loader)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        if args.stage == "teacher":
            teacher.train()
            head.train()
        else:
            student.train()
        total = primary_total = auxiliary_total = count = 0.0
        started = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            history = batch["history"].to(device, non_blocking=True)
            policy_state = batch["policy_state"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            privileged = batch["privileged"].to(device, non_blocking=True)
            normalized_privileged = (privileged - priv_mean) / priv_std
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if args.stage == "teacher":
                    latent, reconstruction = teacher.reconstruct(normalized_privileged)
                    logits = head(policy_state, motion, latent, mode)
                    primary = F.binary_cross_entropy_with_logits(logits, labels)
                    auxiliary = F.smooth_l1_loss(reconstruction, normalized_privileged)
                    loss = primary + args.reconstruction_weight * auxiliary
                else:
                    with torch.no_grad():
                        teacher_latent = teacher(normalized_privileged)
                    student_latent = student(history)
                    logits = head(policy_state, motion, student_latent, mode)
                    primary = F.mse_loss(student_latent, teacher_latent)
                    auxiliary = F.binary_cross_entropy_with_logits(logits, labels)
                    loss = primary + args.student_risk_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            batch_size = labels.shape[0]
            total += float(loss.detach()) * batch_size
            primary_total += float(primary.detach()) * batch_size
            auxiliary_total += float(auxiliary.detach()) * batch_size
            count += batch_size
            if step % args.log_every == 0 or step == len(train_loader):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[risk-{args.stage}] epoch={epoch:03d} step={step}/{len(train_loader)} "
                    f"loss={total / count:.6f} primary={primary_total / count:.6f} "
                    f"aux={auxiliary_total / count:.6f} samples/s={count / elapsed:.1f}",
                    flush=True,
                )
        scheduler.step()
        metrics = validate(
            teacher,
            head,
            val_loader,
            priv_mean,
            priv_std,
            device,
            student=None if args.stage == "teacher" else student,
        )
        score = (
            -metrics["teacher_recovery_auc"]
            if args.stage == "teacher" else
            metrics["student_teacher_latent_mse"]
            - 0.25 * metrics["student_recovery_auc"]
        )
        dataset_config = {
            "history_steps": args.history_steps,
            "history_stride": args.history_stride,
            "prediction_steps": args.prediction_steps,
            "future_stride": args.future_stride,
            "sample_stride": args.sample_stride,
            "risk_horizon_seconds": args.risk_horizon_seconds,
            "adapter_hz": train_set.control_hz,
            "generator_hz": float(base_checkpoint["dataset_config"]["controller_hz"]),
            "disable_tactile_ids": sorted(set(args.disable_tactile_ids)),
            "object_dynamics_only": bool(args.object_dynamics_only),
            "recovery_label_mode": args.recovery_label_mode,
        }
        payload = {
            "format": (
                "xhand_dexgen_contact_risk_teacher_v9"
                if args.stage == "teacher" else
                "xhand_dexgen_contact_risk_student_v9"
            ),
            "epoch": epoch,
            "teacher": teacher.state_dict(),
            "teacher_config": teacher.config(),
            "risk_head": head.state_dict(),
            "risk_head_config": head.config(),
            "privileged_mean": priv_mean.cpu(),
            "privileged_std": priv_std.cpu(),
            "normalizer": normalizer.state_dict(),
            "dataset_config": dataset_config,
            "source_data": [str(path) for path in args.data],
            "base_diffusion": str(args.base_diffusion),
            "metrics": metrics,
        }
        if args.stage == "student":
            payload["student"] = student.state_dict()
            payload["student_config"] = student.config()
            payload["teacher_checkpoint"] = str(args.teacher_checkpoint)
        atomic_torch_save(payload, latest_output)
        improved = score < best - args.early_stopping_min_delta
        if improved:
            best = score
            stale_epochs = 0
            atomic_torch_save(payload, args.output)
        else:
            stale_epochs += 1
        print(
            f"[risk-{args.stage}] epoch={epoch:03d} metrics={metrics} "
            f"score={score:.6f} best={best:.6f}",
            flush=True,
        )
        if (
            args.early_stopping_patience
            and stale_epochs >= args.early_stopping_patience
        ):
            print(
                f"[risk-{args.stage}] early-stop epoch={epoch:03d} "
                f"stale_epochs={stale_epochs}",
                flush=True,
            )
            break
    print(f"[risk-{args.stage}] complete saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
