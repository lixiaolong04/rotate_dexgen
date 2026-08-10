"""Train a teacher-latent-conditioned v8 Diffusion or IDM policy."""

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

from dexgen.data import (
    DexGenDataset,
    FeatureNormalizer,
    checkpoint_representation,
    state_contract,
)
from dexgen.models import (
    ConditionalMotionDiffusion,
    InverseDynamicsModel,
    LatentStudentAdapter,
    PrivilegedLatentTeacher,
)
from dexgen.training_utils import atomic_torch_save, worker_loader_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("diffusion", "idm"), required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latest_output", type=Path, default=None)
    parser.add_argument("--history_steps", type=int, default=4)
    parser.add_argument("--prediction_steps", type=int, default=2)
    parser.add_argument("--future_stride", type=int, default=2)
    parser.add_argument("--history_stride", type=int, default=2)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--drop_trim_seconds", type=float, default=2.0)
    parser.add_argument("--recovery_fraction", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_adapter(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "xhand_dexgen_teacher_student_v8":
        raise ValueError(f"Unsupported adapter format: {checkpoint.get('format')!r}")
    teacher = PrivilegedLatentTeacher(**checkpoint["teacher_config"]).to(device)
    teacher.load_state_dict(checkpoint["teacher"])
    teacher.eval().requires_grad_(False)
    student = LatentStudentAdapter(**checkpoint["student_config"]).to(device)
    student.load_state_dict(checkpoint["student"])
    student.eval().requires_grad_(False)
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    return checkpoint, teacher, student, normalizer


def policy_latent(batch, adapter, teacher, student, device):
    indices = adapter["priv_info_indices"]
    mean = adapter["priv_mean"].to(device)
    std = adapter["priv_std"].to(device)
    priv = batch["priv_info"][:, indices].to(device, non_blocking=True)
    with torch.no_grad():
        teacher_value = teacher((priv - mean) / std)
        student_value = student(batch["state"].to(device, non_blocking=True))
        valid = batch["priv_info_valid"].to(device, non_blocking=True).bool()
        return torch.where(valid[:, None], teacher_value, student_value)


@torch.no_grad()
def validate_diffusion(model, teacher, student, adapter, loader, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    total = tactile_total = count = 0.0
    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        motion = batch["motion"].to(device, non_blocking=True)
        mode = batch["mode"].to(device, non_blocking=True)
        future_tactile = batch["future_tactile"].to(device, non_blocking=True)
        latent = policy_latent(batch, adapter, teacher, student, device)
        timesteps = torch.randint(
            model.diffusion_steps, (state.shape[0],), device=device, generator=generator
        )
        noise = torch.randn(motion.shape, device=device, generator=generator)
        motion_loss, tactile_loss = model.training_loss_components(
            motion,
            state,
            mode,
            future_tactile=future_tactile,
            timesteps=timesteps,
            noise=noise,
            env_latent=latent,
        )
        total += float(motion_loss.sum())
        tactile_total += float(tactile_loss.sum())
        count += state.shape[0]
    return total / max(count, 1), tactile_total / max(count, 1)


@torch.no_grad()
def validate_idm(model, teacher, student, adapter, loader, device):
    model.eval()
    squared = absolute = tactile_total = elements = samples = 0.0
    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        motion = batch["motion"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        future_tactile = batch["future_tactile"].to(device, non_blocking=True)
        latent = policy_latent(batch, adapter, teacher, student, device)
        prediction, _, tactile_prediction = model(
            state,
            motion,
            env_latent=latent,
            return_tactile_prediction=True,
        )
        squared += float((prediction - target).square().sum())
        absolute += float((prediction - target).abs().sum())
        tactile_total += float(
            F.smooth_l1_loss(tactile_prediction, future_tactile, reduction="sum")
        )
        elements += target.numel()
        samples += state.shape[0]
    return (
        squared / max(elements, 1),
        absolute / max(elements, 1),
        tactile_total / max(samples, 1),
    )


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.epochs) < 1:
        raise ValueError("batch size and epochs must be positive")
    if not 0.0 < args.recovery_fraction < 1.0:
        raise ValueError("--recovery_fraction must be in (0, 1)")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    adapter, teacher, student, normalizer = load_adapter(args.adapter, device)
    if adapter["student_config"]["history_steps"] != args.history_steps:
        raise ValueError("policy and student history contracts must match")

    fields = [
        "state", "motion", "mode", "future_tactile",
        "priv_info", "priv_info_valid",
    ]
    if args.component == "idm":
        fields.append("target")
    dataset_kwargs = dict(
        rollout_path=args.data,
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        future_stride=args.future_stride,
        history_stride=args.history_stride,
        sample_stride=args.sample_stride,
        drop_trim_seconds=args.drop_trim_seconds,
        include_tactile=True,
        return_fields=tuple(fields),
    )
    train_set = DexGenDataset(split="train", normalizer=normalizer, **dataset_kwargs)
    val_set = DexGenDataset(split="val", normalizer=normalizer, **dataset_kwargs)
    sample_weights, recovery_labels = train_set.recovery_sample_weights(
        target_fraction=args.recovery_fraction,
        explicit_only=True,
        boost=4.0,
        contact_threshold=0.1,
        low_contact_count=2,
        stable_contact_count=4,
        horizon_seconds=0.5,
        type_weights=None,
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_set), replacement=True
    )
    loader_kwargs = worker_loader_kwargs(args.num_workers, args.prefetch_factor)
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
    threshold = normalizer.normalized_tactile_threshold(0.1).tolist()
    latent_dim = adapter["teacher_config"]["latent_dim"]
    if args.component == "diffusion":
        model = ConditionalMotionDiffusion(
            state_dim=train_set.state_dim,
            history_steps=args.history_steps,
            keypoint_count=train_set.keypoint_count,
            prediction_steps=args.prediction_steps,
            hidden_dim=768,
            state_hidden_dim=1024,
            state_layers=6,
            num_blocks=3,
            diffusion_steps=100,
            noise_schedule="cosine",
            num_modes=train_set.num_modes,
            prediction_type="epsilon",
            tactile_arch="v2b",
            tactile_contact_threshold_normalized=threshold,
            env_latent_dim=latent_dim,
        ).to(device)
        learning_rate = args.learning_rate or 1e-4
    else:
        model = InverseDynamicsModel(
            state_dim=train_set.state_dim,
            history_steps=args.history_steps,
            keypoint_count=train_set.keypoint_count,
            prediction_steps=args.prediction_steps,
            action_dim=train_set.action_dim,
            hidden_dim=512,
            num_blocks=4,
            tactile_arch="v2b",
            tactile_contact_threshold_normalized=threshold,
            env_latent_dim=latent_dim,
        ).to(device)
        learning_rate = args.learning_rate or 3e-4

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    latest_output = args.latest_output or args.output.with_name(
        args.output.stem.replace("_best", "") + "_latest" + args.output.suffix
    )
    dataset_config = {
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "future_stride": args.future_stride,
        "history_stride": args.history_stride,
        "sample_stride": args.sample_stride,
        "controller_hz": 10.0,
        "drop_trim_seconds": args.drop_trim_seconds,
        "validation_fraction": 0.1,
        "split_seed": 42,
        "num_modes": train_set.num_modes,
        "recovery_fraction": args.recovery_fraction,
        "explicit_recovery_only": True,
        "v8_latent_source": "teacher_train_student_deploy",
        "v8_latent_dim": latent_dim,
        **state_contract(True, "v2b"),
    }
    best = float("inf")
    print(
        f"[v8-{args.component}] state={train_set.state_dim} latent={latent_dim} "
        f"train={len(train_set)} val={len(val_set)} recoveries={int(recovery_labels.sum())} "
        f"batch={args.batch_size} updates/epoch={len(train_loader)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = primary_total = tactile_total = count = 0.0
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            state = batch["state"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            latent = policy_latent(batch, adapter, teacher, student, device)
            future_tactile = batch["future_tactile"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if args.component == "diffusion":
                    mode = batch["mode"].to(device, non_blocking=True)
                    primary_loss, tactile_loss = model.training_loss_components(
                        motion,
                        state,
                        mode,
                        future_tactile=future_tactile,
                        env_latent=latent,
                    )
                    loss = (primary_loss + 0.05 * tactile_loss).mean()
                else:
                    target = batch["target"].to(device, non_blocking=True)
                    primary_loss, tactile_loss = model.loss_components(
                        state,
                        motion,
                        target,
                        future_tactile,
                        env_latent=latent,
                    )
                    loss = (primary_loss + 0.02 * tactile_loss).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * state.shape[0]
            primary_total += float(primary_loss.detach().mean()) * state.shape[0]
            tactile_total += float(tactile_loss.detach().mean()) * state.shape[0]
            count += state.shape[0]
            if step % args.log_every == 0 or step == len(train_loader):
                elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                print(
                    f"[v8-{args.component}] epoch={epoch:03d} step={step}/{len(train_loader)} "
                    f"train={total / count:.6f} primary={primary_total / count:.6f} "
                    f"touch={tactile_total / count:.6f} samples/s={count / elapsed:.1f}",
                    flush=True,
                )
        scheduler.step()
        if args.component == "diffusion":
            val_primary, val_tactile = validate_diffusion(
                model, teacher, student, adapter, val_loader, device, args.seed
            )
            val_metrics = {
                "val_loss": val_primary,
                "val_tactile": val_tactile,
            }
        else:
            val_primary, val_mae, val_tactile = validate_idm(
                model, teacher, student, adapter, val_loader, device
            )
            val_metrics = {
                "val_mse": val_primary,
                "val_mae": val_mae,
                "val_tactile": val_tactile,
            }
        improved = val_primary < best
        if improved:
            best = val_primary
        payload = {
            "format": f"xhand_dexgen_{args.component}_teacher_student_v8",
            "model": model.state_dict(),
            "model_config": model.config(),
            "normalizer": normalizer.state_dict(),
            "representation": checkpoint_representation(train_set.payload),
            "dataset_config": dataset_config,
            "source_data": [str(path) for path in args.data],
            "epoch": epoch,
            "best_val": best,
            **val_metrics,
            "v8_adapter_format": adapter["format"],
            "teacher_config": adapter["teacher_config"],
            "teacher": adapter["teacher"],
            "student_config": adapter["student_config"],
            "student": adapter["student"],
            "priv_info_indices": adapter["priv_info_indices"],
            "priv_mean": adapter["priv_mean"],
            "priv_std": adapter["priv_std"],
            "deployment_latent_source": "student",
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        atomic_torch_save(payload, latest_output)
        if improved:
            atomic_torch_save(payload, args.output)
        print(
            f"[v8-{args.component}] epoch={epoch:03d} val={val_primary:.6f} "
            f"val_touch={val_tactile:.6f} best={best:.6f}",
            flush=True,
        )
    print(
        f"[v8-{args.component}] complete best={best:.6f} saved={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
