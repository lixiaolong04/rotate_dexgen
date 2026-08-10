"""Train the learned inverse-dynamics model on oracle keypoint motion."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.data import (
    DexGenDataset,
    FeatureNormalizer,
    TACTILE_ARCH_DIRECT,
    TACTILE_ARCH_V2B,
    checkpoint_representation,
    state_contract,
)
from dexgen.models import InverseDynamicsModel
from dexgen.training_utils import (
    atomic_torch_save,
    capture_rng_state,
    default_latest_path,
    initialize_tactile_encoder_from_rma,
    freeze_tactile_feature_backbone,
    restore_rng_state,
    validate_resume_contract,
    worker_loader_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, nargs="+",
        default=[Path("dexgen/data/cylinder_rollouts_cw_v3.pt"), Path("dexgen/data/cylinder_rollouts_ccw_v3.pt")],
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Best checkpoint path; defaults to a separate tactile name with --include_tactile.",
    )
    parser.add_argument("--latest_output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume a full training-state checkpoint.")
    parser.add_argument("--rma_init", type=Path, default=None)
    parser.add_argument("--freeze_rma_tactile_backbone", action="store_true")
    parser.add_argument("--history_steps", type=int, default=4)
    parser.add_argument("--prediction_steps", type=int, default=2)
    parser.add_argument("--future_stride", type=int, default=2)
    parser.add_argument("--history_stride", type=int, default=2)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--drop_trim_seconds", type=float, default=2.0)
    parser.add_argument(
        "--include_tactile", action="store_true",
        help="Append five finger-contact magnitudes to every state frame (78 -> 83 dimensions).",
    )
    parser.add_argument(
        "--tactile_arch", choices=(TACTILE_ARCH_DIRECT, TACTILE_ARCH_V2B),
        default=TACTILE_ARCH_DIRECT,
        help="direct_v1 preserves concatenation; v2b uses an independent temporal tactile encoder.",
    )
    parser.add_argument("--tactile_per_finger_dim", type=int, default=32)
    parser.add_argument("--tactile_embedding_dim", type=int, default=128)
    parser.add_argument("--tactile_contact_threshold", type=float, default=0.1)
    parser.add_argument("--tactile_aux_weight", type=float, default=0.02)
    parser.add_argument("--recovery_boost", type=float, default=4.0)
    parser.add_argument(
        "--recovery_fraction", type=float, default=None,
        help="If set, make labelled recoveries this fraction of each mode's sampled batches.",
    )
    parser.add_argument("--recovery_contact_threshold", type=float, default=0.1)
    parser.add_argument("--recovery_low_contact_count", type=int, default=2)
    parser.add_argument("--recovery_stable_contact_count", type=int, default=4)
    parser.add_argument("--recovery_horizon_seconds", type=float, default=0.5)
    parser.add_argument(
        "--explicit_recovery_only", action="store_true",
        help="Only oversample collector-labelled successful recoveries; ignore tactile-only heuristic recoveries.",
    )
    parser.add_argument(
        "--recovery_type_weights", type=float, nargs=4, default=None,
        metavar=("ACTION", "RELEASE", "FORCE", "NATURAL"),
        help="Relative probability mass within the requested recovery fraction.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--normalizer_samples", type=int, default=200_000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def validate(
    model: InverseDynamicsModel, loader: DataLoader, device: torch.device
) -> tuple[float, float, float | None, dict[int, tuple[float, float]]]:
    model.eval()
    squared_error = absolute_error = count = 0.0
    tactile_error = tactile_count = 0.0
    mode_sums: dict[int, list[float]] = {}
    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        motion = batch["motion"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mode = batch["mode"].to(device, non_blocking=True)
        if model.tactile_arch == TACTILE_ARCH_V2B:
            future_tactile = batch["future_tactile"].to(device, non_blocking=True)
            prediction, _, tactile_prediction = model(
                state, motion, return_tactile_prediction=True
            )
            tactile_error += float(torch.nn.functional.smooth_l1_loss(
                tactile_prediction, future_tactile, reduction="sum"
            ))
            tactile_count += future_tactile.numel()
        else:
            prediction = model.act(state, motion)
        squared_error += float((prediction - target).square().sum())
        absolute_error += float((prediction - target).abs().sum())
        count += target.numel()
        for mode_id in mode.unique().tolist():
            error = prediction[mode == mode_id] - target[mode == mode_id]
            values = mode_sums.setdefault(mode_id, [0.0, 0.0, 0.0])
            values[0] += float(error.square().sum())
            values[1] += float(error.abs().sum())
            values[2] += error.numel()
    return (
        squared_error / max(count, 1),
        absolute_error / max(count, 1),
        tactile_error / max(tactile_count, 1) if tactile_count else None,
        {mode_id: (values[0] / max(values[2], 1), values[1] / max(values[2], 1)) for mode_id, values in mode_sums.items()},
    )


def main() -> None:
    args = parse_args()
    if args.freeze_rma_tactile_backbone and args.rma_init is None:
        raise ValueError("--freeze_rma_tactile_backbone requires --rma_init")
    if args.output is None:
        args.output = Path(
            "dexgen/checkpoints/idm_modes_tactile_v2b_best.pt"
            if args.include_tactile and args.tactile_arch == TACTILE_ARCH_V2B
            else (
                "dexgen/checkpoints/idm_modes_tactile_best.pt"
                if args.include_tactile else "dexgen/checkpoints/idm_modes_best.pt"
            )
        )
    if args.epochs < 1 or args.batch_size < 1 or args.normalizer_samples < 1:
        raise ValueError("epochs, batch_size, and normalizer_samples must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1 or args.log_every < 1:
        raise ValueError("invalid worker, prefetch, or logging configuration")
    if args.tactile_arch == TACTILE_ARCH_V2B and not args.include_tactile:
        raise ValueError("--tactile_arch v2b requires --include_tactile")
    if min(args.tactile_per_finger_dim, args.tactile_embedding_dim) < 1:
        raise ValueError("tactile encoder dimensions must be positive")
    if args.tactile_aux_weight < 0.0:
        raise ValueError("tactile auxiliary weight must be non-negative")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume is not None else None
    )
    if resume is not None and resume.get("format") not in {
        "xhand_dexgen_idm_v4", "xhand_dexgen_idm_v5",
        "xhand_dexgen_idm_tactile_v1", "xhand_dexgen_idm_tactile_v2b",
    }:
        raise ValueError(f"Unsupported resume checkpoint: {resume.get('format')!r}")

    dataset_kwargs = dict(
        rollout_path=args.data,
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        future_stride=args.future_stride,
        history_stride=args.history_stride,
        sample_stride=args.sample_stride,
        validation_fraction=args.val_fraction,
        split_seed=args.split_seed,
        drop_trim_seconds=args.drop_trim_seconds,
        include_tactile=args.include_tactile,
        return_fields=(
            ("state", "motion", "target", "mode", "future_tactile")
            if args.include_tactile else ("state", "motion", "target", "mode")
        ),
    )
    train_set = DexGenDataset(split="train", **dataset_kwargs)
    if resume is None:
        print(f"[idm] fitting normalizer from up to {args.normalizer_samples} windows", flush=True)
        normalizer = train_set.fit_normalizer(max_samples=args.normalizer_samples)
    else:
        normalizer = FeatureNormalizer.from_state_dict(resume["normalizer"])
    train_set.normalizer = normalizer
    val_set = DexGenDataset(split="val", normalizer=normalizer, **dataset_kwargs)

    recovery_count = 0
    if args.tactile_arch == TACTILE_ARCH_V2B:
        sample_weights, recovery_labels = train_set.recovery_sample_weights(
            boost=args.recovery_boost,
            target_fraction=args.recovery_fraction,
            contact_threshold=args.recovery_contact_threshold,
            low_contact_count=args.recovery_low_contact_count,
            stable_contact_count=args.recovery_stable_contact_count,
            horizon_seconds=args.recovery_horizon_seconds,
            explicit_only=args.explicit_recovery_only,
            type_weights=args.recovery_type_weights,
        )
        recovery_count = int(recovery_labels.sum())
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_set), replacement=True
        )
    else:
        sampler = WeightedRandomSampler(
            train_set.balanced_sample_weights(), num_samples=len(train_set), replacement=True
        ) if train_set.num_modes > 1 else None
    worker_kwargs = worker_loader_kwargs(args.num_workers, args.prefetch_factor)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", **worker_kwargs,
    )

    if resume is None:
        threshold_normalized = (
            normalizer.normalized_tactile_threshold(args.tactile_contact_threshold).tolist()
            if args.tactile_arch == TACTILE_ARCH_V2B else None
        )
        model = InverseDynamicsModel(
            state_dim=train_set.state_dim,
            history_steps=args.history_steps,
            keypoint_count=train_set.keypoint_count,
            prediction_steps=args.prediction_steps,
            action_dim=train_set.action_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            tactile_arch=args.tactile_arch if args.include_tactile else "none",
            tactile_per_finger_dim=args.tactile_per_finger_dim,
            tactile_embedding_dim=args.tactile_embedding_dim,
            tactile_contact_threshold_normalized=threshold_normalized,
        ).to(device)
        if args.rma_init is not None:
            transferred = initialize_tactile_encoder_from_rma(model, args.rma_init)
            print(f"[idm] RMA tactile init: {len(transferred)} tensors", flush=True)
            if args.freeze_rma_tactile_backbone:
                frozen = freeze_tactile_feature_backbone(model)
                print(f"[idm] frozen RMA tactile backbone: {len(frozen)} parameters", flush=True)
    else:
        model = InverseDynamicsModel(**resume["model_config"]).to(device)
        model.load_state_dict(resume["model"])
        if args.freeze_rma_tactile_backbone:
            frozen = freeze_tactile_feature_backbone(model)
            print(f"[idm] resume-frozen RMA tactile backbone: {len(frozen)} parameters", flush=True)
    if (
        model.state_dim != train_set.state_dim
        or model.keypoint_count != train_set.keypoint_count
        or model.action_dim != train_set.action_dim
    ):
        raise ValueError("Resume/model dimensions do not match the supplied dataset")
    requested_tactile_arch = args.tactile_arch if args.include_tactile else "none"
    if resume is not None and (
        (model.tactile_arch == TACTILE_ARCH_V2B)
        != (requested_tactile_arch == TACTILE_ARCH_V2B)
    ):
        raise ValueError(
            f"Resume tactile_arch={model.tactile_arch!r}, requested={requested_tactile_arch!r}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_val = float("inf")
    if resume is not None:
        for key, target in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
            if key not in resume:
                raise ValueError(f"Resume checkpoint is missing {key!r} training state")
            target.load_state_dict(resume[key])
        start_epoch = int(resume["epoch"]) + 1
        best_val = float(resume.get("best_val", resume.get("val_mse", float("inf"))))
        restore_rng_state(resume)

    latest_output = args.latest_output or default_latest_path(args.output)
    dataset_config = {
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "future_stride": args.future_stride,
        "history_stride": args.history_stride,
        "sample_stride": args.sample_stride,
        "controller_hz": 10.0,
        "drop_trim_seconds": args.drop_trim_seconds,
        "validation_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "num_modes": train_set.num_modes,
        **state_contract(
            args.include_tactile,
            args.tactile_arch if args.include_tactile else "none",
        ),
    }
    if args.tactile_arch == TACTILE_ARCH_V2B:
        dataset_config.update({
            "tactile_aux_weight": args.tactile_aux_weight,
            "recovery_boost": args.recovery_boost,
            "recovery_fraction": args.recovery_fraction,
            "recovery_contact_threshold": args.recovery_contact_threshold,
            "recovery_low_contact_count": args.recovery_low_contact_count,
            "recovery_stable_contact_count": args.recovery_stable_contact_count,
            "recovery_horizon_seconds": args.recovery_horizon_seconds,
            "explicit_recovery_only": bool(args.explicit_recovery_only),
            "recovery_type_weights": args.recovery_type_weights,
            "tactile_contact_threshold": args.tactile_contact_threshold,
        })
    source_data = [str(path) for path in args.data]
    if resume is not None:
        validate_resume_contract(resume, dataset_config, source_data)

    def checkpoint(
        epoch: int,
        val_mse: float,
        val_mae: float,
        val_tactile: float | None,
        mode_val: dict[int, tuple[float, float]],
    ) -> dict:
        return {
            "format": (
                "xhand_dexgen_idm_tactile_v2b"
                if args.tactile_arch == TACTILE_ARCH_V2B else (
                    "xhand_dexgen_idm_tactile_v1"
                    if args.include_tactile else "xhand_dexgen_idm_v5"
                )
            ),
            "model": model.state_dict(),
            "model_config": model.config(),
            "normalizer": normalizer.state_dict(),
            "representation": checkpoint_representation(train_set.payload),
            "dataset_config": dataset_config,
            "source_data": source_data,
            "epoch": epoch,
            "best_val": best_val,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_tactile": val_tactile,
            "val_by_mode": mode_val,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            **capture_rng_state(),
        }

    print(
        f"[idm] state={dataset_config['state_schema']}({train_set.state_dim}) "
        f"train={len(train_set)} val={len(val_set)} batches={len(train_loader)}/{len(val_loader)} "
        f"recoveries={recovery_count} start_epoch={start_epoch} "
        f"best={args.output} latest={latest_output}", flush=True,
    )
    if start_epoch > args.epochs:
        print(f"[idm] checkpoint already reached epoch {start_epoch - 1}", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = target_total = tactile_total = count = 0
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, 1):
            state = batch["state"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if model.tactile_arch == TACTILE_ARCH_V2B:
                    future_tactile = batch["future_tactile"].to(device, non_blocking=True)
                    target_losses, tactile_losses = model.loss_components(
                        state, motion, target, future_tactile
                    )
                    assert tactile_losses is not None
                    loss = (target_losses + args.tactile_aux_weight * tactile_losses).mean()
                    target_batch = float(target_losses.detach().mean())
                    tactile_batch = float(tactile_losses.detach().mean())
                else:
                    loss = model.loss(state, motion, target)
                    target_batch = float(loss.detach())
                    tactile_batch = 0.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * state.shape[0]
            target_total += target_batch * state.shape[0]
            tactile_total += tactile_batch * state.shape[0]
            count += state.shape[0]
            if batch_index % args.log_every == 0 or batch_index == len(train_loader):
                elapsed = max(time.perf_counter() - epoch_start, 1.0e-9)
                print(
                    f"[idm] epoch={epoch:03d} step={batch_index}/{len(train_loader)} "
                    f"train={total / max(count, 1):.6f} "
                    f"target={target_total / max(count, 1):.6f} "
                    f"touch={tactile_total / max(count, 1):.6f} "
                    f"samples/s={count / elapsed:.1f}",
                    flush=True,
                )
        scheduler.step()
        train_loss = total / max(count, 1)
        val_mse, val_mae, val_tactile, mode_val = validate(model, val_loader, device)
        mode_text = " ".join(
            f"mode{mode_id}_mse={values[0]:.6f}" for mode_id, values in sorted(mode_val.items())
        )
        improved = val_mse < best_val
        if improved:
            best_val = val_mse
        payload = checkpoint(epoch, val_mse, val_mae, val_tactile, mode_val)
        atomic_torch_save(payload, latest_output)
        if improved:
            atomic_torch_save(payload, args.output)
        print(
            f"[idm] epoch={epoch:03d} train={train_loss:.6f} val_mse={val_mse:.6f} "
            f"val_mae={val_mae:.6f} val_touch={val_tactile if val_tactile is not None else float('nan'):.6f} "
            f"{mode_text} best={best_val:.6f} "
            f"saved_latest={latest_output}", flush=True,
        )
    print(f"[idm] best normalized target val MSE={best_val:.6f}; saved to {args.output}")


if __name__ == "__main__":
    main()
