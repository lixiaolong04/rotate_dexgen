"""Train the state/mode-conditioned keypoint-motion diffusion model."""

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
    TACTILE_ARCH_FILM,
    TACTILE_ARCH_V2B,
    checkpoint_representation,
    state_contract,
)
from dexgen.models import ConditionalMotionDiffusion
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
    parser.add_argument("--future_stride", type=int, default=2, help="2 at 20 Hz gives 0.1 s between predicted points.")
    parser.add_argument("--history_stride", type=int, default=2, help="Four state frames sampled at the 10 Hz DexGen rate.")
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--drop_trim_seconds", type=float, default=2.0)
    parser.add_argument(
        "--include_tactile", action="store_true",
        help="Append five finger-contact magnitudes to every state frame (78 -> 83 dimensions).",
    )
    parser.add_argument(
        "--tactile_arch", choices=(TACTILE_ARCH_DIRECT, TACTILE_ARCH_V2B, TACTILE_ARCH_FILM),
        default=TACTILE_ARCH_DIRECT,
        help="direct_v1 concatenates; v2b fuses touch once; film_v3 injects touch into every FiLM block.",
    )
    parser.add_argument("--tactile_per_finger_dim", type=int, default=32)
    parser.add_argument("--tactile_embedding_dim", type=int, default=128)
    parser.add_argument("--tactile_contact_threshold", type=float, default=0.1)
    parser.add_argument("--tactile_aux_weight", type=float, default=0.05)
    parser.add_argument("--tactile_condition_dropout", type=float, default=0.1)
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
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--state_hidden_dim", type=int, default=1024)
    parser.add_argument("--state_layers", type=int, default=6)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--prediction_type", choices=("epsilon", "x0"), default="epsilon")
    parser.add_argument(
        "--noise_schedule", choices=("linear", "cosine"), default="linear",
        help="Use cosine for 100-step epsilon training so terminal SNR is effectively zero.",
    )
    parser.add_argument("--normalizer_samples", type=int, default=200_000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def validate(
    model: ConditionalMotionDiffusion,
    loader: DataLoader,
    device: torch.device,
    validation_seed: int,
) -> tuple[float, float | None, dict[int, float]]:
    """Use identical timestep/noise draws every epoch for comparable validation."""
    model.eval()
    total = count = 0.0
    mode_total: dict[int, float] = {}
    mode_count: dict[int, int] = {}
    tactile_total = tactile_count = 0.0
    generator = torch.Generator(device=device).manual_seed(validation_seed)
    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        motion = batch["motion"].to(device, non_blocking=True)
        mode = batch["mode"].to(device, non_blocking=True)
        timesteps = torch.randint(
            model.diffusion_steps, (state.shape[0],), device=device, generator=generator
        )
        noise = torch.randn(motion.shape, device=device, dtype=motion.dtype, generator=generator)
        future_tactile = (
            batch["future_tactile"].to(device, non_blocking=True)
            if model.tactile_arch in {TACTILE_ARCH_V2B, TACTILE_ARCH_FILM} else None
        )
        losses, tactile_losses = model.training_loss_components(
            motion,
            state,
            mode,
            future_tactile=future_tactile,
            timesteps=timesteps,
            noise=noise,
        )
        total += float(losses.sum())
        count += losses.numel()
        for mode_id in mode.unique().tolist():
            selected = mode == mode_id
            mode_total[mode_id] = mode_total.get(mode_id, 0.0) + float(losses[selected].sum())
            mode_count[mode_id] = mode_count.get(mode_id, 0) + int(selected.sum())
        if tactile_losses is not None:
            tactile_total += float(tactile_losses.sum())
            tactile_count += tactile_losses.numel()
    return total / max(count, 1.0), (
        tactile_total / max(tactile_count, 1.0) if tactile_count else None
    ), {
        mode_id: value / max(mode_count[mode_id], 1) for mode_id, value in mode_total.items()
    }


def main() -> None:
    args = parse_args()
    if args.freeze_rma_tactile_backbone and args.rma_init is None:
        raise ValueError("--freeze_rma_tactile_backbone requires --rma_init")
    if args.output is None:
        args.output = Path(
            "dexgen/checkpoints/diffusion_modes_tactile_film_v3_epsilon_best.pt"
            if args.include_tactile and args.tactile_arch == TACTILE_ARCH_FILM else (
                "dexgen/checkpoints/diffusion_modes_tactile_v2b_epsilon_best.pt"
                if args.include_tactile and args.tactile_arch == TACTILE_ARCH_V2B else (
                "dexgen/checkpoints/diffusion_modes_tactile_epsilon_best.pt"
                if args.include_tactile
                else "dexgen/checkpoints/diffusion_modes_epsilon_best.pt"
                )
            )
        )
    if args.epochs < 1 or args.batch_size < 1 or args.normalizer_samples < 1:
        raise ValueError("epochs, batch_size, and normalizer_samples must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1 or args.log_every < 1:
        raise ValueError("invalid worker, prefetch, or logging configuration")
    if args.tactile_arch in {TACTILE_ARCH_V2B, TACTILE_ARCH_FILM} and not args.include_tactile:
        raise ValueError(f"--tactile_arch {args.tactile_arch} requires --include_tactile")
    if min(args.tactile_per_finger_dim, args.tactile_embedding_dim) < 1:
        raise ValueError("tactile encoder dimensions must be positive")
    if args.tactile_aux_weight < 0.0:
        raise ValueError("tactile auxiliary weight must be non-negative")
    if not 0.0 <= args.tactile_condition_dropout < 1.0:
        raise ValueError("tactile condition dropout must be in [0, 1)")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume is not None else None
    )
    if resume is not None and resume.get("format") not in {
        "xhand_dexgen_diffusion_v4", "xhand_dexgen_diffusion_v5",
        "xhand_dexgen_diffusion_tactile_v1", "xhand_dexgen_diffusion_tactile_v2b",
        "xhand_dexgen_diffusion_tactile_film_v3",
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
            ("state", "motion", "mode", "future_tactile")
            if args.include_tactile else ("state", "motion", "mode")
        ),
    )
    train_set = DexGenDataset(split="train", **dataset_kwargs)
    if resume is None:
        print(f"[diffusion] fitting normalizer from up to {args.normalizer_samples} windows", flush=True)
        normalizer = train_set.fit_normalizer(max_samples=args.normalizer_samples)
    else:
        normalizer = FeatureNormalizer.from_state_dict(resume["normalizer"])
    train_set.normalizer = normalizer
    val_set = DexGenDataset(split="val", normalizer=normalizer, **dataset_kwargs)

    recovery_count = 0
    if args.tactile_arch in {TACTILE_ARCH_V2B, TACTILE_ARCH_FILM}:
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
            if args.tactile_arch in {TACTILE_ARCH_V2B, TACTILE_ARCH_FILM} else None
        )
        model = ConditionalMotionDiffusion(
            state_dim=train_set.state_dim,
            history_steps=args.history_steps,
            keypoint_count=train_set.keypoint_count,
            prediction_steps=args.prediction_steps,
            hidden_dim=args.hidden_dim,
            state_hidden_dim=args.state_hidden_dim,
            state_layers=args.state_layers,
            num_blocks=args.num_blocks,
            diffusion_steps=args.diffusion_steps,
            noise_schedule=args.noise_schedule,
            num_modes=train_set.num_modes,
            prediction_type=args.prediction_type,
            tactile_arch=args.tactile_arch if args.include_tactile else "none",
            tactile_per_finger_dim=args.tactile_per_finger_dim,
            tactile_embedding_dim=args.tactile_embedding_dim,
            tactile_contact_threshold_normalized=threshold_normalized,
            tactile_condition_dropout=(
                args.tactile_condition_dropout if args.tactile_arch == TACTILE_ARCH_FILM else 0.0
            ),
        ).to(device)
        if args.rma_init is not None:
            transferred = initialize_tactile_encoder_from_rma(model, args.rma_init)
            print(f"[diffusion] RMA tactile init: {len(transferred)} tensors", flush=True)
            if args.freeze_rma_tactile_backbone:
                frozen = freeze_tactile_feature_backbone(model)
                print(f"[diffusion] frozen RMA tactile backbone: {len(frozen)} parameters", flush=True)
    else:
        model = ConditionalMotionDiffusion(**resume["model_config"]).to(device)
        model.load_state_dict(resume["model"])
        if args.freeze_rma_tactile_backbone:
            frozen = freeze_tactile_feature_backbone(model)
            print(f"[diffusion] resume-frozen RMA tactile backbone: {len(frozen)} parameters", flush=True)
        if model.prediction_type != args.prediction_type:
            raise ValueError(
                f"Resume checkpoint prediction_type={model.prediction_type!r}, "
                f"but CLI requested {args.prediction_type!r}"
            )
        if model.noise_schedule != args.noise_schedule:
            raise ValueError(
                f"Resume checkpoint noise_schedule={model.noise_schedule!r}, "
                f"but CLI requested {args.noise_schedule!r}"
            )
    if (
        model.state_dim != train_set.state_dim
        or model.keypoint_count != train_set.keypoint_count
        or model.num_modes != train_set.num_modes
    ):
        raise ValueError("Resume/model dimensions do not match the supplied dataset")
    requested_tactile_arch = args.tactile_arch if args.include_tactile else "none"
    if resume is not None and model.tactile_arch != requested_tactile_arch:
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
        best_val = float(resume.get("best_val", resume.get("val_loss", float("inf"))))
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
    if args.tactile_arch in {TACTILE_ARCH_V2B, TACTILE_ARCH_FILM}:
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
            "tactile_condition_dropout": (
                args.tactile_condition_dropout if args.tactile_arch == TACTILE_ARCH_FILM else 0.0
            ),
        })
    source_data = [str(path) for path in args.data]
    if resume is not None:
        validate_resume_contract(resume, dataset_config, source_data)

    def checkpoint(
        epoch: int,
        val_loss: float,
        val_tactile: float | None,
        mode_val: dict[int, float],
    ) -> dict:
        return {
            "format": (
                "xhand_dexgen_diffusion_tactile_film_v3"
                if args.tactile_arch == TACTILE_ARCH_FILM else (
                    "xhand_dexgen_diffusion_tactile_v2b"
                    if args.tactile_arch == TACTILE_ARCH_V2B else (
                    "xhand_dexgen_diffusion_tactile_v1"
                    if args.include_tactile else "xhand_dexgen_diffusion_v5"
                    )
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
            "val_loss": val_loss,
            "val_tactile": val_tactile,
            "val_loss_by_mode": mode_val,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            **capture_rng_state(),
        }

    print(
        f"[diffusion] prediction={model.prediction_type} schedule={model.noise_schedule} "
        f"state={dataset_config['state_schema']}({train_set.state_dim}) "
        f"train={len(train_set)} val={len(val_set)} "
        f"batches={len(train_loader)}/{len(val_loader)} recoveries={recovery_count} "
        f"start_epoch={start_epoch} "
        f"best={args.output} latest={latest_output}", flush=True,
    )
    if start_epoch > args.epochs:
        print(f"[diffusion] checkpoint already reached epoch {start_epoch - 1}", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = motion_total = tactile_total = count = 0
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, 1):
            state = batch["state"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if model.tactile_arch in {TACTILE_ARCH_V2B, TACTILE_ARCH_FILM}:
                    future_tactile = batch["future_tactile"].to(device, non_blocking=True)
                    motion_losses, tactile_losses = model.training_loss_components(
                        motion, state, mode, future_tactile=future_tactile
                    )
                    assert tactile_losses is not None
                    loss = (motion_losses + args.tactile_aux_weight * tactile_losses).mean()
                    motion_batch = float(motion_losses.detach().mean())
                    tactile_batch = float(tactile_losses.detach().mean())
                else:
                    loss = model.training_loss(motion, state, mode)
                    motion_batch = float(loss.detach())
                    tactile_batch = 0.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * state.shape[0]
            motion_total += motion_batch * state.shape[0]
            tactile_total += tactile_batch * state.shape[0]
            count += state.shape[0]
            if batch_index % args.log_every == 0 or batch_index == len(train_loader):
                elapsed = max(time.perf_counter() - epoch_start, 1.0e-9)
                print(
                    f"[diffusion] epoch={epoch:03d} step={batch_index}/{len(train_loader)} "
                    f"train={total / max(count, 1):.6f} "
                    f"motion={motion_total / max(count, 1):.6f} "
                    f"touch={tactile_total / max(count, 1):.6f} "
                    f"samples/s={count / elapsed:.1f}",
                    flush=True,
                )
        scheduler.step()
        train_loss = total / max(count, 1)
        val_loss, val_tactile, mode_val = validate(
            model, val_loader, device, validation_seed=args.seed + 10_000
        )
        mode_text = " ".join(f"mode{mode_id}={value:.6f}" for mode_id, value in sorted(mode_val.items()))
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
        payload = checkpoint(epoch, val_loss, val_tactile, mode_val)
        atomic_torch_save(payload, latest_output)
        if improved:
            atomic_torch_save(payload, args.output)
        print(
            f"[diffusion] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"val_touch={val_tactile if val_tactile is not None else float('nan'):.6f} "
            f"{mode_text} best={best_val:.6f} saved_latest={latest_output}", flush=True,
        )
    print(f"[diffusion] best val={best_val:.6f}; saved to {args.output}")


if __name__ == "__main__":
    main()
