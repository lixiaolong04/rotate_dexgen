"""Train the shared v8 privileged teacher and deployable latent student."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.data import DexGenDataset
from dexgen.models import LatentStudentAdapter, PrivilegedLatentTeacher
from dexgen.training_utils import atomic_torch_save, worker_loader_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latest_output", type=Path, default=None)
    parser.add_argument("--priv_indices", type=int, nargs="+", default=[3, 4, 8])
    parser.add_argument("--history_steps", type=int, default=4)
    parser.add_argument("--history_stride", type=int, default=2)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--teacher_hidden_dim", type=int, default=128)
    parser.add_argument("--student_hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--student_weight", type=float, default=1.0)
    parser.add_argument("--normalizer_samples", type=int, default=200_000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _priv_stats(
    dataset: DexGenDataset,
    limit: int,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = (
        range(len(dataset))
        if len(dataset) <= limit
        else torch.linspace(0, len(dataset) - 1, limit).long().tolist()
    )
    values = torch.stack(
        [dataset._raw_item(index)["priv_info"][indices] for index in selected]
    )
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-6)


@torch.no_grad()
def validate(
    teacher: PrivilegedLatentTeacher,
    student: LatentStudentAdapter,
    loader: DataLoader,
    priv_mean: torch.Tensor,
    priv_std: torch.Tensor,
    priv_indices: list[int],
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    teacher.eval()
    student.eval()
    reconstruction_total = student_total = count = 0.0
    latent_values = []
    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        priv = batch["priv_info"][:, priv_indices].to(device, non_blocking=True)
        priv = (priv - priv_mean) / priv_std
        teacher_latent, reconstruction = teacher.reconstruct(priv)
        student_latent = student(state)
        reconstruction_total += float(F.mse_loss(reconstruction, priv, reduction="sum"))
        student_total += float(F.mse_loss(student_latent, teacher_latent, reduction="sum"))
        count += state.shape[0]
        latent_values.append(teacher_latent.cpu())
    latent = torch.cat(latent_values)
    return (
        reconstruction_total / max(count * len(priv_indices), 1),
        student_total / max(count * teacher.latent_dim, 1),
        latent.std(0, unbiased=False),
    )


def main() -> None:
    args = parse_args()
    if not args.priv_indices or len(set(args.priv_indices)) != len(args.priv_indices):
        raise ValueError("--priv_indices must contain unique values")
    if min(args.priv_indices) < 0 or max(args.priv_indices) >= 9:
        raise ValueError("--priv_indices must be within [0, 8]")
    if min(args.batch_size, args.epochs, args.latent_dim) < 1:
        raise ValueError("batch size, epochs, and latent dim must be positive")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dataset_kwargs = dict(
        rollout_path=args.data,
        history_steps=args.history_steps,
        history_stride=args.history_stride,
        sample_stride=args.sample_stride,
        include_tactile=True,
        return_fields=("state", "priv_info"),
    )
    train_set = DexGenDataset(split="train", **dataset_kwargs)
    normalizer = train_set.fit_normalizer(args.normalizer_samples)
    priv_mean, priv_std = _priv_stats(
        train_set, args.normalizer_samples, args.priv_indices
    )
    train_set.normalizer = normalizer
    val_set = DexGenDataset(split="val", normalizer=normalizer, **dataset_kwargs)
    threshold = normalizer.normalized_tactile_threshold(0.1).tolist()
    teacher = PrivilegedLatentTeacher(
        len(args.priv_indices), args.latent_dim, args.teacher_hidden_dim
    ).to(device)
    student = LatentStudentAdapter(
        history_steps=args.history_steps,
        state_dim=train_set.state_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.student_hidden_dim,
        tactile_contact_threshold_normalized=threshold,
    ).to(device)
    parameters = list(teacher.parameters()) + list(student.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader_kwargs = worker_loader_kwargs(args.num_workers, args.prefetch_factor)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
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
    priv_mean = priv_mean.to(device)
    priv_std = priv_std.to(device)
    latest_output = args.latest_output or args.output.with_name(
        args.output.stem.replace("_best", "") + "_latest" + args.output.suffix
    )
    best = float("inf")
    print(
        f"[v8-adapter] train={len(train_set)} val={len(val_set)} "
        f"history={args.history_steps} latent={args.latent_dim} "
        f"priv_indices={args.priv_indices}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        teacher.train()
        student.train()
        reconstruction_total = student_total = count = 0.0
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            state = batch["state"].to(device, non_blocking=True)
            priv = batch["priv_info"][:, args.priv_indices].to(device, non_blocking=True)
            priv = (priv - priv_mean) / priv_std
            teacher_latent, reconstruction = teacher.reconstruct(priv)
            student_latent = student(state)
            reconstruction_loss = F.smooth_l1_loss(reconstruction, priv)
            student_loss = F.mse_loss(student_latent, teacher_latent.detach())
            loss = (
                args.reconstruction_weight * reconstruction_loss
                + args.student_weight * student_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            reconstruction_total += float(reconstruction_loss.detach()) * state.shape[0]
            student_total += float(student_loss.detach()) * state.shape[0]
            count += state.shape[0]
            if step % args.log_every == 0 or step == len(train_loader):
                elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                print(
                    f"[v8-adapter] epoch={epoch:03d} step={step}/{len(train_loader)} "
                    f"recon={reconstruction_total / count:.6f} "
                    f"student={student_total / count:.6f} "
                    f"samples/s={count / elapsed:.1f}",
                    flush=True,
                )
        scheduler.step()
        val_reconstruction, val_student, latent_std = validate(
            teacher,
            student,
            val_loader,
            priv_mean,
            priv_std,
            args.priv_indices,
            device,
        )
        score = val_reconstruction + val_student
        payload = {
            "format": "xhand_dexgen_teacher_student_v8",
            "epoch": epoch,
            "teacher": teacher.state_dict(),
            "teacher_config": teacher.config(),
            "student": student.state_dict(),
            "student_config": student.config(),
            "normalizer": normalizer.state_dict(),
            "priv_info_indices": list(args.priv_indices),
            "priv_mean": priv_mean.cpu(),
            "priv_std": priv_std.cpu(),
            "val_reconstruction_mse": val_reconstruction,
            "val_student_mse": val_student,
            "teacher_latent_std": latent_std,
            "source_data": [str(path) for path in args.data],
        }
        atomic_torch_save(payload, latest_output)
        if score < best:
            best = score
            atomic_torch_save(payload, args.output)
        print(
            f"[v8-adapter] epoch={epoch:03d} val_recon={val_reconstruction:.6f} "
            f"val_student={val_student:.6f} latent_std_min={float(latent_std.min()):.6f} "
            f"best_score={best:.6f}",
            flush=True,
        )
    print(f"[v8-adapter] complete best_score={best:.6f} saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
