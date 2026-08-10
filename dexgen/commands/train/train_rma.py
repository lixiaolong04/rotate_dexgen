"""Train a tactile recurrent motor-adaptation (RMA) privileged predictor."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.data import DexGenDataset
from dexgen.models import RMAAdapter
from dexgen.training_utils import atomic_torch_save, worker_loader_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--history_steps", type=int, default=16)
    parser.add_argument("--history_stride", type=int, default=2)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--normalizer_samples", type=int, default=200000)
    parser.add_argument(
        "--priv_indices", type=int, nargs="+", default=list(range(9)),
        help="Indices of priv_info targets to predict; e.g. 3 4 8 for friction/mass/scale.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latest_output", type=Path, default=None)
    return parser.parse_args()


def priv_stats(
    dataset: DexGenDataset, limit: int, target_indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_indices = (
        range(len(dataset))
        if len(dataset) <= limit
        else torch.linspace(0, len(dataset) - 1, limit).long().tolist()
    )
    values = torch.stack(
        [
            dataset._raw_item(index)["priv_info"][target_indices]
            for index in sample_indices
        ]
    )
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-6)


@torch.no_grad()
def validate(model, loader, mean, std, target_indices, device) -> tuple[float, float]:
    model.eval(); total = count = 0.0
    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        target = (
            batch["priv_info"][:, target_indices].to(device, non_blocking=True) - mean
        ) / std
        total += torch.nn.functional.mse_loss(model(state), target, reduction="sum").item()
        count += target.numel()
    mse = total / max(count, 1.0)
    return mse, mse ** 0.5


def main() -> None:
    args = parse_args()
    if not args.priv_indices or len(set(args.priv_indices)) != len(args.priv_indices):
        raise ValueError("--priv_indices must contain unique indices")
    if min(args.priv_indices) < 0 or max(args.priv_indices) >= 9:
        raise ValueError("--priv_indices must be within [0, 8]")
    train = DexGenDataset(args.data, "train", history_steps=args.history_steps,
        history_stride=args.history_stride, sample_stride=args.sample_stride,
        include_tactile=True)
    normalizer = train.fit_normalizer(args.normalizer_samples)
    mean, std = priv_stats(train, args.normalizer_samples, args.priv_indices)
    train.normalizer = normalizer
    val = DexGenDataset(args.data, "val", history_steps=args.history_steps,
        history_stride=args.history_stride, sample_stride=args.sample_stride,
        include_tactile=True, normalizer=normalizer)
    device = torch.device(args.device)
    mean, std = mean.to(device), std.to(device)
    threshold = normalizer.normalized_tactile_threshold(0.1).tolist()
    model = RMAAdapter(history_steps=args.history_steps, state_dim=train.state_dim,
        priv_info_dim=len(args.priv_indices), tactile_contact_threshold_normalized=threshold).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader_args = worker_loader_kwargs(args.num_workers, args.prefetch_factor)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, **loader_args)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, **loader_args)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = count = 0.0
        for batch in train_loader:
            state = batch["state"].to(device, non_blocking=True)
            target = (
                batch["priv_info"][:, args.priv_indices].to(device, non_blocking=True)
                - mean
            ) / std
            loss = torch.nn.functional.smooth_l1_loss(model(state), target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * state.shape[0]; count += state.shape[0]
        val_mse, val_rmse = validate(
            model, val_loader, mean, std, args.priv_indices, device
        )
        payload = {"format": "xhand_dexgen_rma_v1", "epoch": epoch,
            "model": model.state_dict(), "config": model.config(),
            "normalizer": normalizer.state_dict(), "priv_mean": mean.cpu(), "priv_std": std.cpu(),
            "priv_info_indices": list(args.priv_indices),
            "val_mse": val_mse, "val_rmse": val_rmse}
        if args.latest_output: atomic_torch_save(payload, args.latest_output)
        if val_mse < best:
            best = val_mse; atomic_torch_save(payload, args.output)
        print(f"[rma] epoch={epoch}/{args.epochs} train_huber={total/max(count,1):.6f} val_mse={val_mse:.6f} val_rmse={val_rmse:.6f}", flush=True)
    print(f"[rma] best normalized val MSE={best:.6f}; saved={args.output}")


if __name__ == "__main__":
    main()
