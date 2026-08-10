"""Utilities shared by long-running DexGen training jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def default_latest_path(best_path: Path) -> Path:
    """Derive a readable latest-checkpoint path from the best path."""
    stem = best_path.stem
    if stem.endswith("_best"):
        stem = stem[:-5]
    return best_path.with_name(f"{stem}_latest{best_path.suffix}")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def worker_loader_kwargs(num_workers: int, prefetch_factor: int) -> dict[str, Any]:
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }


def initialize_tactile_encoder_from_rma(
    model: torch.nn.Module, checkpoint_path: Path
) -> list[str]:
    """Transfer shape-compatible RMA tactile features into a DexGen model."""
    encoder = getattr(model, "tactile_encoder", None)
    if encoder is None:
        raise ValueError("RMA initialization requires a model with tactile_encoder")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "xhand_dexgen_rma_v1":
        raise ValueError(f"Unsupported RMA checkpoint: {checkpoint.get('format')!r}")
    source = checkpoint["model"]
    target = encoder.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    for key, value in target.items():
        # This buffer depends on the destination dataset normalizer and must
        # retain the value computed by the DexGen training script.
        if key == "contact_threshold_normalized":
            continue
        source_key = f"tactile_encoder.{key}"
        if source_key in source and source[source_key].shape == value.shape:
            transferred[key] = source[source_key]
    if not transferred:
        raise ValueError("RMA checkpoint has no shape-compatible tactile encoder weights")
    target.update(transferred)
    encoder.load_state_dict(target)
    return sorted(transferred)


def freeze_tactile_feature_backbone(model: torch.nn.Module) -> list[str]:
    """Freeze the shared per-finger/fusion trunk while leaving its output trainable."""
    encoder = getattr(model, "tactile_encoder", None)
    if encoder is None:
        raise ValueError("Tactile backbone freezing requires tactile_encoder")
    frozen = []
    for name, parameter in encoder.named_parameters():
        if not name.startswith("output."):
            parameter.requires_grad_(False)
            frozen.append(name)
    if not frozen:
        raise ValueError("No tactile backbone parameters were frozen")
    return frozen


def validate_resume_contract(
    checkpoint: dict[str, Any],
    dataset_config: dict[str, Any],
    source_data: list[str],
) -> None:
    """Reject resumes that silently change data selection or time semantics."""
    saved_config = checkpoint.get("dataset_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Resume checkpoint has no dataset_config")
    # v3-v5 proprio-only checkpoints predate explicit state-schema metadata.
    # Treat only a verified 78-D model as the legacy proprio contract; an
    # untagged 83-D checkpoint must never be silently interpreted as tactile.
    legacy_state_contract: dict[str, Any] = {}
    if (
        "state_schema" not in saved_config
        and checkpoint.get("model_config", {}).get("state_dim") == 78
    ):
        legacy_state_contract = {
            "state_schema": "xhand_proprio_v1",
            "state_dim": 78,
            "include_tactile": False,
            "tactile_names": [],
            "tactile_arch": "none",
        }
    for key, value in dataset_config.items():
        if key == "tactile_arch" and key not in saved_config:
            saved_value = (
                "direct_v1" if saved_config.get("include_tactile") is True else
                legacy_state_contract.get(key)
            )
        else:
            saved_value = saved_config.get(key, legacy_state_contract.get(key))
        if saved_value != value:
            raise ValueError(
                f"Resume dataset_config mismatch for {key}: "
                f"checkpoint={saved_value!r}, current={value!r}"
            )
    saved_sources = checkpoint.get("source_data")
    if saved_sources != source_data:
        raise ValueError(
            "Resume source_data differs from the current --data list; "
            "use the same ordered rollout shards"
        )
