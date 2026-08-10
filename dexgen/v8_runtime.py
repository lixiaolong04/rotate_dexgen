"""Runtime helpers for teacher/student-latent DexGen checkpoints."""

from __future__ import annotations

from typing import Any

import torch

from dexgen.models import LatentStudentAdapter, PrivilegedLatentTeacher


V8_DIFFUSION_FORMAT = "xhand_dexgen_diffusion_teacher_student_v8"
V8_IDM_FORMAT = "xhand_dexgen_idm_teacher_student_v8"


def _same_state_dict(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        left[key].shape == right[key].shape
        and torch.equal(left[key].detach().cpu(), right[key].detach().cpu())
        for key in left
    )


def load_v8_runtime(
    diffusion_checkpoint: dict[str, Any],
    idm_checkpoint: dict[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    """Load the shared teacher/student pair, or return None for legacy models."""
    diffusion_latent_dim = int(
        diffusion_checkpoint.get("model_config", {}).get("env_latent_dim", 0)
    )
    idm_latent_dim = int(idm_checkpoint.get("model_config", {}).get("env_latent_dim", 0))
    if diffusion_latent_dim == idm_latent_dim == 0:
        return None
    if diffusion_checkpoint.get("format") != V8_DIFFUSION_FORMAT:
        raise ValueError("Latent-conditioned Diffusion checkpoint is not v8")
    if idm_checkpoint.get("format") != V8_IDM_FORMAT:
        raise ValueError("Latent-conditioned IDM checkpoint is not v8")
    if diffusion_latent_dim != idm_latent_dim:
        raise ValueError("Diffusion and IDM env latent dimensions differ")
    for key in (
        "student_config",
        "teacher_config",
        "priv_info_indices",
        "deployment_latent_source",
    ):
        if diffusion_checkpoint.get(key) != idm_checkpoint.get(key):
            raise ValueError(f"v8 Diffusion and IDM differ in {key}")
    if not _same_state_dict(
        diffusion_checkpoint["student"], idm_checkpoint["student"]
    ):
        raise ValueError("v8 Diffusion and IDM embed different student adapters")
    if not _same_state_dict(
        diffusion_checkpoint["teacher"], idm_checkpoint["teacher"]
    ):
        raise ValueError("v8 Diffusion and IDM embed different teacher encoders")
    for key in ("priv_mean", "priv_std"):
        if not torch.equal(
            diffusion_checkpoint[key].detach().cpu(),
            idm_checkpoint[key].detach().cpu(),
        ):
            raise ValueError(f"v8 Diffusion and IDM differ in {key}")

    student = LatentStudentAdapter(**diffusion_checkpoint["student_config"]).to(device)
    student.load_state_dict(diffusion_checkpoint["student"])
    student.eval().requires_grad_(False)
    teacher = PrivilegedLatentTeacher(**diffusion_checkpoint["teacher_config"]).to(device)
    teacher.load_state_dict(diffusion_checkpoint["teacher"])
    teacher.eval().requires_grad_(False)
    return {
        "student": student,
        "teacher": teacher,
        "priv_info_indices": list(diffusion_checkpoint["priv_info_indices"]),
        "priv_mean": diffusion_checkpoint["priv_mean"].to(device),
        "priv_std": diffusion_checkpoint["priv_std"].to(device),
        "latent_dim": diffusion_latent_dim,
    }


def student_latent(runtime: dict[str, Any] | None, normalized_state: torch.Tensor):
    if runtime is None:
        return None
    return runtime["student"](normalized_state)


def teacher_latent(runtime: dict[str, Any], priv_info: torch.Tensor):
    indices = runtime["priv_info_indices"]
    normalized = (
        priv_info[:, indices].to(runtime["priv_mean"]) - runtime["priv_mean"]
    ) / runtime["priv_std"]
    return runtime["teacher"](normalized)
