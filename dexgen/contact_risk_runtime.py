"""Runtime loading and scoring for frozen-v3 contact-risk reranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dexgen.data import FeatureNormalizer, TACTILE_DIM
from dexgen.risk_models import CandidateContactRiskHead, TactileProprioTConv


CONTACT_RISK_STUDENT_FORMAT = "xhand_dexgen_contact_risk_student_v9"


@torch.no_grad()
def sample_motion_candidates(
    diffusion: torch.nn.Module,
    state: torch.Tensor,
    mode: torch.Tensor,
    *,
    candidate_count: int,
    ddim_steps: int,
    baseline_generator: torch.Generator,
    extra_generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``[B,C,T,K,3]`` while preserving candidate zero's RNG stream.

    The first candidate always consumes exactly the same noise tensor from
    ``baseline_generator`` as the original one-candidate v3 path.  Additional
    candidates use an independent generator, so adding reranking cannot advance
    the baseline random stream.
    """
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if state.ndim != 3 or mode.shape != (state.shape[0],):
        raise ValueError("Expected state [B,H,F] and mode [B]")
    batch = state.shape[0]
    motion_shape = (
        diffusion.prediction_steps,
        diffusion.keypoint_count,
        3,
    )
    baseline_noise = torch.randn(
        (batch, *motion_shape),
        device=state.device,
        dtype=state.dtype,
        generator=baseline_generator,
    )
    if candidate_count > 1:
        extra_noise = torch.randn(
            (batch, candidate_count - 1, *motion_shape),
            device=state.device,
            dtype=state.dtype,
            generator=extra_generator,
        )
        noise = torch.cat((baseline_noise[:, None], extra_noise), dim=1)
    else:
        noise = baseline_noise[:, None]
    repeated_state = state[:, None].expand(
        batch, candidate_count, *state.shape[1:]
    ).reshape(batch * candidate_count, *state.shape[1:])
    repeated_mode = mode[:, None].expand(batch, candidate_count).reshape(-1)
    return diffusion.sample(
        repeated_state,
        ddim_steps=ddim_steps,
        mode=repeated_mode,
        initial_noise=noise.reshape(batch * candidate_count, *motion_shape),
    ).view(batch, candidate_count, *motion_shape)


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and torch.equal(
        left.detach().cpu(), right.detach().cpu()
    )


def load_contact_risk_runtime(
    path: Path,
    device: torch.device,
    *,
    base_normalizer: FeatureNormalizer,
    state_dim: int,
    history_steps: int,
    prediction_steps: int,
    keypoint_count: int,
    num_modes: int,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CONTACT_RISK_STUDENT_FORMAT:
        raise ValueError(
            f"Unsupported contact-risk checkpoint: {checkpoint.get('format')!r}"
        )
    dataset_config = checkpoint.get("dataset_config", {})
    if dataset_config.get("recovery_label_mode") != "local":
        raise ValueError("Online reranking requires a candidate-local risk checkpoint")
    if int(dataset_config.get("adapter_hz", 0)) != 20:
        raise ValueError("Contact-risk adapter must use the proven 20 Hz history")
    if int(dataset_config.get("generator_hz", 0)) != 10:
        raise ValueError("Contact-risk checkpoint must target the 10 Hz v3 generator")

    risk_normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    for name in FeatureNormalizer.__dataclass_fields__:
        if not _same_tensor(
            getattr(risk_normalizer, name), getattr(base_normalizer, name)
        ):
            raise ValueError(
                f"Contact-risk and v3 normalizers differ in {name}"
            )

    student_config = checkpoint["student_config"]
    head_config = checkpoint["risk_head_config"]
    expected = {
        "state_dim": state_dim,
        "policy_history_steps": history_steps,
        "prediction_steps": prediction_steps,
        "keypoint_count": keypoint_count,
        "num_modes": num_modes,
    }
    actual = {
        "state_dim": int(student_config["state_dim"]),
        "policy_history_steps": int(head_config["policy_history_steps"]),
        "prediction_steps": int(head_config["prediction_steps"]),
        "keypoint_count": int(head_config["keypoint_count"]),
        "num_modes": int(head_config["num_modes"]),
    }
    if actual != expected:
        raise ValueError(
            f"Contact-risk/v3 representation mismatch: {actual} != {expected}"
        )
    if int(student_config["latent_dim"]) != int(head_config["latent_dim"]):
        raise ValueError("Contact-risk Student and head latent dimensions differ")

    student = TactileProprioTConv(**student_config).to(device)
    student.load_state_dict(checkpoint["student"])
    student.eval().requires_grad_(False)
    head = CandidateContactRiskHead(**head_config).to(device)
    head.load_state_dict(checkpoint["risk_head"])
    head.eval().requires_grad_(False)
    tactile_mask = tuple(
        sorted(set(int(value) for value in dataset_config["disable_tactile_ids"]))
    )
    return {
        "student": student,
        "head": head,
        "normalizer": risk_normalizer,
        "history_steps": int(student_config["history_steps"]),
        "state_dim": int(student_config["state_dim"]),
        "tactile_mask": tactile_mask,
        "checkpoint": str(path),
    }


def _masked_state(runtime: dict[str, Any], state: torch.Tensor) -> torch.Tensor:
    tactile_mask = runtime["tactile_mask"]
    if not tactile_mask:
        return state
    result = state.clone()
    indices = torch.tensor(
        tactile_mask, dtype=torch.long, device=result.device
    )
    result[..., indices - TACTILE_DIM] = 0.0
    return result


@torch.no_grad()
def score_motion_candidates(
    runtime: dict[str, Any],
    *,
    adapter_history: torch.Tensor,
    policy_history: torch.Tensor,
    motion_candidates: torch.Tensor,
    mode: torch.Tensor,
) -> torch.Tensor:
    """Return safe-contact logits for normalized candidate motions.

    Args:
        adapter_history: raw deployable state, ``[B,30,83]`` at 20 Hz.
        policy_history: raw v3 state, ``[B,4,83]`` at 10 Hz.
        motion_candidates: normalized v3 motions, ``[B,C,2,10,3]``.
        mode: rotation mode ids, ``[B]``.
    """
    if adapter_history.shape[1:] != (
        runtime["history_steps"],
        runtime["state_dim"],
    ):
        raise ValueError("Contact-risk adapter history has the wrong shape")
    if policy_history.ndim != 3 or policy_history.shape[0] != adapter_history.shape[0]:
        raise ValueError("Contact-risk policy history has the wrong shape")
    if motion_candidates.ndim != 5 or motion_candidates.shape[0] != adapter_history.shape[0]:
        raise ValueError("Contact-risk candidates must have shape [B,C,T,K,3]")
    batch, candidates = motion_candidates.shape[:2]
    if mode.shape != (batch,):
        raise ValueError("Contact-risk mode ids have the wrong shape")

    normalizer: FeatureNormalizer = runtime["normalizer"]
    adapter_state = normalizer.normalize_state(
        _masked_state(runtime, adapter_history)
    )
    policy_state = normalizer.normalize_state(
        _masked_state(runtime, policy_history)
    )
    latent = runtime["student"](adapter_state)
    repeated_policy = policy_state[:, None].expand(
        batch, candidates, *policy_state.shape[1:]
    ).reshape(batch * candidates, *policy_state.shape[1:])
    repeated_latent = latent[:, None].expand(
        batch, candidates, latent.shape[-1]
    ).reshape(batch * candidates, latent.shape[-1])
    repeated_mode = mode[:, None].expand(batch, candidates).reshape(-1)
    logits = runtime["head"](
        repeated_policy,
        motion_candidates.reshape(batch * candidates, *motion_candidates.shape[2:]),
        repeated_latent,
        repeated_mode,
    )
    return logits.view(batch, candidates)
