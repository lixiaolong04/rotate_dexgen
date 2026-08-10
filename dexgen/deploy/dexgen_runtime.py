"""Simulator-free DexGen and RL-command inference for one real XHand."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import torch

from dexgen.data import (
    FeatureNormalizer,
    validate_checkpoint_representations,
    validate_checkpoint_state_contracts,
)
from dexgen.models import ConditionalMotionDiffusion, InverseDynamicsModel
from dexgen.modes import rotation_mode
from dexgen.xhand_keypoints import xhand_keypoints_fk
from rl_isaaclab.deploy_xhand.policy_runner import PolicyRunner


SUPPORTED_DIFFUSION_FORMATS = {
    "xhand_dexgen_diffusion_v3",
    "xhand_dexgen_diffusion_v4",
    "xhand_dexgen_diffusion_v5",
    "xhand_dexgen_diffusion_tactile_v1",
    "xhand_dexgen_diffusion_tactile_v2b",
    "xhand_dexgen_diffusion_tactile_film_v3",
}
SUPPORTED_IDM_FORMATS = {
    "xhand_dexgen_idm_v3",
    "xhand_dexgen_idm_v4",
    "xhand_dexgen_idm_v5",
    "xhand_dexgen_idm_tactile_v1",
    "xhand_dexgen_idm_tactile_v2b",
}


class _PolicyShapeEnv:
    """The policy loader only needs these two tensor shapes."""

    observation_space = torch.zeros(1, 141)
    action_space = torch.zeros(1, 21)
    prop_hist_len = 30
    num_obs = 141


class RLCommandRuntime:
    """Rebuild the proven 141-D/30x47 policy observation from remote hardware."""

    def __init__(
        self,
        *,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        cw_checkpoint: str | Path,
        ccw_checkpoint: str | Path,
        device: torch.device,
    ) -> None:
        self.device = device
        self.lower = joint_lower.to(device=device, dtype=torch.float32).reshape(1, 21)
        self.upper = joint_upper.to(device=device, dtype=torch.float32).reshape(1, 21)
        shape_env = _PolicyShapeEnv()
        self.runners = {
            "cw": PolicyRunner(shape_env, str(cw_checkpoint), device=str(device)),
            "ccw": PolicyRunner(shape_env, str(ccw_checkpoint), device=str(device)),
        }
        self.frames: deque[torch.Tensor] = deque(maxlen=30)

    def reset(self) -> None:
        self.frames.clear()

    def update(
        self,
        joint_pos: torch.Tensor,
        joint_target: torch.Tensor,
        tactile: torch.Tensor,
    ) -> None:
        q = joint_pos.to(self.device).reshape(1, 21)
        target = joint_target.to(self.device).reshape(1, 21)
        contact = tactile.to(self.device).reshape(1, 5).clone()
        # DexGen v3 was collected with the raw 9402/9503 expert contract:
        # all five tactile channels remain enabled.
        normalized_q = (2.0 * q - self.upper - self.lower) / (
            self.upper - self.lower
        )
        frame = torch.cat((normalized_q, target, contact), dim=-1)
        if not self.frames:
            for _ in range(30):
                self.frames.append(frame.clone())
        else:
            self.frames.append(frame)

    def action(self, mode: str) -> torch.Tensor:
        if len(self.frames) < 30:
            raise RuntimeError("RL observation history is not initialized")
        history = torch.stack(tuple(self.frames), dim=1)
        obs = torch.cat(tuple(self.frames)[-3:], dim=-1)
        return self.runners[rotation_mode(mode).name].act(
            {"obs": obs, "policy": obs, "proprio_hist": history}
        )


class DexGenRuntime:
    def __init__(
        self,
        diffusion_path: str | Path,
        idm_path: str | Path,
        *,
        joint_names: list[str],
        device: torch.device,
        guidance_strength: float = 50_000.0,
        ddim_steps: int = 25,
        seed: int = 2026,
    ) -> None:
        self.device = device
        self.joint_names = tuple(joint_names)
        self.guidance_strength = float(guidance_strength)
        self.ddim_steps = int(ddim_steps)
        diffusion_ckpt = torch.load(
            diffusion_path, map_location=device, weights_only=False
        )
        idm_ckpt = torch.load(idm_path, map_location=device, weights_only=False)
        if diffusion_ckpt.get("format") not in SUPPORTED_DIFFUSION_FORMATS:
            raise ValueError(
                f"unsupported diffusion format: {diffusion_ckpt.get('format')!r}"
            )
        if idm_ckpt.get("format") not in SUPPORTED_IDM_FORMATS:
            raise ValueError(f"unsupported IDM format: {idm_ckpt.get('format')!r}")
        validate_checkpoint_representations(diffusion_ckpt, idm_ckpt)
        contract = validate_checkpoint_state_contracts(diffusion_ckpt, idm_ckpt)
        if not contract["include_tactile"] or int(contract["state_dim"]) != 83:
            raise ValueError("real deployment requires the 83-D tactile DexGen contract")
        representation_names = tuple(
            diffusion_ckpt.get("representation", {}).get("joint_names", ())
        )
        if representation_names and representation_names != self.joint_names:
            raise ValueError("DexGen checkpoint joint order differs from deployment order")
        self.diffusion = ConditionalMotionDiffusion(
            **diffusion_ckpt["model_config"]
        ).to(device)
        self.diffusion.load_state_dict(diffusion_ckpt["model"])
        self.diffusion.eval()
        self.idm = InverseDynamicsModel(**idm_ckpt["model_config"]).to(device)
        self.idm.load_state_dict(idm_ckpt["model"])
        self.idm.eval()
        self.diffusion_norm = FeatureNormalizer.from_state_dict(
            diffusion_ckpt["normalizer"]
        )
        self.idm_norm = FeatureNormalizer.from_state_dict(idm_ckpt["normalizer"])
        self.history_steps = int(diffusion_ckpt["dataset_config"]["history_steps"])
        self.history: deque[torch.Tensor] = deque(maxlen=self.history_steps)
        self.generator = torch.Generator(device=device).manual_seed(int(seed))

    @property
    def ready(self) -> bool:
        return len(self.history) == self.history_steps

    def reset(self) -> None:
        self.history.clear()

    @torch.inference_mode()
    def warmup(self) -> None:
        """Initialize CUDA kernels before the hardware control loop starts."""
        self.reset()
        q = torch.zeros(1, 21, device=self.device)
        tactile = torch.zeros(1, 5, device=self.device)
        for _ in range(self.history_steps):
            self.append_state(q, q, tactile)
        self.sample_target(
            torch.zeros(1, 10, 3, device=self.device), "cw"
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reset()

    def append_state(
        self,
        joint_pos: torch.Tensor,
        joint_target: torch.Tensor,
        tactile: torch.Tensor,
    ) -> torch.Tensor:
        q = joint_pos.to(self.device).reshape(1, 21)
        target = joint_target.to(self.device).reshape(1, 21)
        contact = tactile.to(self.device).reshape(1, 5)
        keypoints = xhand_keypoints_fk(q, self.joint_names)
        fingertips = keypoints[:, 1::2].flatten(1)
        frame = torch.cat((fingertips, q, target, target - q, contact), dim=-1)
        if frame.shape != (1, 83) or not bool(torch.isfinite(frame).all()):
            raise RuntimeError(f"invalid DexGen state frame: {tuple(frame.shape)}")
        self.history.append(frame)
        return frame

    @torch.inference_mode()
    def sample_target(self, command_motion: torch.Tensor, mode: str) -> torch.Tensor:
        if not self.ready:
            raise RuntimeError(
                f"DexGen history not ready: {len(self.history)}/{self.history_steps}"
            )
        history = torch.stack(tuple(self.history), dim=1)
        command = command_motion.to(self.device, dtype=torch.float32)
        if command.shape not in {(1, 10, 3), (1, 2, 10, 3)}:
            raise ValueError(f"invalid command shape: {tuple(command.shape)}")
        mode_ids = torch.tensor(
            [rotation_mode(mode).mode_id], dtype=torch.long, device=self.device
        )
        motion_z = self.diffusion.guided_sample(
            self.diffusion_norm.normalize_state(history),
            command_motion=command,
            guidance_strength=self.guidance_strength,
            motion_mean=self.diffusion_norm.motion_mean,
            motion_std=self.diffusion_norm.motion_std,
            ddim_steps=self.ddim_steps,
            mode=mode_ids,
            generator=self.generator,
        )
        motion = self.diffusion_norm.denormalize_motion(motion_z)
        target_z = self.idm.act(
            self.idm_norm.normalize_state(history),
            self.idm_norm.normalize_motion(motion),
        )
        return self.idm_norm.denormalize_target(target_z)
