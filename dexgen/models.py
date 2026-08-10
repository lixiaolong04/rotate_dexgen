"""Neural models used by the minimal DexterityGen reproduction."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device))
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.condition = nn.Linear(condition_dim, hidden_dim * 2)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm(value) * (1.0 + scale) + shift
        hidden = self.linear2(F.silu(self.linear1(F.silu(hidden))))
        return value + hidden


class IndependentTactileFiLMResidualBlock(nn.Module):
    """Apply base and tactile FiLM transforms through separate learned paths."""

    def __init__(self, hidden_dim: int, condition_dim: int, tactile_condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.base_condition = nn.Linear(condition_dim, hidden_dim * 2)
        self.tactile_condition = nn.Linear(tactile_condition_dim, hidden_dim * 2)
        self.tactile_gate = nn.Linear(condition_dim + tactile_condition_dim, hidden_dim)
        # Start as the paper-style three-condition model, then learn only the
        # tactile correction justified by the recovery examples.
        nn.init.zeros_(self.tactile_condition.weight)
        nn.init.zeros_(self.tactile_condition.bias)
        nn.init.zeros_(self.tactile_gate.weight)
        nn.init.constant_(self.tactile_gate.bias, -2.0)

    def forward(
        self,
        value: torch.Tensor,
        base_condition: torch.Tensor,
        tactile_condition: torch.Tensor,
    ) -> torch.Tensor:
        base_scale, base_shift = self.base_condition(base_condition).chunk(2, dim=-1)
        touch_scale, touch_shift = self.tactile_condition(tactile_condition).chunk(2, dim=-1)
        gate = torch.sigmoid(
            self.tactile_gate(torch.cat((base_condition, tactile_condition), dim=-1))
        )
        scale = base_scale + gate * touch_scale
        shift = base_shift + gate * touch_shift
        hidden = self.norm(value) * (1.0 + scale) + shift
        hidden = self.linear2(F.silu(self.linear1(F.silu(hidden))))
        return value + hidden


class TactileHistoryEncoder(nn.Module):
    """Encode five scalar force histories with shared per-finger dynamics.

    The raw force, temporal difference, and a threshold-derived contact mask are
    encoded independently for each finger before a small cross-finger fusion
    MLP.  Finger embeddings preserve the asymmetric role of the thumb while the
    shared temporal MLP keeps the parameter count and inductive bias modest.
    """

    def __init__(
        self,
        history_steps: int,
        tactile_dim: int,
        output_dim: int,
        per_finger_dim: int = 32,
        embedding_dim: int = 128,
        contact_threshold_normalized: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    ):
        super().__init__()
        if history_steps < 2 or tactile_dim < 1:
            raise ValueError("TactileHistoryEncoder requires at least two history frames")
        self.history_steps = history_steps
        self.tactile_dim = tactile_dim
        self.per_finger_dim = per_finger_dim
        self.embedding_dim = embedding_dim
        if contact_threshold_normalized is None:
            threshold = torch.zeros(tactile_dim)
        else:
            threshold = torch.as_tensor(contact_threshold_normalized, dtype=torch.float32)
        if threshold.shape != (tactile_dim,):
            raise ValueError(
                f"contact threshold must have shape ({tactile_dim},), got {tuple(threshold.shape)}"
            )
        self.register_buffer("contact_threshold_normalized", threshold)
        per_finger_input = history_steps + (history_steps - 1) + history_steps
        self.per_finger = nn.Sequential(
            nn.Linear(per_finger_input, per_finger_dim),
            nn.SiLU(),
            nn.LayerNorm(per_finger_dim),
            nn.Linear(per_finger_dim, per_finger_dim),
            nn.SiLU(),
        )
        self.finger_embedding = nn.Parameter(torch.empty(tactile_dim, per_finger_dim))
        nn.init.normal_(self.finger_embedding, std=0.02)
        self.fuse = nn.Sequential(
            nn.Linear(tactile_dim * per_finger_dim, embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(embedding_dim, output_dim)

    def forward(self, tactile_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.history_steps, self.tactile_dim)
        if tactile_history.ndim != 3 or tuple(tactile_history.shape[1:]) != expected:
            raise ValueError(
                f"tactile history must have shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(tactile_history.shape)}"
            )
        force = tactile_history.transpose(1, 2)
        delta = force[:, :, 1:] - force[:, :, :-1]
        threshold = self.contact_threshold_normalized[None, :, None].to(force)
        contact = (force > threshold).to(force.dtype)
        per_finger = self.per_finger(torch.cat((force, delta, contact), dim=-1))
        per_finger = per_finger + self.finger_embedding[None].to(per_finger)
        embedding = self.fuse(per_finger.flatten(1))
        return self.output(embedding), embedding


class RMAAdapter(nn.Module):
    """Predict simulator-only object/contact parameters from deployable history.

    The adapter sees exactly the 83-D DexGen tactile state at deployment.  Its
    target is never appended to the robot observation; `priv_info` is used only
    as supervised training signal, following the RMA teacher/student pattern.
    """

    def __init__(
        self,
        *,
        history_steps: int,
        state_dim: int,
        priv_info_dim: int,
        proprio_state_dim: int = 78,
        tactile_dim: int = 5,
        hidden_dim: int = 256,
        tactile_per_finger_dim: int = 32,
        tactile_embedding_dim: int = 128,
        tactile_contact_threshold_normalized: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        if state_dim != proprio_state_dim + tactile_dim:
            raise ValueError("RMAAdapter requires proprio + tactile state layout")
        self.history_steps = history_steps
        self.state_dim = state_dim
        self.priv_info_dim = priv_info_dim
        self.proprio_state_dim = proprio_state_dim
        self.tactile_dim = tactile_dim
        self.tactile_encoder = TactileHistoryEncoder(
            history_steps, tactile_dim, hidden_dim, tactile_per_finger_dim,
            tactile_embedding_dim, tactile_contact_threshold_normalized,
        )
        self.proprio = nn.Sequential(
            nn.Linear(history_steps * proprio_state_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            ResidualMLPBlock(hidden_dim), ResidualMLPBlock(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, priv_info_dim)

    def config(self) -> dict[str, int | list[float]]:
        return {
            "history_steps": self.history_steps,
            "state_dim": self.state_dim,
            "priv_info_dim": self.priv_info_dim,
            "proprio_state_dim": self.proprio_state_dim,
            "tactile_dim": self.tactile_dim,
            "hidden_dim": self.output.in_features,
            "tactile_per_finger_dim": self.tactile_encoder.per_finger_dim,
            "tactile_embedding_dim": self.tactile_encoder.embedding_dim,
            "tactile_contact_threshold_normalized": self.tactile_encoder.contact_threshold_normalized.detach().cpu().tolist(),
        }

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.history_steps, self.state_dim):
            raise ValueError(f"Expected [B,{self.history_steps},{self.state_dim}], got {tuple(state.shape)}")
        proprio = self.proprio(state[..., :self.proprio_state_dim].flatten(1))
        tactile_hidden, _ = self.tactile_encoder(state[..., self.proprio_state_dim:])
        return self.output(self.fuse(torch.cat((proprio, tactile_hidden), dim=-1)))


class PrivilegedLatentTeacher(nn.Module):
    """Encode normalized simulator-only parameters into a learned latent."""

    def __init__(
        self,
        priv_info_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if min(priv_info_dim, latent_dim, hidden_dim) < 1:
            raise ValueError("teacher dimensions must be positive")
        self.priv_info_dim = priv_info_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(priv_info_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, priv_info_dim),
        )

    def forward(self, priv_info: torch.Tensor) -> torch.Tensor:
        return self.encoder(priv_info)

    def reconstruct(self, priv_info: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self(priv_info)
        return latent, self.decoder(latent)

    def config(self) -> dict[str, int]:
        return {
            "priv_info_dim": self.priv_info_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }


class LatentStudentAdapter(nn.Module):
    """Predict a teacher environment latent from deployable state/touch history."""

    def __init__(
        self,
        *,
        history_steps: int,
        state_dim: int,
        latent_dim: int,
        proprio_state_dim: int = 78,
        tactile_dim: int = 5,
        hidden_dim: int = 256,
        tactile_per_finger_dim: int = 32,
        tactile_embedding_dim: int = 128,
        tactile_contact_threshold_normalized: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.history_steps = history_steps
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.proprio_state_dim = proprio_state_dim
        self.tactile_dim = tactile_dim
        self.hidden_dim = hidden_dim
        self.tactile_per_finger_dim = tactile_per_finger_dim
        self.tactile_embedding_dim = tactile_embedding_dim
        if state_dim != proprio_state_dim + tactile_dim:
            raise ValueError("LatentStudentAdapter requires proprio + tactile state layout")
        self.tactile_encoder = TactileHistoryEncoder(
            history_steps,
            tactile_dim,
            hidden_dim,
            tactile_per_finger_dim,
            tactile_embedding_dim,
            tactile_contact_threshold_normalized,
        )
        self.proprio = nn.Sequential(
            nn.Linear(history_steps * proprio_state_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            ResidualMLPBlock(hidden_dim),
            ResidualMLPBlock(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        expected = (self.history_steps, self.state_dim)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]}], got {tuple(state.shape)}")
        proprio = self.proprio(state[..., : self.proprio_state_dim].flatten(1))
        tactile, _ = self.tactile_encoder(state[..., self.proprio_state_dim :])
        return self.output(self.fuse(torch.cat((proprio, tactile), dim=-1)))

    def config(self) -> dict[str, int | list[float]]:
        return {
            "history_steps": self.history_steps,
            "state_dim": self.state_dim,
            "latent_dim": self.latent_dim,
            "proprio_state_dim": self.proprio_state_dim,
            "tactile_dim": self.tactile_dim,
            "hidden_dim": self.hidden_dim,
            "tactile_per_finger_dim": self.tactile_per_finger_dim,
            "tactile_embedding_dim": self.tactile_embedding_dim,
            "tactile_contact_threshold_normalized": (
                self.tactile_encoder.contact_threshold_normalized.detach().cpu().tolist()
            ),
        }


class ConditionalMotionDiffusion(nn.Module):
    """Fully-connected U-Net-style DDPM model with FiLM conditioning.

    The paper replaces the 1-D convolutions in a three-block U-Net with fully
    connected layers.  This implementation keeps that design intent while using
    residual FiLM blocks, which makes the small single-object reproduction stable.
    """

    def __init__(
        self,
        state_dim: int,
        history_steps: int,
        keypoint_count: int,
        prediction_steps: int = 2,
        hidden_dim: int = 768,
        state_hidden_dim: int = 1024,
        state_layers: int = 6,
        num_blocks: int = 3,
        time_dim: int = 128,
        diffusion_steps: int = 100,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        noise_schedule: str = "linear",
        num_modes: int = 1,
        norm_groups: int = 8,
        prediction_type: str = "x0",
        tactile_arch: str = "direct_v1",
        proprio_state_dim: int = 78,
        tactile_dim: int = 5,
        tactile_per_finger_dim: int = 32,
        tactile_embedding_dim: int = 128,
        tactile_contact_threshold_normalized: list[float] | tuple[float, ...] | None = None,
        tactile_condition_dropout: float = 0.0,
        env_latent_dim: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.history_steps = history_steps
        self.keypoint_count = keypoint_count
        self.prediction_steps = prediction_steps
        self.motion_dim = prediction_steps * keypoint_count * 3
        self.diffusion_steps = diffusion_steps
        self.time_dim = time_dim
        self.num_modes = num_modes
        self.norm_groups = norm_groups
        if tactile_arch not in {"none", "direct_v1", "v2b", "film_v3"}:
            raise ValueError(f"Unsupported tactile architecture: {tactile_arch!r}")
        self.tactile_arch = tactile_arch
        self.proprio_state_dim = proprio_state_dim
        self.tactile_dim = tactile_dim
        self.tactile_per_finger_dim = tactile_per_finger_dim
        self.tactile_embedding_dim = tactile_embedding_dim
        self.tactile_condition_dropout = tactile_condition_dropout
        self.env_latent_dim = env_latent_dim
        if env_latent_dim < 0:
            raise ValueError("env_latent_dim must be non-negative")
        if tactile_arch in {"v2b", "film_v3"} and state_dim != proprio_state_dim + tactile_dim:
            raise ValueError(f"{tactile_arch} state_dim must equal proprio_state_dim + tactile_dim")
        if not 0.0 <= tactile_condition_dropout < 1.0:
            raise ValueError("tactile_condition_dropout must be in [0, 1)")
        if noise_schedule not in {"linear", "cosine"}:
            raise ValueError("noise_schedule must be 'linear' or 'cosine'")
        self.noise_schedule = noise_schedule
        if prediction_type not in {"x0", "epsilon"}:
            raise ValueError("prediction_type must be 'x0' or 'epsilon'")
        self.prediction_type = prediction_type
        if state_hidden_dim % norm_groups != 0:
            raise ValueError("state_hidden_dim must be divisible by norm_groups")

        state_modules: list[nn.Module] = []
        encoded_state_dim = proprio_state_dim if tactile_arch in {"v2b", "film_v3"} else state_dim
        input_dim = encoded_state_dim * history_steps
        for _ in range(state_layers):
            state_modules.extend(
                (nn.Linear(input_dim, state_hidden_dim), nn.SiLU(), nn.GroupNorm(norm_groups, state_hidden_dim))
            )
            input_dim = state_hidden_dim
        state_modules.append(nn.Linear(input_dim, hidden_dim))
        self.state_encoder = nn.Sequential(*state_modules)
        if tactile_arch in {"v2b", "film_v3"}:
            self.tactile_encoder = TactileHistoryEncoder(
                history_steps=history_steps,
                tactile_dim=tactile_dim,
                output_dim=hidden_dim,
                per_finger_dim=tactile_per_finger_dim,
                embedding_dim=tactile_embedding_dim,
                contact_threshold_normalized=tactile_contact_threshold_normalized,
            )
            self.tactile_gate = (
                nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
                if tactile_arch == "v2b" else None
            )
            self.future_tactile_head = nn.Linear(
                tactile_embedding_dim, prediction_steps * tactile_dim
            )
        else:
            self.tactile_encoder = None
            self.tactile_gate = None
            self.future_tactile_head = None
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # A single fixed rotation mode is used now.  Keeping the embedding in the
        # architecture makes later multi-primitive training checkpoint-compatible.
        self.mode_embedding = nn.Embedding(num_modes, hidden_dim)
        self.env_latent_projection = (
            nn.Sequential(
                nn.LayerNorm(env_latent_dim),
                nn.Linear(env_latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if env_latent_dim else None
        )
        self.input_projection = nn.Linear(self.motion_dim, hidden_dim)
        block_type = IndependentTactileFiLMResidualBlock if tactile_arch == "film_v3" else FiLMResidualBlock
        if tactile_arch == "film_v3":
            self.encoder = nn.ModuleList(block_type(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_blocks))
            self.middle = block_type(hidden_dim, hidden_dim, hidden_dim)
            self.decoder = nn.ModuleList(block_type(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_blocks))
        else:
            self.encoder = nn.ModuleList(block_type(hidden_dim, hidden_dim) for _ in range(num_blocks))
            self.middle = block_type(hidden_dim, hidden_dim)
            self.decoder = nn.ModuleList(block_type(hidden_dim, hidden_dim) for _ in range(num_blocks))
        self.skip_projection = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_blocks))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.motion_dim))

        if noise_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        else:
            # A 100-step linear 1e-4..2e-2 schedule retains alpha_bar_T ~= 0.36,
            # so its highest-noise training input does not match inference from
            # pure Gaussian noise.  The cosine schedule reaches near-zero
            # terminal SNR and restores that training/sampling contract.
            steps = torch.arange(diffusion_steps + 1, dtype=torch.float64)
            alpha_bar = torch.cos(
                ((steps / diffusion_steps + 0.008) / 1.008) * torch.pi / 2.0
            ).square()
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def config(self) -> dict[str, int | float | str | list[float]]:
        first_state_linear = next(module for module in self.state_encoder if isinstance(module, nn.Linear))
        state_hidden_dim = first_state_linear.out_features
        state_layers = sum(isinstance(module, nn.GroupNorm) for module in self.state_encoder)
        config: dict[str, int | float | str | list[float]] = {
            "state_dim": self.state_dim,
            "history_steps": self.history_steps,
            "keypoint_count": self.keypoint_count,
            "prediction_steps": self.prediction_steps,
            "hidden_dim": self.input_projection.out_features,
            "state_hidden_dim": state_hidden_dim,
            "state_layers": state_layers,
            "num_blocks": len(self.encoder),
            "time_dim": self.time_dim,
            "diffusion_steps": self.diffusion_steps,
            "beta_start": float(self.betas[0]),
            "beta_end": float(self.betas[-1]),
            "noise_schedule": self.noise_schedule,
            "num_modes": self.num_modes,
            "norm_groups": self.norm_groups,
            "prediction_type": self.prediction_type,
            "tactile_arch": self.tactile_arch,
            "proprio_state_dim": self.proprio_state_dim,
            "tactile_dim": self.tactile_dim,
            "tactile_per_finger_dim": self.tactile_per_finger_dim,
            "tactile_embedding_dim": self.tactile_embedding_dim,
            "tactile_condition_dropout": self.tactile_condition_dropout,
            "env_latent_dim": self.env_latent_dim,
        }
        if self.tactile_encoder is not None:
            config["tactile_contact_threshold_normalized"] = (
                self.tactile_encoder.contact_threshold_normalized.detach().cpu().tolist()
            )
        return config

    def _encode_state(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch = state.shape[0]
        if self.tactile_arch not in {"v2b", "film_v3"}:
            return self.state_encoder(state.reshape(batch, -1)), None, None
        proprio = state[..., : self.proprio_state_dim]
        tactile = state[..., self.proprio_state_dim :]
        proprio_embedding = self.state_encoder(proprio.reshape(batch, -1))
        assert self.tactile_encoder is not None
        tactile_condition, tactile_embedding = self.tactile_encoder(tactile)
        if self.tactile_arch == "film_v3":
            if self.training and self.tactile_condition_dropout > 0.0:
                # Modality dropout removes the complete tactile condition for a
                # sample, so proprioception remains a usable fallback at runtime.
                keep = F.dropout(
                    torch.ones(
                        (batch, 1), device=tactile_condition.device,
                        dtype=tactile_condition.dtype,
                    ),
                    p=self.tactile_condition_dropout,
                    training=True,
                )
                tactile_condition = tactile_condition * keep
            return proprio_embedding, tactile_condition, tactile_embedding
        assert self.tactile_gate is not None
        gate = self.tactile_gate(torch.cat((proprio_embedding, tactile_condition), dim=-1))
        return proprio_embedding + gate * tactile_condition, None, tactile_embedding

    def forward(
        self,
        noisy_motion: torch.Tensor,
        timesteps: torch.Tensor,
        state: torch.Tensor,
        mode: torch.Tensor | None = None,
        *,
        env_latent: torch.Tensor | None = None,
        return_tactile_prediction: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = noisy_motion.shape[0]
        if mode is None:
            mode = torch.zeros(batch, device=noisy_motion.device, dtype=torch.long)
        value = self.input_projection(noisy_motion.reshape(batch, -1))
        condition, tactile_condition, tactile_embedding = self._encode_state(state)
        condition = condition + self.time_encoder(sinusoidal_embedding(timesteps, self.time_dim))
        condition = condition + self.mode_embedding(mode)
        if self.env_latent_projection is not None:
            if env_latent is None or env_latent.shape != (batch, self.env_latent_dim):
                raise ValueError(
                    f"env_latent must have shape [B,{self.env_latent_dim}]"
                )
            condition = condition + self.env_latent_projection(env_latent)
        skips = []
        for block in self.encoder:
            value = (
                block(value, condition, tactile_condition)
                if tactile_condition is not None else block(value, condition)
            )
            skips.append(value)
        value = (
            self.middle(value, condition, tactile_condition)
            if tactile_condition is not None else self.middle(value, condition)
        )
        for block, projection, skip in zip(self.decoder, self.skip_projection, reversed(skips)):
            value = projection(torch.cat((value, skip), dim=-1))
            value = (
                block(value, condition, tactile_condition)
                if tactile_condition is not None else block(value, condition)
            )
        motion = self.output(value).view(batch, self.prediction_steps, self.keypoint_count, 3)
        if not return_tactile_prediction:
            return motion
        if tactile_embedding is None or self.future_tactile_head is None:
            raise ValueError("Future tactile prediction requires structured tactile conditioning")
        tactile = self.future_tactile_head(tactile_embedding).view(
            batch, self.prediction_steps, self.tactile_dim
        )
        return motion, tactile

    def training_loss_components(
        self,
        motion: torch.Tensor,
        state: torch.Tensor,
        mode: torch.Tensor | None = None,
        *,
        future_tactile: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        env_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return per-sample diffusion and optional future-touch losses."""
        batch = motion.shape[0]
        if timesteps is None:
            timesteps = torch.randint(self.diffusion_steps, (batch,), device=motion.device)
        if noise is None:
            noise = torch.randn_like(motion)
        alpha_bar = self.alphas_cumprod[timesteps].view(batch, 1, 1, 1)
        noisy = alpha_bar.sqrt() * motion + (1.0 - alpha_bar).sqrt() * noise
        if future_tactile is None:
            prediction = self(noisy, timesteps, state, mode, env_latent=env_latent)
            tactile_prediction = None
        else:
            prediction, tactile_prediction = self(
                noisy, timesteps, state, mode, env_latent=env_latent,
                return_tactile_prediction=True
            )
        target = motion if self.prediction_type == "x0" else noise
        motion_loss = F.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
        tactile_loss = None
        if future_tactile is not None:
            tactile_loss = F.smooth_l1_loss(
                tactile_prediction, future_tactile, reduction="none"
            ).flatten(1).mean(1)
        return motion_loss, tactile_loss

    def training_loss(
        self,
        motion: torch.Tensor,
        state: torch.Tensor,
        mode: torch.Tensor | None = None,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        future_tactile: torch.Tensor | None = None,
        tactile_aux_weight: float = 0.0,
        env_latent: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        per_sample, tactile_loss = self.training_loss_components(
            motion,
            state,
            mode,
            future_tactile=future_tactile,
            timesteps=timesteps,
            noise=noise,
            env_latent=env_latent,
        )
        if tactile_loss is not None:
            per_sample = per_sample + tactile_aux_weight * tactile_loss
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        if reduction == "sum":
            return per_sample.sum()
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        ddim_steps: int = 8,
        mode: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        env_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM sampling (eta=0), deterministic for a fixed initial latent."""
        return self._sample_impl(
            state, ddim_steps, mode=mode, initial_noise=initial_noise,
            generator=generator, env_latent=env_latent
        )

    @torch.no_grad()
    def guided_sample(
        self,
        state: torch.Tensor,
        command_motion: torch.Tensor,
        guidance_strength: float,
        motion_mean: torch.Tensor,
        motion_std: torch.Tensor,
        ddim_steps: int = 12,
        mode: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        env_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample with the paper's motion-distance gradient guidance.

        ``command_motion`` is a physical wrist-frame offset with shape [B,K,3]
        (the same command is compared with all T predictions) or [B,T,K,3].
        The diffusion variable ``z`` is normalized, while Eq. (3) is evaluated
        in metres:

            Dist = sum_i ||delta_x_i - delta_x_input||^2
            z0 <- z0 - guidance_strength * Sigma_z * grad_z0 Dist

        With ``delta_x = z0 * motion_std + motion_mean``, the chain rule gives
        ``grad_z0 Dist = 2 * (delta_x - delta_x_input) * motion_std``.  The
        reverse-process covariance is also in normalized diffusion space.  It
        would therefore be dimensionally incorrect to apply ``Sigma_z`` as a
        direct metre-space update.

        ``Sigma`` uses the effective posterior variance between consecutive
        subsampled DDIM timesteps.  DDIM remains deterministic after correction.
        """
        if guidance_strength < 0.0:
            raise ValueError("guidance_strength must be non-negative")
        if command_motion.ndim == 3:
            command_motion = command_motion[:, None]
        expected = (state.shape[0], self.prediction_steps, self.keypoint_count, 3)
        if (
            command_motion.shape[0] != expected[0]
            or command_motion.shape[1] not in (1, self.prediction_steps)
            or command_motion.shape[-2:] != expected[-2:]
        ):
            raise ValueError(f"command_motion must broadcast to {expected}, got {tuple(command_motion.shape)}")
        return self._sample_impl(
            state,
            ddim_steps,
            mode=mode,
            command_motion=command_motion,
            guidance_strength=guidance_strength,
            motion_mean=motion_mean,
            motion_std=motion_std,
            initial_noise=initial_noise,
            generator=generator,
            env_latent=env_latent,
        )

    def _sample_impl(
        self,
        state: torch.Tensor,
        ddim_steps: int,
        mode: torch.Tensor | None = None,
        command_motion: torch.Tensor | None = None,
        guidance_strength: float = 0.0,
        motion_mean: torch.Tensor | None = None,
        motion_std: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        env_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = state.shape[0]
        expected_shape = (batch, self.prediction_steps, self.keypoint_count, 3)
        if initial_noise is not None and generator is not None:
            raise ValueError("Pass either initial_noise or generator, not both")
        if initial_noise is not None:
            if initial_noise.shape != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}, got {tuple(initial_noise.shape)}"
                )
            sample = initial_noise.to(device=state.device, dtype=state.dtype).clone()
        else:
            sample = torch.randn(
                expected_shape,
                device=state.device,
                dtype=state.dtype,
                generator=generator,
            )
        schedule = torch.linspace(self.diffusion_steps - 1, 0, ddim_steps, device=state.device).long()
        schedule = torch.unique_consecutive(schedule)
        for schedule_index, timestep in enumerate(schedule):
            t = torch.full((batch,), int(timestep), device=state.device, dtype=torch.long)
            alpha_bar = self.alphas_cumprod[timestep]
            prediction = self(sample, t, state, mode, env_latent=env_latent)
            if self.prediction_type == "epsilon":
                predicted_noise = prediction
                clean = (
                    sample - (1.0 - alpha_bar).sqrt() * predicted_noise
                ) / alpha_bar.sqrt().clamp_min(1.0e-8)
            else:
                clean = prediction
            clean = clean.clamp(-5.0, 5.0)

            if command_motion is not None and guidance_strength > 0.0:
                if motion_mean is None or motion_std is None:
                    raise ValueError("guided sampling requires motion_mean and motion_std")
                mean = motion_mean.to(clean)
                std = motion_std.to(clean).clamp_min(1.0e-6)
                clean_physical = clean * std + mean
                if schedule_index == len(schedule) - 1:
                    variance = torch.zeros((), device=sample.device, dtype=sample.dtype)
                else:
                    previous_timestep = schedule[schedule_index + 1]
                    previous_alpha = self.alphas_cumprod[previous_timestep]
                    variance = (
                        (1.0 - previous_alpha) / (1.0 - alpha_bar).clamp_min(1.0e-8)
                        * (1.0 - alpha_bar / previous_alpha).clamp_min(0.0)
                    )
                gradient_normalized = (
                    2.0 * (clean_physical - command_motion.to(clean_physical)) * std
                )
                clean = (clean - guidance_strength * variance * gradient_normalized).clamp(-5.0, 5.0)

            # Recompute epsilon from the (possibly guided or clamped) clean
            # prediction so the correction propagates through the DDIM update.
            predicted_noise = (
                sample - alpha_bar.sqrt() * clean
            ) / (1.0 - alpha_bar).sqrt().clamp_min(1.0e-8)
            if schedule_index == len(schedule) - 1:
                sample = clean
            else:
                previous_timestep = schedule[schedule_index + 1]
                previous_alpha = self.alphas_cumprod[previous_timestep]
                sample = previous_alpha.sqrt() * clean + (1.0 - previous_alpha).sqrt() * predicted_noise
        return sample


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class InverseDynamicsModel(nn.Module):
    """Residual MLP mapping robot state and motion to absolute PD targets.

    ``action_dim`` is retained as the serialized configuration key, but denotes
    the number of controlled joint targets in the v2 pipeline.
    """

    def __init__(
        self,
        state_dim: int,
        history_steps: int,
        keypoint_count: int,
        prediction_steps: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        tactile_arch: str = "direct_v1",
        proprio_state_dim: int = 78,
        tactile_dim: int = 5,
        tactile_per_finger_dim: int = 32,
        tactile_embedding_dim: int = 128,
        tactile_contact_threshold_normalized: list[float] | tuple[float, ...] | None = None,
        env_latent_dim: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.history_steps = history_steps
        self.keypoint_count = keypoint_count
        self.prediction_steps = prediction_steps
        self.action_dim = action_dim
        if tactile_arch not in {"none", "direct_v1", "v2b"}:
            raise ValueError(f"Unsupported tactile architecture: {tactile_arch!r}")
        self.tactile_arch = tactile_arch
        self.proprio_state_dim = proprio_state_dim
        self.tactile_dim = tactile_dim
        self.tactile_per_finger_dim = tactile_per_finger_dim
        self.tactile_embedding_dim = tactile_embedding_dim
        self.env_latent_dim = env_latent_dim
        if env_latent_dim < 0:
            raise ValueError("env_latent_dim must be non-negative")
        if tactile_arch == "v2b" and state_dim != proprio_state_dim + tactile_dim:
            raise ValueError("v2b state_dim must equal proprio_state_dim + tactile_dim")
        encoded_state_dim = proprio_state_dim if tactile_arch == "v2b" else state_dim
        input_dim = encoded_state_dim * history_steps + keypoint_count * prediction_steps * 3
        self.input = nn.Linear(input_dim, hidden_dim)
        if tactile_arch == "v2b":
            self.tactile_encoder = TactileHistoryEncoder(
                history_steps=history_steps,
                tactile_dim=tactile_dim,
                output_dim=hidden_dim,
                per_finger_dim=tactile_per_finger_dim,
                embedding_dim=tactile_embedding_dim,
                contact_threshold_normalized=tactile_contact_threshold_normalized,
            )
            self.tactile_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
            )
            self.future_tactile_head = nn.Linear(
                tactile_embedding_dim, prediction_steps * tactile_dim
            )
        else:
            self.tactile_encoder = None
            self.tactile_gate = None
            self.future_tactile_head = None
        self.blocks = nn.Sequential(*(ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)))
        self.env_latent_projection = (
            nn.Sequential(
                nn.LayerNorm(env_latent_dim),
                nn.Linear(env_latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if env_latent_dim else None
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def config(self) -> dict[str, int | str | list[float]]:
        config: dict[str, int | str | list[float]] = {
            "state_dim": self.state_dim,
            "history_steps": self.history_steps,
            "keypoint_count": self.keypoint_count,
            "prediction_steps": self.prediction_steps,
            "action_dim": self.action_dim,
            "hidden_dim": self.input.out_features,
            "num_blocks": len(self.blocks),
            "tactile_arch": self.tactile_arch,
            "proprio_state_dim": self.proprio_state_dim,
            "tactile_dim": self.tactile_dim,
            "tactile_per_finger_dim": self.tactile_per_finger_dim,
            "tactile_embedding_dim": self.tactile_embedding_dim,
            "env_latent_dim": self.env_latent_dim,
        }
        if self.tactile_encoder is not None:
            config["tactile_contact_threshold_normalized"] = (
                self.tactile_encoder.contact_threshold_normalized.detach().cpu().tolist()
            )
        return config

    def forward(
        self,
        state: torch.Tensor,
        motion: torch.Tensor,
        *,
        env_latent: torch.Tensor | None = None,
        return_tactile_prediction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.tactile_arch == "v2b":
            proprio = state[..., : self.proprio_state_dim]
            tactile = state[..., self.proprio_state_dim :]
            value = torch.cat((proprio.flatten(1), motion.flatten(1)), dim=-1)
            hidden = F.silu(self.input(value))
            assert self.tactile_encoder is not None and self.tactile_gate is not None
            tactile_hidden, tactile_embedding = self.tactile_encoder(tactile)
            gate = self.tactile_gate(torch.cat((hidden, tactile_hidden), dim=-1))
            hidden = hidden + gate * tactile_hidden
        else:
            value = torch.cat((state.flatten(1), motion.flatten(1)), dim=-1)
            hidden = F.silu(self.input(value))
            tactile_embedding = None
        if self.env_latent_projection is not None:
            if env_latent is None or env_latent.shape != (value.shape[0], self.env_latent_dim):
                raise ValueError(
                    f"env_latent must have shape [B,{self.env_latent_dim}]"
                )
            hidden = hidden + self.env_latent_projection(env_latent)
        hidden = self.blocks(hidden)
        mean = self.mean(hidden)
        log_std = self.log_std.clamp(-5.0, 2.0).expand(value.shape[0], -1)
        if not return_tactile_prediction:
            return mean, log_std
        if tactile_embedding is None or self.future_tactile_head is None:
            raise ValueError("Future tactile prediction requires tactile_arch='v2b'")
        tactile_prediction = self.future_tactile_head(tactile_embedding).view(
            value.shape[0], self.prediction_steps, self.tactile_dim
        )
        return mean, log_std, tactile_prediction

    def loss_components(
        self,
        state: torch.Tensor,
        motion: torch.Tensor,
        target: torch.Tensor,
        future_tactile: torch.Tensor | None = None,
        env_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if future_tactile is None:
            mean, _ = self(state, motion, env_latent=env_latent)
            tactile_loss = None
        else:
            mean, _, tactile_prediction = self(
                state, motion, env_latent=env_latent, return_tactile_prediction=True
            )
            tactile_loss = F.smooth_l1_loss(
                tactile_prediction, future_tactile, reduction="none"
            ).flatten(1).mean(1)
        target_loss = F.mse_loss(mean, target, reduction="none").flatten(1).mean(1)
        return target_loss, tactile_loss

    def loss(
        self,
        state: torch.Tensor,
        motion: torch.Tensor,
        target: torch.Tensor,
        future_tactile: torch.Tensor | None = None,
        tactile_aux_weight: float = 0.0,
        env_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The paper parameterizes a normal distribution but trains the inverse
        # dynamics regressor with supervised MSE.
        target_loss, tactile_loss = self.loss_components(
            state, motion, target, future_tactile, env_latent
        )
        if tactile_loss is not None:
            target_loss = target_loss + tactile_aux_weight * tactile_loss
        return target_loss.mean()

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        motion: torch.Tensor,
        env_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self(state, motion, env_latent=env_latent)[0]
