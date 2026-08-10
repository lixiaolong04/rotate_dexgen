"""Candidate-conditioned privileged teacher and tactile-history student."""

from __future__ import annotations

import torch
from torch import nn


class PrivilegedContactEncoder(nn.Module):
    def __init__(self, privileged_dim: int = 22, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.privileged_dim = int(privileged_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.privileged_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.privileged_dim),
        )

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.encoder(privileged)

    def reconstruct(self, privileged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self(privileged)
        return latent, self.decoder(latent)

    def config(self) -> dict[str, int]:
        return {
            "privileged_dim": self.privileged_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }


class TactileProprioTConv(nn.Module):
    """Thirty-frame 20 Hz adapter patterned after the proven RL ProprioAdapt TConv."""

    def __init__(
        self,
        state_dim: int = 83,
        history_steps: int = 30,
        latent_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if history_steps < 29:
            raise ValueError("TactileProprioTConv requires at least 29 history frames")
        self.state_dim = int(state_dim)
        self.history_steps = int(history_steps)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(3),
        )
        self.output = nn.Linear(self.hidden_dim * 3, self.latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        expected = (self.history_steps, self.state_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"Expected history [B,{expected[0]},{expected[1]}], got {tuple(history.shape)}")
        hidden = self.frame_encoder(history).transpose(1, 2)
        return torch.tanh(self.output(self.temporal(hidden).flatten(1)))

    def config(self) -> dict[str, int]:
        return {
            "state_dim": self.state_dim,
            "history_steps": self.history_steps,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }


class CandidateContactRiskHead(nn.Module):
    """Score whether a proposed two-step keypoint motion preserves rotation."""

    def __init__(
        self,
        state_dim: int = 83,
        policy_history_steps: int = 4,
        keypoint_count: int = 10,
        prediction_steps: int = 2,
        latent_dim: int = 16,
        num_modes: int = 2,
        mode_dim: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.policy_history_steps = int(policy_history_steps)
        self.keypoint_count = int(keypoint_count)
        self.prediction_steps = int(prediction_steps)
        self.latent_dim = int(latent_dim)
        self.num_modes = int(num_modes)
        self.mode_dim = int(mode_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode_embedding = nn.Embedding(self.num_modes, self.mode_dim)
        input_dim = (
            self.policy_history_steps * self.state_dim
            + self.prediction_steps * self.keypoint_count * 3
            + self.latent_dim
            + self.mode_dim
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(
        self,
        policy_state: torch.Tensor,
        motion: torch.Tensor,
        latent: torch.Tensor,
        mode: torch.Tensor,
    ) -> torch.Tensor:
        if policy_state.shape[1:] != (self.policy_history_steps, self.state_dim):
            raise ValueError("Policy state does not match the v3 four-frame contract")
        if motion.shape[1:] != (self.prediction_steps, self.keypoint_count, 3):
            raise ValueError("Candidate motion does not match the two-step keypoint contract")
        if latent.shape[1:] != (self.latent_dim,):
            raise ValueError("Contact latent has the wrong dimension")
        value = torch.cat(
            (
                policy_state.flatten(1),
                motion.flatten(1),
                latent,
                self.mode_embedding(mode),
            ),
            dim=-1,
        )
        return self.network(value).squeeze(-1)

    def config(self) -> dict[str, int]:
        return {
            "state_dim": self.state_dim,
            "policy_history_steps": self.policy_history_steps,
            "keypoint_count": self.keypoint_count,
            "prediction_steps": self.prediction_steps,
            "latent_dim": self.latent_dim,
            "num_modes": self.num_modes,
            "mode_dim": self.mode_dim,
            "hidden_dim": self.hidden_dim,
        }
