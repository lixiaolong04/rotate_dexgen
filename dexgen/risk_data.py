"""Datasets for candidate-conditioned tactile recovery risk learning.

Unlike the action-prior dataset, this dataset intentionally keeps failed
recovery tails.  They are unsafe examples for the risk model and must not be
removed by DexterityGen's normal two-second failed-tail filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from dexgen.data import (
    FeatureNormalizer,
    TACTILE_DIM,
    build_state_frame,
    load_rollout_collection,
)


PRIVILEGED_RISK_DIM = 22
OBJECT_DYNAMICS_DIM = 13


class RecoveryRiskDataset(Dataset):
    """Recovery windows with deployable history, candidate motion, and oracle state."""

    def __init__(
        self,
        rollout_path: str | Path | Sequence[str | Path],
        *,
        split: str,
        normalizer: FeatureNormalizer,
        history_steps: int = 30,
        history_stride: int = 1,
        prediction_steps: int = 2,
        future_stride: int = 2,
        sample_stride: int = 1,
        risk_horizon_seconds: float = 1.0,
        min_contact_fingers: int = 3,
        min_signed_rotation: float = 0.15,
        validation_fraction: float = 0.1,
        split_seed: int = 42,
        disable_tactile_ids: Sequence[int] = (),
        use_priv_info: bool = True,
        recovery_label_mode: str = "event",
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train, val, or all; got {split!r}")
        if min(history_steps, history_stride, prediction_steps, future_stride, sample_stride) < 1:
            raise ValueError("history, prediction, future, and sample strides must be positive")
        if risk_horizon_seconds <= 0.0:
            raise ValueError("risk_horizon_seconds must be positive")
        if not 1 <= min_contact_fingers <= TACTILE_DIM:
            raise ValueError("min_contact_fingers is outside the tactile channel range")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        tactile_ids = tuple(sorted(set(int(value) for value in disable_tactile_ids)))
        if any(value < 0 or value >= TACTILE_DIM for value in tactile_ids):
            raise ValueError(f"Invalid tactile channel ids: {tactile_ids}")

        self.payload = load_rollout_collection(rollout_path)
        self.normalizer = normalizer
        self.history_steps = int(history_steps)
        self.history_stride = int(history_stride)
        self.prediction_steps = int(prediction_steps)
        self.future_stride = int(future_stride)
        self.sample_stride = int(sample_stride)
        self.risk_horizon_seconds = float(risk_horizon_seconds)
        self.min_contact_fingers = int(min_contact_fingers)
        self.min_signed_rotation = float(min_signed_rotation)
        self.disable_tactile_ids = tactile_ids
        self.use_priv_info = bool(use_priv_info)
        if recovery_label_mode not in {"event", "local"}:
            raise ValueError("recovery_label_mode must be 'event' or 'local'")
        self.recovery_label_mode = recovery_label_mode
        self.control_hz = float(self.payload["metadata"]["control_hz"])
        self.risk_horizon_steps = max(
            1, int(round(self.risk_horizon_seconds * self.control_hz))
        )

        episode_ids = list(range(len(self.payload["episodes"])))
        by_mode: dict[int, list[int]] = {}
        for episode_id in episode_ids:
            episode = self.payload["episodes"][episode_id]
            required = (
                "object_pos",
                "object_quat",
                "object_linvel",
                "object_angvel",
                "rotation_axis",
                "perturbation_phase",
                "recovery_success",
            )
            if self.use_priv_info:
                required = ("priv_info",) + required
            missing = [name for name in required if name not in episode]
            if missing:
                raise ValueError(
                    f"Risk training episode {episode_id} is missing oracle fields: {missing}"
                )
            by_mode.setdefault(int(episode.get("mode_id", 0)), []).append(episode_id)

        train_ids: list[int] = []
        val_ids: list[int] = []
        for mode_id, mode_ids in sorted(by_mode.items()):
            val_count = (
                max(1, round(len(mode_ids) * validation_fraction))
                if len(mode_ids) > 1 else 0
            )
            generator = torch.Generator().manual_seed(split_seed + 1_000_003 * mode_id)
            permutation = torch.randperm(len(mode_ids), generator=generator).tolist()
            shuffled = [mode_ids[index] for index in permutation]
            val_ids.extend(shuffled[:val_count])
            train_ids.extend(shuffled[val_count:])
        selected_ids = episode_ids if split == "all" else (train_ids if split == "train" else val_ids)
        self.episode_ids = sorted(selected_ids)
        if not self.episode_ids:
            raise ValueError(f"No episodes are available for risk split {split!r}")

        # (episode id, current index, safe label, recovery-window flag)
        self.samples: list[tuple[int, int, bool, bool]] = []
        first_index = (self.history_steps - 1) * self.history_stride
        last_motion_offset = self.prediction_steps * self.future_stride
        for episode_id in self.episode_ids:
            episode = self.payload["episodes"][episode_id]
            length = int(episode["action"].shape[0])
            last_index = length - last_motion_offset - 1
            if last_index < first_index:
                continue
            phase = episode["perturbation_phase"].long()
            recovery_success = episode["recovery_success"].bool()
            for index in range(first_index, last_index + 1, self.sample_stride):
                phase_id = int(phase[index])
                if phase_id == 1:
                    # The injected perturbation is not a policy candidate.
                    continue
                is_recovery = phase_id == 2
                if is_recovery and self.recovery_label_mode == "event":
                    safe = bool(recovery_success[index])
                else:
                    horizon_end = index + self.risk_horizon_steps
                    if horizon_end >= length:
                        # A timeout is right-censored; a drop within the horizon is unsafe.
                        if episode.get("termination", "timeout") != "drop":
                            continue
                        safe = False
                    else:
                        tactile_end = episode["tactile"][horizon_end]
                        contact_ok = int((tactile_end > 0.1).sum()) >= self.min_contact_fingers
                        future_angvel = episode["object_angvel"][index + 1 : horizon_end + 1]
                        future_axis = episode["rotation_axis"][index + 1 : horizon_end + 1]
                        signed_rotation = (future_angvel * future_axis).sum(dim=-1).mean()
                        safe = contact_ok and float(signed_rotation) >= self.min_signed_rotation
                self.samples.append((episode_id, index, safe, is_recovery))
        if not self.samples:
            raise ValueError("No risk windows remain; use longer privileged recovery episodes")

    @property
    def state_dim(self) -> int:
        return int(self.normalizer.state_mean.numel())

    @property
    def keypoint_count(self) -> int:
        episode = self.payload["episodes"][self.episode_ids[0]]
        return int(episode["keypoints"].shape[1])

    @property
    def privileged_dim(self) -> int:
        return PRIVILEGED_RISK_DIM if self.use_priv_info else OBJECT_DYNAMICS_DIM

    @property
    def num_modes(self) -> int:
        modes = sorted(
            {
                int(self.payload["episodes"][episode_id].get("mode_id", 0))
                for episode_id in self.episode_ids
            }
        )
        if modes != list(range(max(modes) + 1)):
            raise ValueError(f"Mode ids must be contiguous from zero, got {modes}")
        return len(modes)

    def _masked_state(self, episode: dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        state = build_state_frame(episode, indices, include_tactile=True)
        if self.disable_tactile_ids:
            state = state.clone()
            state[..., -TACTILE_DIM + torch.tensor(self.disable_tactile_ids)] = 0.0
        return state

    def raw_privileged(self, sample_index: int) -> torch.Tensor:
        episode_id, index, _, _ = self.samples[sample_index]
        episode = self.payload["episodes"][episode_id]
        features = (
            episode["object_pos"][index],
            episode["object_quat"][index],
            episode["object_linvel"][index],
            episode["object_angvel"][index],
        )
        if self.use_priv_info:
            features = (episode["priv_info"][index],) + features
        return torch.cat(features).float()

    def fit_privileged_stats(
        self, max_samples: int | None = 200_000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices: Sequence[int] | range = range(len(self))
        if max_samples is not None and len(self) > max_samples:
            indices = torch.linspace(0, len(self) - 1, max_samples).long().tolist()
        values = torch.stack([self.raw_privileged(index) for index in indices])
        return values.mean(0), values.std(0, unbiased=False).clamp_min(1.0e-6)

    def balanced_sample_weights(self) -> torch.Tensor:
        """Balance direction, safe/unsafe outcome, and recovery/regular context."""
        groups: dict[tuple[int, bool, bool], int] = {}
        keys: list[tuple[int, bool, bool]] = []
        for episode_id, _, safe, recovery in self.samples:
            mode = int(self.payload["episodes"][episode_id].get("mode_id", 0))
            key = (mode, safe, recovery)
            keys.append(key)
            groups[key] = groups.get(key, 0) + 1
        return torch.tensor(
            [1.0 / groups[key] for key in keys], dtype=torch.double
        )

    def class_summary(self) -> dict[str, int]:
        summary = {
            "safe": 0,
            "unsafe": 0,
            "recovery_safe": 0,
            "recovery_unsafe": 0,
        }
        for _, _, safe, recovery in self.samples:
            summary["safe" if safe else "unsafe"] += 1
            if recovery:
                summary["recovery_safe" if safe else "recovery_unsafe"] += 1
        return summary

    def __getitem__(self, sample_index: int) -> dict[str, torch.Tensor]:
        episode_id, index, safe, is_recovery = self.samples[sample_index]
        episode = self.payload["episodes"][episode_id]
        history_ids = index - self.history_stride * torch.arange(
            self.history_steps - 1, -1, -1
        )
        history = self._masked_state(episode, history_ids)
        # Four 10 Hz frames are the exact v3 policy-state contract, sampled from
        # the 20 Hz adapter history without changing the base generator rate.
        policy_ids = index - 2 * torch.arange(3, -1, -1)
        policy_state = self._masked_state(episode, policy_ids)
        current_keypoints = episode["keypoints"][index]
        future_ids = index + self.future_stride * torch.arange(
            1, self.prediction_steps + 1
        )
        motion = (
            episode["keypoints"].index_select(0, future_ids)
            - current_keypoints.unsqueeze(0)
        ).float()
        return {
            "history": self.normalizer.normalize_state(history),
            "policy_state": self.normalizer.normalize_state(policy_state),
            "motion": self.normalizer.normalize_motion(motion),
            "privileged": self.raw_privileged(sample_index),
            "mode": torch.tensor(int(episode.get("mode_id", 0)), dtype=torch.long),
            "label": torch.tensor(float(safe), dtype=torch.float32),
            "is_recovery": torch.tensor(is_recovery),
            "perturbation_type": episode["perturbation_type_id"][index].long(),
        }

    def __len__(self) -> int:
        return len(self.samples)
