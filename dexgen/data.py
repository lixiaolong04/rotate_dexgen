"""Dataset and normalization utilities for the cylinder DexGen pipeline.

The representation follows the DexterityGen appendix:

* four historical robot-state frames;
* PIP and physical fingertip-pad keypoints represented in the wrist frame;
* a short sequence of future keypoint offsets as the motion action;
* the absolute future PD target as the inverse-dynamics supervision target.

Episodes are split as complete units.  Transitions immediately preceding a drop
are excluded, rather than leaking pieces of the same episode across splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from dexgen.xhand_keypoints import validate_representation_metadata


PROPRIO_STATE_DIM = 78
TACTILE_DIM = 5
TACTILE_STATE_DIM = PROPRIO_STATE_DIM + TACTILE_DIM
PROPRIO_STATE_SCHEMA = "xhand_proprio_v1"
TACTILE_STATE_SCHEMA = "xhand_proprio_tactile_v1"
TACTILE_NAMES = ("thumb", "index", "mid", "ring", "pinky")
TACTILE_ARCH_NONE = "none"
TACTILE_ARCH_DIRECT = "direct_v1"
TACTILE_ARCH_V2B = "v2b"
TACTILE_ARCH_FILM = "film_v3"
TACTILE_ARCHITECTURES = (
    TACTILE_ARCH_NONE, TACTILE_ARCH_DIRECT, TACTILE_ARCH_V2B, TACTILE_ARCH_FILM,
)


REQUIRED_FIELDS = (
    "joint_pos",
    "joint_vel",
    "joint_target",
    "next_joint_target",
    "action",
    "keypoints",
    "tactile",
    "reward",
)


def load_rollouts(path: str | Path) -> dict[str, Any]:
    """Load and validate a rollout file produced by ``collect_rl_data.py``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    rollout_format = payload.get("format")
    if rollout_format not in {"xhand_dexgen_rollout_v2", "xhand_dexgen_rollout_v3"}:
        raise ValueError(
            f"DexGen training requires xhand_dexgen_rollout_v2/v3, but {path!s} contains "
            f"{payload.get('format')!r}. Recollect data with the updated collect_rl_data.py; "
            "the old file is intentionally not overwritten."
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Rollout file contains no episodes")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Rollout file contains no metadata dictionary")
    validate_representation_metadata(metadata)
    for ep_idx, episode in enumerate(episodes):
        missing = [name for name in REQUIRED_FIELDS if name not in episode]
        if missing:
            raise ValueError(f"Episode {ep_idx} is missing fields: {missing}")
        lengths = {name: int(episode[name].shape[0]) for name in REQUIRED_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Episode {ep_idx} has inconsistent lengths: {lengths}")
        if episode["keypoints"].ndim != 3 or tuple(episode["keypoints"].shape[1:]) != (10, 3):
            raise ValueError(
                f"Episode {ep_idx} keypoints must have shape [N,10,3], "
                f"got {tuple(episode['keypoints'].shape)}"
            )
        if episode["joint_pos"].shape[-1] != 21 or episode["next_joint_target"].shape[-1] != 21:
            raise ValueError(f"Episode {ep_idx} does not use the canonical 21-DOF XHand representation")
        if episode["tactile"].ndim != 2 or episode["tactile"].shape[-1] != TACTILE_DIM:
            raise ValueError(
                f"Episode {ep_idx} tactile must have shape [N,{TACTILE_DIM}], "
                f"got {tuple(episode['tactile'].shape)}"
            )
        if rollout_format == "xhand_dexgen_rollout_v3" and "mode_id" not in episode:
            raise ValueError(f"Episode {ep_idx} is missing mode_id required by rollout v3")
        episode.setdefault("mode_id", 0)
    return payload


def load_rollout_collection(paths: str | Path | Sequence[str | Path]) -> dict[str, Any]:
    """Load one or more compatible rollout files into one episode collection."""
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not path_list:
        raise ValueError("At least one rollout path is required")
    payloads = [load_rollouts(path) for path in path_list]
    first_metadata = payloads[0]["metadata"]
    for payload in payloads[1:]:
        metadata = payload["metadata"]
        for key in (
            "control_hz", "dexgen_control_hz", "keypoint_names", "keypoint_frame",
            "fingertip_offsets_in_dip_frame", "keypoint_semantics", "joint_names",
            "domain_randomization_profile", "domain_randomization",
        ):
            if key in first_metadata or key in metadata:
                if first_metadata.get(key) != metadata.get(key):
                    raise ValueError(f"Rollout files use different {key}")
        if first_metadata.get("sensor_profile") != metadata.get("sensor_profile"):
            raise ValueError("Rollout files use different sensor_profile values")
    combined_metadata = dict(first_metadata)
    mode_names: dict[str, str] = {}
    mode_axis_signs: dict[str, int] = {}
    for payload in payloads:
        metadata = payload["metadata"]
        mode_names.update(metadata.get("mode_names", {}))
        mode_axis_signs.update(metadata.get("mode_axis_signs", {}))
    if mode_names:
        combined_metadata["mode_names"] = mode_names
        combined_metadata["mode_axis_signs"] = mode_axis_signs
    combined_metadata["source_rollouts"] = [str(path) for path in path_list]
    return {
        "format": "xhand_dexgen_rollout_collection_v1",
        "metadata": combined_metadata,
        "episodes": [episode for payload in payloads for episode in payload["episodes"]],
    }


def checkpoint_representation(payload: dict[str, Any]) -> dict[str, object]:
    """Copy the representation fields stored in a rollout into a checkpoint."""
    metadata = payload["metadata"]
    keys = (
        "joint_names",
        "keypoint_names",
        "keypoint_frame",
        "fingertip_offsets_in_dip_frame",
        "keypoint_semantics",
        "sensor_profile",
        "mode_names",
        "mode_axis_signs",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def validate_checkpoint_representations(*checkpoints: dict[str, Any]) -> None:
    """Validate and compare representation contracts when checkpoints provide them."""
    representations = [checkpoint.get("representation") for checkpoint in checkpoints]
    present = [value for value in representations if value is not None]
    if not present:
        return  # Legacy v3 checkpoints predate the serialized contract.
    if len(present) != len(representations):
        raise ValueError("Cannot mix legacy and representation-tagged DexGen checkpoints")
    for value in present:
        if not isinstance(value, dict):
            raise ValueError("Checkpoint representation must be a dictionary")
        validate_representation_metadata(value, require_joint_names=True)
    if any(value != present[0] for value in present[1:]):
        raise ValueError("Diffusion and IDM checkpoints use different XHand representations")


def state_contract(
    include_tactile: bool,
    tactile_arch: str | None = None,
) -> dict[str, object]:
    """Return the serialized state contract shared by training and inference."""
    if tactile_arch is None:
        tactile_arch = TACTILE_ARCH_DIRECT if include_tactile else TACTILE_ARCH_NONE
    if tactile_arch not in TACTILE_ARCHITECTURES:
        raise ValueError(f"Unsupported tactile architecture: {tactile_arch!r}")
    if include_tactile != (tactile_arch != TACTILE_ARCH_NONE):
        raise ValueError("tactile_arch must be 'none' exactly when tactile is disabled")
    return {
        "state_schema": TACTILE_STATE_SCHEMA if include_tactile else PROPRIO_STATE_SCHEMA,
        "state_dim": TACTILE_STATE_DIM if include_tactile else PROPRIO_STATE_DIM,
        "include_tactile": bool(include_tactile),
        "tactile_names": list(TACTILE_NAMES) if include_tactile else [],
        "tactile_arch": tactile_arch,
    }


def checkpoint_state_contract(checkpoint: dict[str, Any]) -> dict[str, object]:
    """Read and validate a checkpoint's proprio/tactile state declaration.

    Existing v3-v5 checkpoints predate the explicit schema tag.  They remain
    valid only when their model and normalizer use the legacy 78-D state.
    """
    model_config = checkpoint.get("model_config")
    dataset_config = checkpoint.get("dataset_config")
    if not isinstance(model_config, dict) or not isinstance(dataset_config, dict):
        raise ValueError("DexGen checkpoint is missing model_config or dataset_config")
    model_state_dim = int(model_config.get("state_dim", -1))
    schema = dataset_config.get("state_schema")
    if schema is None:
        if model_state_dim != PROPRIO_STATE_DIM:
            raise ValueError(
                "Only legacy 78-D checkpoints may omit dataset_config.state_schema"
            )
        contract = state_contract(False)
    else:
        include_tactile = dataset_config.get("include_tactile")
        if not isinstance(include_tactile, bool):
            raise ValueError("Checkpoint include_tactile must be a boolean")
        # tactile-v1 checkpoints predate the explicit architecture field.  Their
        # 83-D representation is the direct-concatenation baseline by definition.
        tactile_arch = dataset_config.get(
            "tactile_arch",
            TACTILE_ARCH_DIRECT if include_tactile else TACTILE_ARCH_NONE,
        )
        contract = state_contract(include_tactile, tactile_arch)
        model_tactile_arch = model_config.get(
            "tactile_arch",
            TACTILE_ARCH_DIRECT if include_tactile else TACTILE_ARCH_NONE,
        )
        if model_tactile_arch != tactile_arch:
            raise ValueError(
                "Checkpoint model and dataset tactile architectures differ: "
                f"{model_tactile_arch!r} != {tactile_arch!r}"
            )
        for key, expected in contract.items():
            actual = dataset_config.get(key, expected if key == "tactile_arch" else None)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint state contract mismatch for {key}: "
                    f"expected {expected!r}, got {actual!r}"
                )
    if model_state_dim != int(contract["state_dim"]):
        raise ValueError(
            f"Checkpoint model state_dim={model_state_dim} does not match "
            f"{contract['state_schema']} ({contract['state_dim']})"
        )
    normalizer = checkpoint.get("normalizer")
    if not isinstance(normalizer, dict):
        raise ValueError("DexGen checkpoint is missing state normalizer metadata")
    for name in ("state_mean", "state_std"):
        if name not in normalizer or int(normalizer[name].numel()) != model_state_dim:
            raise ValueError(
                f"Checkpoint {name} dimension does not match the model state"
            )
    return contract


def validate_checkpoint_state_contracts(*checkpoints: dict[str, Any]) -> dict[str, object]:
    """Reject checkpoints with incompatible input states.

    ``tactile_arch`` describes each model's internal encoder, not the external
    83-D input contract.  Diffusion film_v3 is therefore intentionally allowed
    to run with a v2b IDM when schema, dimensions, and tactile ordering match.
    """
    contracts = [checkpoint_state_contract(checkpoint) for checkpoint in checkpoints]
    comparable_keys = (
        "state_schema", "state_dim", "include_tactile", "tactile_names",
    )
    reference = {key: contracts[0][key] for key in comparable_keys}
    if any(
        {key: contract[key] for key in comparable_keys} != reference
        for contract in contracts[1:]
    ):
        raise ValueError("Diffusion and IDM checkpoints use different state contracts")
    return contracts[0]


def tactile_from_observation(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract the live five-finger tactile frame used during collection."""
    history = obs_dict.get("proprio_hist")
    if history is None or history.ndim != 3 or history.shape[-1] < TACTILE_DIM:
        shape = None if history is None else tuple(history.shape)
        raise ValueError(f"Expected proprio_hist [B,H,F>=5], got {shape}")
    return history[:, -1, -TACTILE_DIM:]


@dataclass
class FeatureNormalizer:
    """Per-feature affine normalization with a protected minimum scale."""

    state_mean: torch.Tensor
    state_std: torch.Tensor
    motion_mean: torch.Tensor
    motion_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    @staticmethod
    def _safe_std(value: torch.Tensor) -> torch.Tensor:
        return value.clamp_min(1.0e-6)

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.state_mean.to(value)) / self._safe_std(self.state_std.to(value))

    def normalize_tactile(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize five-finger tactile values with the state tail statistics."""
        if self.state_mean.numel() < TACTILE_DIM:
            raise ValueError("State normalizer does not contain tactile statistics")
        mean = self.state_mean[-TACTILE_DIM:].to(value)
        std = self._safe_std(self.state_std[-TACTILE_DIM:].to(value))
        return (value - mean) / std

    def normalized_tactile_threshold(self, threshold: float) -> torch.Tensor:
        """Map a physical scalar contact threshold into each normalized channel."""
        mean = self.state_mean[-TACTILE_DIM:]
        std = self._safe_std(self.state_std[-TACTILE_DIM:])
        return (torch.full_like(mean, float(threshold)) - mean) / std

    def normalize_motion(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.motion_mean.to(value)) / self._safe_std(self.motion_std.to(value))

    def denormalize_motion(self, value: torch.Tensor) -> torch.Tensor:
        return value * self._safe_std(self.motion_std.to(value)) + self.motion_mean.to(value)

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.action_mean.to(value)) / self._safe_std(self.action_std.to(value))

    def denormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return value * self._safe_std(self.action_std.to(value)) + self.action_mean.to(value)

    # The serialized field names remain action_mean/std for backward readability,
    # but in v2 they normalize the IDM's absolute joint-target output.
    normalize_target = normalize_action
    denormalize_target = denormalize_action

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).cpu() for name in self.__dataclass_fields__}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FeatureNormalizer":
        return cls(**{name: state[name].float() for name in cls.__dataclass_fields__})


def build_state_frame(
    episode: dict[str, torch.Tensor],
    index: int | slice | torch.Tensor,
    *,
    include_tactile: bool = False,
) -> torch.Tensor:
    """Build the deployable DexGen state used by both offline and online code.

    Layout: wrist-frame fingertip positions, joint position, current PD target,
    and control error.  PIP points are motion-label keypoints rather than state.
    The paper-aligned baseline is 78-D; the optional tactile variant appends five
    current/history contact magnitudes in thumb/index/mid/ring/pinky order.
    """
    keypoints = episode["keypoints"][index]
    # Collection stores [PIP, fingertip] for each of the five XHand fingers.
    fingertips = keypoints[..., 1::2, :]
    fingertips = fingertips.reshape(*fingertips.shape[:-2], -1)
    joint_pos = episode["joint_pos"][index]
    joint_target = episode["joint_target"][index]
    features = (fingertips, joint_pos, joint_target, joint_target - joint_pos)
    if include_tactile:
        features += (episode["tactile"][index],)
    return torch.cat(features, dim=-1).float()


class DexGenDataset(Dataset):
    """Episode-safe sliding-window dataset for diffusion and IDM training."""

    def __init__(
        self,
        rollout_path: str | Path | Sequence[str | Path],
        split: str,
        history_steps: int = 4,
        prediction_steps: int = 2,
        future_stride: int = 2,
        history_stride: int = 2,
        sample_stride: int = 2,
        validation_fraction: float = 0.1,
        split_seed: int = 42,
        drop_trim_seconds: float = 2.0,
        include_tactile: bool = False,
        normalizer: FeatureNormalizer | None = None,
        return_fields: Sequence[str] | None = None,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train, val, or all; got {split!r}")
        if min(history_steps, prediction_steps, future_stride, history_stride, sample_stride) < 1:
            raise ValueError("All history, prediction, future, and sample strides must be positive")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")

        self.payload = load_rollout_collection(rollout_path)
        self.history_steps = history_steps
        self.prediction_steps = prediction_steps
        self.future_stride = future_stride
        self.history_stride = history_stride
        self.sample_stride = sample_stride
        self.include_tactile = include_tactile
        self.return_fields = (
            frozenset(return_fields) if return_fields is not None else None
        )
        self.control_hz = float(self.payload["metadata"]["control_hz"])
        self.drop_trim_steps = int(round(drop_trim_seconds * self.control_hz))

        all_ids = list(range(len(self.payload["episodes"])))
        # Split complete episodes within each mode so validation represents both directions.
        ids_by_mode: dict[int, list[int]] = {}
        for episode_id in all_ids:
            mode_id = int(self.payload["episodes"][episode_id].get("mode_id", 0))
            ids_by_mode.setdefault(mode_id, []).append(episode_id)
        train_ids, val_ids = [], []
        for mode_id, mode_ids in sorted(ids_by_mode.items()):
            val_count = max(1, round(len(mode_ids) * validation_fraction)) if len(mode_ids) > 1 else 0
            generator = torch.Generator().manual_seed(int(split_seed) + 1_000_003 * mode_id)
            permutation = torch.randperm(len(mode_ids), generator=generator).tolist()
            shuffled_ids = [mode_ids[index] for index in permutation]
            val_ids.extend(shuffled_ids[:val_count])
            train_ids.extend(shuffled_ids[val_count:])
        train_ids.sort()
        val_ids.sort()
        self.episode_ids = all_ids if split == "all" else (train_ids if split == "train" else val_ids)
        if not self.episode_ids:
            raise ValueError(f"No episodes are available for split {split!r}")

        self.samples: list[tuple[int, int]] = []
        last_future = prediction_steps * future_stride
        for episode_id in self.episode_ids:
            episode = self.payload["episodes"][episode_id]
            end = int(episode["action"].shape[0])
            if episode.get("termination", "timeout") == "drop":
                end = max(0, end - self.drop_trim_steps)
            first_index = (history_steps - 1) * history_stride
            last_index = end - last_future - 1
            candidates = range(first_index, last_index + 1, sample_stride)
            supervision_valid = episode.get("supervision_valid")
            if supervision_valid is None:
                self.samples.extend((episode_id, index) for index in candidates)
            else:
                supervision_valid = supervision_valid.bool()
                self.samples.extend(
                    (episode_id, index)
                    for index in candidates
                    if bool(supervision_valid[index : index + last_future + 1].all())
                )
        if not self.samples:
            raise ValueError(
                "No valid windows remain. Collect longer episodes or reduce history/prediction settings."
            )

        self.normalizer = normalizer

    @property
    def state_dim(self) -> int:
        episode = self.payload["episodes"][self.episode_ids[0]]
        return int(
            build_state_frame(
                episode, 0, include_tactile=self.include_tactile
            ).numel()
        )

    @property
    def keypoint_count(self) -> int:
        return int(self.payload["episodes"][self.episode_ids[0]]["keypoints"].shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.payload["episodes"][self.episode_ids[0]]["next_joint_target"].shape[-1])

    @property
    def priv_info_dim(self) -> int:
        """Dimension of optional simulator-only RMA supervision labels."""
        episode = self.payload["episodes"][self.episode_ids[0]]
        if "priv_info" not in episode:
            raise ValueError("This rollout collection has no priv_info RMA labels")
        return int(episode["priv_info"].shape[-1])

    @property
    def mode_ids(self) -> tuple[int, ...]:
        return tuple(sorted({int(self.payload["episodes"][episode_id].get("mode_id", 0)) for episode_id in self.episode_ids}))

    @property
    def num_modes(self) -> int:
        ids = self.mode_ids
        if ids != tuple(range(max(ids) + 1)):
            raise ValueError(f"Mode ids must be contiguous from zero, got {ids}")
        return len(ids)

    def balanced_sample_weights(self) -> torch.Tensor:
        counts: dict[int, int] = {}
        modes = []
        for episode_id, _ in self.samples:
            mode_id = int(self.payload["episodes"][episode_id].get("mode_id", 0))
            modes.append(mode_id)
            counts[mode_id] = counts.get(mode_id, 0) + 1
        return torch.tensor([1.0 / counts[mode_id] for mode_id in modes], dtype=torch.double)

    def recovery_sample_weights(
        self,
        *,
        boost: float = 4.0,
        target_fraction: float | None = None,
        contact_threshold: float = 0.1,
        low_contact_count: int = 2,
        stable_contact_count: int = 4,
        horizon_seconds: float = 0.5,
        explicit_only: bool = False,
        type_weights: Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mode-balanced weights that oversample successful tactile recoveries.

        A sample is a recovery when the support count falls from a stable
        multi-finger contact to at most ``low_contact_count`` and returns to the
        stable count within the requested horizon.  Samples must remain inside
        the already-safe portion of the episode; the existing drop-tail trim
        therefore guarantees that labelled recoveries are followed by the
        configured survival margin.
        """
        if not self.include_tactile:
            raise ValueError("Recovery weighting requires tactile observations")
        if boost < 1.0 or contact_threshold < 0.0:
            raise ValueError("Invalid recovery contact threshold or boost")
        if target_fraction is not None and not 0.0 < target_fraction < 1.0:
            raise ValueError("Recovery target fraction must be in (0,1)")
        if type_weights is not None and (
            len(type_weights) != 4
            or any(value < 0.0 for value in type_weights)
            or sum(type_weights) <= 0.0
        ):
            raise ValueError(
                "Recovery type weights must contain four non-negative values "
                "for action/release/force/natural"
            )
        if not 0 <= low_contact_count < stable_contact_count <= TACTILE_DIM:
            raise ValueError("Invalid low/stable recovery contact counts")
        if horizon_seconds <= 0.0:
            raise ValueError("Invalid recovery horizon")

        horizon_steps = max(1, int(round(horizon_seconds * self.control_hz)))
        labels = torch.zeros(len(self.samples), dtype=torch.bool)
        factors = torch.ones(len(self.samples), dtype=torch.double)
        cursor = 0
        for episode_id in self.episode_ids:
            start = cursor
            while cursor < len(self.samples) and self.samples[cursor][0] == episode_id:
                cursor += 1
            if cursor == start:
                continue
            episode = self.payload["episodes"][episode_id]
            if "perturbation_phase" in episode and "recovery_success" in episode:
                indices = torch.tensor(
                    [index for _, index in self.samples[start:cursor]], dtype=torch.long
                )
                explicit = (
                    episode["perturbation_phase"].index_select(0, indices).long() == 2
                ) & episode["recovery_success"].index_select(0, indices).bool()
                labels[start:cursor] = explicit
                continue
            if explicit_only:
                continue
            tactile = episode["tactile"].float()
            indices = torch.tensor(
                [index for _, index in self.samples[start:cursor]], dtype=torch.long
            )
            usable_end = int(tactile.shape[0])
            if episode.get("termination", "timeout") == "drop":
                usable_end = max(0, usable_end - self.drop_trim_steps)
            valid = indices + horizon_steps < usable_end
            if not bool(valid.any()):
                continue

            selected = indices[valid]
            history_offsets = self.history_stride * torch.arange(
                self.history_steps - 1, -1, -1, dtype=torch.long
            )
            history_ids = selected[:, None] - history_offsets[None, :]
            future_ids = selected[:, None] + torch.arange(
                1, horizon_steps + 1, dtype=torch.long
            )[None, :]
            history_force = tactile[history_ids]
            future_force = tactile[future_ids]
            history_count = (history_force > contact_threshold).sum(dim=-1)
            future_count = (future_force > contact_threshold).sum(dim=-1)
            current_count = history_count[:, -1]
            recovered = (
                (history_count.amax(dim=1) >= stable_contact_count)
                & (current_count <= low_contact_count)
                & (future_count.amax(dim=1) >= stable_contact_count)
            )
            local_positions = torch.arange(start, cursor, dtype=torch.long)[valid]
            labels[local_positions] = recovered

        factors[labels] = float(boost)
        # Normalize each mode independently so recovery oversampling does not
        # undo the existing 50/50 CW/CCW sampler contract.  When an explicit
        # fraction is requested, assign exactly that probability mass to
        # labelled recovery windows in each direction.  This avoids recovery
        # data being silently diluted by millions of steady-rotation windows.
        sample_modes = torch.tensor(
            [
                int(self.payload["episodes"][episode_id].get("mode_id", 0))
                for episode_id, _ in self.samples
            ],
            dtype=torch.long,
        )
        sample_types = torch.tensor(
            [
                int(self.payload["episodes"][episode_id]["perturbation_type_id"][index])
                if "perturbation_type_id" in self.payload["episodes"][episode_id]
                else 0
                for episode_id, index in self.samples
            ],
            dtype=torch.long,
        )
        for mode_id in self.mode_ids:
            selected = sample_modes == mode_id
            if target_fraction is None:
                factors[selected] /= factors[selected].sum().clamp_min(1.0)
                continue
            recovery = selected & labels
            regular = selected & ~labels
            if not bool(recovery.any()):
                raise ValueError(
                    f"Mode {mode_id} has no labelled recovery windows; "
                    "collect and audit recovery data before requesting a target fraction"
                )
            if not bool(regular.any()):
                raise ValueError(f"Mode {mode_id} has no regular rotation windows")
            factors[regular] = (1.0 - float(target_fraction)) / int(regular.sum())
            if type_weights is None:
                factors[recovery] = float(target_fraction) / int(recovery.sum())
                continue
            normalized_type_weights = torch.tensor(
                type_weights, dtype=torch.double
            )
            normalized_type_weights /= normalized_type_weights.sum()
            factors[recovery] = 0.0
            for type_id, type_weight in enumerate(
                normalized_type_weights.tolist(), start=1
            ):
                typed_recovery = recovery & (sample_types == type_id)
                if type_weight == 0.0:
                    continue
                if not bool(typed_recovery.any()):
                    raise ValueError(
                        f"Mode {mode_id} has no successful recovery windows "
                        f"for perturbation type {type_id}"
                    )
                factors[typed_recovery] = (
                    float(target_fraction) * type_weight
                    / int(typed_recovery.sum())
                )
        return factors, labels

    def _raw_item(self, sample_index: int) -> dict[str, torch.Tensor]:
        episode_id, index = self.samples[sample_index]
        episode = self.payload["episodes"][episode_id]
        history_ids = index - self.history_stride * torch.arange(self.history_steps - 1, -1, -1)
        history = build_state_frame(
            episode, history_ids, include_tactile=self.include_tactile
        )
        current_keypoints = episode["keypoints"][index]
        future_ids = index + self.future_stride * torch.arange(1, self.prediction_steps + 1)
        motion = episode["keypoints"].index_select(0, future_ids) - current_keypoints.unsqueeze(0)
        # At a 10 Hz controller tick, the target supervised here is the absolute
        # PD target reached by the recorded 20 Hz expert after two raw actions.
        target = episode["next_joint_target"][index + self.future_stride - 1]
        item = {
            "state": history,
            "motion": motion.float(),
            "target": target.float(),
            "current_target": episode["joint_target"][index].float(),
            "expert_action": episode["action"][index].float(),
            "mode": torch.tensor(int(episode.get("mode_id", 0)), dtype=torch.long),
            "episode_id": torch.tensor(episode_id, dtype=torch.long),
            "step": torch.tensor(index, dtype=torch.long),
            # Keep the mapping schema identical for old steady-rollout and new
            # perturbation episodes so default DataLoader collation is safe.
            "perturbation_phase": torch.tensor(0, dtype=torch.long),
            "perturbation_type": torch.tensor(0, dtype=torch.long),
            "perturbation_subtype": torch.tensor(0, dtype=torch.long),
            "perturbed_finger": torch.tensor(-1, dtype=torch.long),
            "perturbed_finger_mask": torch.zeros(TACTILE_DIM, dtype=torch.bool),
            "recovery_label": torch.tensor(0.0),
            # Mixed v8 policy datasets include legacy stable/recovery episodes
            # without simulator-only labels.  A fixed shape plus validity mask
            # keeps default collation safe while preventing label leakage.
            "priv_info": torch.zeros(9, dtype=torch.float32),
            "priv_info_valid": torch.tensor(False),
        }
        if "perturbation_phase" in episode:
            item["perturbation_phase"] = episode["perturbation_phase"][index].long()
            item["perturbation_type"] = episode["perturbation_type_id"][index].long()
            if "perturbation_subtype_id" in episode:
                item["perturbation_subtype"] = episode["perturbation_subtype_id"][
                    index
                ].long()
            item["perturbed_finger"] = episode["perturbed_finger_id"][index].long()
            if "perturbed_finger_mask" in episode:
                item["perturbed_finger_mask"] = episode["perturbed_finger_mask"][
                    index
                ].bool()
            item["recovery_label"] = (
                (episode["perturbation_phase"][index].long() == 2)
                & episode["recovery_success"][index].bool()
            ).float()
        if "object_angvel" in episode:
            future_angvel = episode["object_angvel"].index_select(0, future_ids).float()
            if "rotation_axis" in episode:
                future_axis = episode["rotation_axis"].index_select(0, future_ids).float()
                signed_rotation = (future_angvel * future_axis).sum(dim=-1)
            else:
                axis_sign = float(self.payload["metadata"].get("rotation_axis_sign", 1.0))
                signed_rotation = future_angvel[:, 2] * axis_sign
            item["future_signed_rotation"] = signed_rotation
        if self.include_tactile:
            item["future_tactile"] = episode["tactile"].index_select(0, future_ids).float()
        if "priv_info" in episode:
            item["priv_info"] = episode["priv_info"][index].float()
            item["priv_info_valid"] = torch.tensor(True)
        return item

    def __getitem__(self, sample_index: int) -> dict[str, torch.Tensor]:
        item = self._raw_item(sample_index)
        if self.normalizer is not None:
            item["state"] = self.normalizer.normalize_state(item["state"])
            item["motion"] = self.normalizer.normalize_motion(item["motion"])
            item["target"] = self.normalizer.normalize_target(item["target"])
            if "future_tactile" in item:
                item["future_tactile"] = self.normalizer.normalize_tactile(
                    item["future_tactile"]
                )
        if self.return_fields is not None:
            missing = self.return_fields.difference(item)
            if missing:
                raise KeyError(f"Requested unavailable dataset fields: {sorted(missing)}")
            item = {key: value for key, value in item.items() if key in self.return_fields}
        return item

    def __len__(self) -> int:
        return len(self.samples)

    def fit_normalizer(self, max_samples: int | None = 200_000) -> FeatureNormalizer:
        """Fit statistics on this dataset (normally the training split only)."""
        indices: Iterable[int] = range(len(self))
        if max_samples is not None and len(self) > max_samples:
            # Deterministic even subsampling avoids loading an arbitrarily large dataset at once.
            indices = torch.linspace(0, len(self) - 1, max_samples).long().tolist()
        state, motion, target = [], [], []
        for index in indices:
            item = self._raw_item(index)
            state.append(item["state"])
            motion.append(item["motion"])
            target.append(item["target"])
        state_tensor = torch.cat(state, dim=0)
        motion_tensor = torch.cat(motion, dim=0)
        target_tensor = torch.stack(target, dim=0)
        return FeatureNormalizer(
            state_mean=state_tensor.mean(0),
            state_std=state_tensor.std(0, unbiased=False),
            motion_mean=motion_tensor.mean(0),
            motion_std=motion_tensor.std(0, unbiased=False),
            action_mean=target_tensor.mean(0),
            action_std=target_tensor.std(0, unbiased=False),
        )
