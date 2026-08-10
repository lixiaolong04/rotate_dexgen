"""Personalized MANUS open-palm pose calibration and hysteretic inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


PROFILE_SCHEMA = "dexgen.manus_open_profile.v1"
POSE_NAMES = ("open", "closed")
JOINT_DIM = 21


def _flatten_segments(segments: dict[str, list[torch.Tensor]], name: str) -> torch.Tensor:
    chunks = []
    for segment in segments.get(name, []):
        value = segment.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim != 2 or value.shape[1] != JOINT_DIM or value.shape[0] == 0:
            raise ValueError(f"{name} calibration segments must have shape [T,{JOINT_DIM}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} calibration contains NaN or Inf")
        chunks.append(value)
    if not chunks:
        raise ValueError(f"No usable {name} calibration samples")
    return torch.cat(chunks, dim=0)


def build_open_profile(segments: dict[str, list[torch.Tensor]], *, rate_hz: float) -> dict[str, object]:
    """Fit a one-dimensional open-vs-closed pose score from personal samples."""

    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    values = {name: _flatten_segments(segments, name) for name in POSE_NAMES}
    all_values = torch.cat(tuple(values.values()), dim=0)
    # Scale each joint robustly, so a high-range MCP axis cannot obscure the
    # cooperative flexion signal from the remaining fingers.
    feature_mean = all_values.mean(dim=0)
    feature_std = all_values.std(dim=0, unbiased=False).clamp_min(1.0e-4)
    standardized = {name: (value - feature_mean) / feature_std for name, value in values.items()}
    centroids = {name: value.mean(dim=0) for name, value in standardized.items()}
    direction = centroids["open"] - centroids["closed"]
    distance = direction.norm()
    if float(distance) < 1.0e-5:
        raise ValueError("open and closed pose samples are indistinguishable")
    direction_unit = direction / distance
    midpoint = 0.5 * (centroids["open"] + centroids["closed"])
    score_scale = 0.5 * distance
    scores = {
        name: ((value - midpoint) @ direction_unit) / score_scale
        for name, value in standardized.items()
    }
    closed_p90 = float(torch.quantile(scores["closed"], 0.90))
    open_p10 = float(torch.quantile(scores["open"], 0.10))
    if open_p10 <= closed_p90 + 0.10:
        raise ValueError(
            "open/closed calibration overlaps; record a flatter open palm and a clearer natural grasp"
        )
    # Entering open is deliberately stricter than leaving it. This ensures a
    # brief partially open pose cannot make the robot release its control mode.
    gap = open_p10 - closed_p90
    enter_threshold = closed_p90 + 0.65 * gap
    exit_threshold = closed_p90 + 0.35 * gap
    training_accuracy = torch.cat((scores["open"] > 0.0, scores["closed"] < 0.0)).float().mean()
    return {
        "schema": PROFILE_SCHEMA,
        "rate_hz": float(rate_hz),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "direction_unit": direction_unit,
        "midpoint": midpoint,
        "score_scale": score_scale,
        "enter_threshold": float(enter_threshold),
        "exit_threshold": float(exit_threshold),
        "training_accuracy": float(training_accuracy),
        "sample_counts": {name: int(value.shape[0]) for name, value in values.items()},
        "score_summary": {
            name: {
                "mean": float(value.mean()),
                "p10": float(torch.quantile(value, 0.10)),
                "p90": float(torch.quantile(value, 0.90)),
            }
            for name, value in scores.items()
        },
        "calibration_segments": {
            name: [segment.detach().to(device="cpu", dtype=torch.float32) for segment in segments[name]]
            for name in POSE_NAMES
        },
    }


def load_open_profile(path: str | Path) -> dict[str, object]:
    profile = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        schema = profile.get("schema") if isinstance(profile, dict) else "unknown"
        raise ValueError(f"{path} uses {schema}; rerun calibrate_open_mode.py")
    for name in (
        "feature_mean", "feature_std", "direction_unit", "midpoint", "score_scale",
        "enter_threshold", "exit_threshold",
    ):
        if name not in profile:
            raise ValueError(f"open profile is missing {name!r}")
    if float(profile["exit_threshold"]) >= float(profile["enter_threshold"]):
        raise ValueError("open profile hysteresis thresholds are invalid")
    return profile


@dataclass(frozen=True)
class OpenDecision:
    score: float
    is_open: bool
    candidate_open: bool
    candidate_samples: int


class OpenPoseEstimator:
    """Classify a frame-level open pose with confirmation and hysteresis."""

    def __init__(
        self,
        profile: dict[str, object],
        *,
        enter_confirmation_seconds: float = 1.0,
        exit_confirmation_seconds: float = 0.2,
    ) -> None:
        if enter_confirmation_seconds <= 0.0 or exit_confirmation_seconds <= 0.0:
            raise ValueError("open-pose confirmation times must be positive")
        self.profile = profile
        self.rate_hz = float(profile["rate_hz"])
        # An open palm is a deliberate HOLD command, so it must be sustained.
        # Releasing it is deliberately faster, allowing a new CW/CCW trend to
        # resume without an unnecessary two-second dead period.
        self.enter_confirmation_samples = max(
            1, int(round(self.rate_hz * enter_confirmation_seconds))
        )
        self.exit_confirmation_samples = max(
            1, int(round(self.rate_hz * exit_confirmation_seconds))
        )
        self.is_open = False
        self._candidate_open: bool | None = None
        self._candidate_samples = 0

    def reset(self) -> None:
        self.is_open = False
        self._candidate_open = None
        self._candidate_samples = 0

    def score(self, q: torch.Tensor) -> float:
        value = q.detach().to(device="cpu", dtype=torch.float32)
        if value.shape != (JOINT_DIM,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"q must be a finite [{JOINT_DIM}] pose")
        standardized = (value - self.profile["feature_mean"]) / self.profile["feature_std"]
        return float(
            ((standardized - self.profile["midpoint"]) @ self.profile["direction_unit"])
            / self.profile["score_scale"]
        )

    def update(self, q: torch.Tensor) -> OpenDecision:
        score = self.score(q)
        requested_open = (
            score >= float(self.profile["enter_threshold"])
            if not self.is_open
            else score > float(self.profile["exit_threshold"])
        )
        if requested_open != self.is_open:
            if requested_open != self._candidate_open:
                self._candidate_open = requested_open
                self._candidate_samples = 1
            else:
                self._candidate_samples += 1
            required_samples = (
                self.enter_confirmation_samples
                if requested_open
                else self.exit_confirmation_samples
            )
            if self._candidate_samples >= required_samples:
                self.is_open = requested_open
                self._candidate_open = None
                self._candidate_samples = 0
        else:
            self._candidate_open = None
            self._candidate_samples = 0
        return OpenDecision(score, self.is_open, requested_open, self._candidate_samples)
