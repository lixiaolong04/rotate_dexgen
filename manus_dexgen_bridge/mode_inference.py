"""Window-integrated MANUS motion-trend inference for CW/CCW mode selection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch


PROFILE_SCHEMA = "dexgen.manus_mode_profile.v2"
MULTI_CONTEXT_PROFILE_SCHEMA = "dexgen.manus_mode_profile.v3_multicontext"
MODE_NAMES = ("cw", "ccw")
CALIBRATION_NAMES = ("empty_cw", "empty_ccw", "object_cw", "object_ccw")
FLAT_POINT_DIM = 30
TEMPORAL_AREA_DIM = FLAT_POINT_DIM * (FLAT_POINT_DIM - 1) // 2
FEATURE_DIM = TEMPORAL_AREA_DIM + 3


def trend_feature(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an integrated direction feature and per-step motion activity.

    ``points`` is a wrist-frame trajectory shaped [T,10,3].  Translation
    increments capture coordinated finger travel.  The cross product between
    centroid-relative position and increment captures clockwise/counter-
    clockwise circulation even when a cyclic gesture ends near its start.
    """

    if points.ndim != 3 or tuple(points.shape[1:]) != (10, 3) or points.shape[0] < 2:
        raise ValueError(f"points must have shape [T>=2,10,3], got {tuple(points.shape)}")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points contain NaN or Inf")
    previous = points[:-1]
    delta = points[1:] - previous
    centered_spatial = previous - previous.mean(dim=1, keepdim=True)
    circulation = torch.cross(centered_spatial, delta, dim=-1).sum(dim=1).mean(dim=0)

    # A human finger-roll is primarily a temporal ordering pattern (one finger
    # hands the object to the next), not a rigid 3-D circle.  The antisymmetric
    # lag-one area of all 30 keypoint coordinates captures that ordering.  It
    # is phase-insensitive and flips sign when the gesture is time-reversed.
    flat = points.reshape(points.shape[0], FLAT_POINT_DIM)
    flat = flat - flat.mean(dim=0, keepdim=True)
    lag_area = (
        flat[:-1].transpose(0, 1) @ flat[1:]
        - flat[1:].transpose(0, 1) @ flat[:-1]
    ) / float(flat.shape[0] - 1)
    upper = torch.triu_indices(FLAT_POINT_DIM, FLAT_POINT_DIM, offset=1)
    temporal_area = lag_area[upper[0], upper[1]]
    feature = torch.cat((temporal_area, circulation))
    activity = delta.square().mean().sqrt()
    return feature, activity


def sequence_features(points: torch.Tensor, window_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    if window_steps < 3:
        raise ValueError("window_steps must be at least 3")
    if points.shape[0] < window_steps:
        return torch.empty((0, FEATURE_DIM)), torch.empty((0,))
    features = []
    activities = []
    for end in range(window_steps, points.shape[0] + 1):
        feature, activity = trend_feature(points[end - window_steps : end])
        features.append(feature)
        activities.append(activity)
    return torch.stack(features), torch.stack(activities)


def build_mode_profile(
    segments: dict[str, list[torch.Tensor]], *, rate_hz: float, window_seconds: float,
) -> dict[str, object]:
    if rate_hz <= 0.0 or window_seconds <= 0.0:
        raise ValueError("rate_hz and window_seconds must be positive")
    window_steps = max(3, int(round(rate_hz * window_seconds)))
    by_mode: dict[str, torch.Tensor] = {}
    activity_by_mode: dict[str, torch.Tensor] = {}
    segment_counts = {}
    for mode_name in MODE_NAMES:
        feature_chunks = []
        activity_chunks = []
        valid_segments = 0
        for segment in segments.get(mode_name, []):
            features, activities = sequence_features(segment.float(), window_steps)
            if features.shape[0]:
                feature_chunks.append(features)
                activity_chunks.append(activities)
                valid_segments += 1
        if not feature_chunks:
            raise ValueError(
                f"No usable {mode_name} segment; record at least {window_seconds:.1f}s continuously"
            )
        by_mode[mode_name] = torch.cat(feature_chunks)
        activity_by_mode[mode_name] = torch.cat(activity_chunks)
        segment_counts[mode_name] = valid_segments

    # All three angular components have the same physical unit. Per-axis
    # standardization can amplify a nearly constant, phase-dependent nuisance
    # axis and overpower the true rotation axis, so classification is fitted
    # directly in angular-trend space.
    feature_mean = torch.zeros_like(by_mode["cw"][0])
    feature_std = torch.ones_like(feature_mean)
    standardized = by_mode
    centroids = {name: values.mean(dim=0) for name, values in standardized.items()}
    direction_vector = centroids["cw"] - centroids["ccw"]
    centroid_distance = direction_vector.norm()
    if float(centroid_distance) < 1.0e-6:
        raise ValueError("CW and CCW calibration features are indistinguishable")
    direction_unit = direction_vector / centroid_distance
    midpoint = 0.5 * (centroids["cw"] + centroids["ccw"])
    score_scale = 0.5 * centroid_distance

    scores = {
        name: ((values - midpoint) @ direction_unit) / score_scale
        for name, values in standardized.items()
    }
    correct = torch.cat((scores["cw"] > 0.0, scores["ccw"] < 0.0)).float().mean()
    all_activity = torch.cat(tuple(activity_by_mode.values()))
    activity_threshold = 0.25 * torch.quantile(all_activity, 0.20)

    return {
        "schema": PROFILE_SCHEMA,
        "rate_hz": float(rate_hz),
        "window_seconds": float(window_seconds),
        "window_steps": window_steps,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "direction_unit": direction_unit,
        "midpoint": midpoint,
        "score_scale": score_scale,
        "activity_threshold": activity_threshold,
        "training_accuracy": float(correct),
        "segment_counts": segment_counts,
        "feature_counts": {name: int(value.shape[0]) for name, value in by_mode.items()},
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
            for name in MODE_NAMES
        },
    }


def build_multi_context_mode_profile(
    segments: dict[str, list[torch.Tensor]], *, rate_hz: float, window_seconds: float,
) -> dict[str, object]:
    """Fit CW/CCW using separate empty-hand and object-holding prototypes."""
    if rate_hz <= 0.0 or window_seconds <= 0.0:
        raise ValueError("rate_hz and window_seconds must be positive")
    window_steps = max(3, int(round(rate_hz * window_seconds)))
    features_by_name: dict[str, torch.Tensor] = {}
    activities_by_name: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    saved: dict[str, list[torch.Tensor]] = {}
    for name in CALIBRATION_NAMES:
        chunks = []
        activity_chunks = []
        valid = []
        for segment in segments.get(name, []):
            features, activities = sequence_features(segment.float(), window_steps)
            if features.shape[0]:
                chunks.append(features)
                activity_chunks.append(activities)
                valid.append(segment.detach().to(device="cpu", dtype=torch.float32))
        if not chunks:
            raise ValueError(f"No usable {name}; record at least {window_seconds:.1f}s continuously")
        features_by_name[name] = torch.cat(chunks)
        activities_by_name[name] = torch.cat(activity_chunks)
        counts[name] = len(valid)
        saved[name] = valid
    prototypes = torch.stack([features_by_name[name].mean(dim=0) for name in CALIBRATION_NAMES])
    prototype_modes = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    def directional_score(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = (values[:, None] - prototypes[None]).square().mean(dim=-1)
        cw_distance = distances[:, prototype_modes == 0].min(dim=1).values
        ccw_distance = distances[:, prototype_modes == 1].min(dim=1).values
        return ccw_distance - cw_distance, distances.argmin(dim=1)

    raw_by_name = {}
    nearest_by_name = {}
    for name, values in features_by_name.items():
        raw_by_name[name], nearest_by_name[name] = directional_score(values)
    all_raw = torch.cat(tuple(raw_by_name.values()))
    score_scale = torch.quantile(all_raw.abs(), 0.50).clamp_min(1.0e-12)
    scores = {name: value / score_scale for name, value in raw_by_name.items()}
    direction_correct = []
    context_correct = []
    for index, name in enumerate(CALIBRATION_NAMES):
        expected_mode = 0 if name.rsplit("_", 1)[1] == "cw" else 1
        direction_correct.append((scores[name] > 0.0) if expected_mode == 0 else (scores[name] < 0.0))
        context_correct.append(nearest_by_name[name] == index)
    return {
        "schema": MULTI_CONTEXT_PROFILE_SCHEMA,
        "rate_hz": float(rate_hz),
        "window_seconds": float(window_seconds),
        "window_steps": window_steps,
        "prototypes": prototypes,
        "prototype_names": list(CALIBRATION_NAMES),
        "prototype_modes": prototype_modes,
        "score_scale": score_scale,
        "activity_threshold": 0.25 * torch.quantile(torch.cat(tuple(activities_by_name.values())), 0.20),
        "training_accuracy": float(torch.cat(direction_correct).float().mean()),
        "context_accuracy": float(torch.cat(context_correct).float().mean()),
        "segment_counts": counts,
        "feature_counts": {name: int(value.shape[0]) for name, value in features_by_name.items()},
        "score_summary": {
            name: {"mean": float(value.mean()), "p10": float(torch.quantile(value, .10)), "p90": float(torch.quantile(value, .90))}
            for name, value in scores.items()
        },
        "calibration_segments": saved,
    }


def load_mode_profile(path: str | Path) -> dict[str, object]:
    profile = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(profile, dict) or profile.get("schema") not in (PROFILE_SCHEMA, MULTI_CONTEXT_PROFILE_SCHEMA):
        raise ValueError(
            f"{path} uses {profile.get('schema') if isinstance(profile, dict) else 'unknown'}; "
            f"rerun calibrate_auto_mode.py to create required {PROFILE_SCHEMA}"
        )
    if profile["schema"] == MULTI_CONTEXT_PROFILE_SCHEMA:
        for name in ("prototypes", "prototype_modes", "score_scale", "activity_threshold"):
            if name not in profile or not isinstance(profile[name], torch.Tensor):
                raise ValueError(f"Multi-context mode profile is missing tensor {name!r}")
        return profile
    for name in (
        "feature_mean", "feature_std", "direction_unit", "midpoint",
        "score_scale", "activity_threshold",
    ):
        if name not in profile or not isinstance(profile[name], torch.Tensor):
            raise ValueError(f"Mode profile is missing tensor {name!r}")
    return profile


@dataclass(frozen=True)
class TrendDecision:
    score: float
    integrated_score: float
    activity_m: float
    active: bool
    suggested_mode: str | None


class MotionTrendModeEstimator:
    """Infer a stable mode using a rolling window, EMA integration and hysteresis."""

    def __init__(
        self,
        profile: dict[str, object],
        *,
        initial_mode: str,
        integration_seconds: float = 0.8,
        threshold: float = 0.35,
        confirmation_seconds: float = 0.4,
        cooldown_seconds: float = 1.0,
    ) -> None:
        if initial_mode not in MODE_NAMES:
            raise ValueError(f"initial_mode must be one of {MODE_NAMES}")
        if integration_seconds <= 0.0 or confirmation_seconds <= 0.0 or cooldown_seconds < 0.0:
            raise ValueError("invalid auto-mode timing")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.profile = profile
        self.rate_hz = float(profile["rate_hz"])
        self.window_steps = int(profile["window_steps"])
        self.points: deque[torch.Tensor] = deque(maxlen=self.window_steps)
        self.decay = math.exp(-1.0 / (self.rate_hz * integration_seconds))
        self.threshold = float(threshold)
        self.confirmation_steps = max(1, int(round(self.rate_hz * confirmation_seconds)))
        self.cooldown_steps = max(0, int(round(self.rate_hz * cooldown_seconds)))
        self.current_mode = initial_mode
        self.integrated_score = 0.0
        self.candidate: str | None = None
        self.candidate_steps = 0
        self.cooldown = 0

    def reset(self, *, keep_mode: bool = True) -> None:
        self.points.clear()
        self.integrated_score = 0.0
        self.candidate = None
        self.candidate_steps = 0
        self.cooldown = 0
        if not keep_mode:
            self.current_mode = "cw"

    def update(self, points: torch.Tensor) -> TrendDecision:
        sample = points.detach().to(device="cpu", dtype=torch.float32)
        if sample.shape != (10, 3) or not bool(torch.isfinite(sample).all()):
            self.reset()
            return TrendDecision(0.0, 0.0, 0.0, False, None)
        self.points.append(sample)
        if len(self.points) < self.window_steps:
            return TrendDecision(0.0, self.integrated_score, 0.0, False, None)

        feature, activity = trend_feature(torch.stack(tuple(self.points)))
        if self.profile["schema"] == MULTI_CONTEXT_PROFILE_SCHEMA:
            prototypes = self.profile["prototypes"]
            modes = self.profile["prototype_modes"]
            distances = (feature[None] - prototypes).square().mean(dim=-1)
            cw = distances[modes == 0].min()
            ccw = distances[modes == 1].min()
            score = float((ccw - cw) / self.profile["score_scale"])
        else:
            standardized = (feature - self.profile["feature_mean"]) / self.profile["feature_std"]
            score = float(
                ((standardized - self.profile["midpoint"]) @ self.profile["direction_unit"])
                / self.profile["score_scale"]
            )
        active = float(activity) >= float(self.profile["activity_threshold"])
        target_score = score if active else 0.0
        self.integrated_score = (
            self.decay * self.integrated_score + (1.0 - self.decay) * target_score
        )
        if self.cooldown > 0:
            self.cooldown -= 1

        candidate = None
        if self.integrated_score >= self.threshold:
            candidate = "cw"
        elif self.integrated_score <= -self.threshold:
            candidate = "ccw"
        if candidate is None or candidate == self.current_mode:
            self.candidate = None
            self.candidate_steps = 0
            return TrendDecision(score, self.integrated_score, float(activity), active, None)
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_steps = 1
        else:
            self.candidate_steps += 1
        suggested = None
        if self.cooldown == 0 and self.candidate_steps >= self.confirmation_steps:
            self.current_mode = candidate
            suggested = candidate
            self.cooldown = self.cooldown_steps
            self.candidate = None
            self.candidate_steps = 0
        return TrendDecision(score, self.integrated_score, float(activity), active, suggested)
