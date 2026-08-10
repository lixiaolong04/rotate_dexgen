"""Relative calibration and ten-keypoint command generation for DexGen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from dexgen.xhand_keypoints import xhand_keypoints_fk


def joint_reorder_indices(
    source_names: Sequence[str], target_names: Sequence[str]
) -> tuple[int, ...]:
    """Indices that reorder a named source vector into target-name order."""

    source = tuple(source_names)
    target = tuple(target_names)
    if len(source) != len(target) or len(set(source)) != len(source):
        raise ValueError("source/target joint contracts have incompatible sizes")
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        raise ValueError(
            f"joint contract mismatch: missing={missing}, unexpected={unexpected}"
        )
    lookup = {name: index for index, name in enumerate(source)}
    return tuple(lookup[name] for name in target)


@dataclass(frozen=True)
class DirectJointTarget:
    """A MANUS joint target after simulator limits and transition limiting."""

    target: torch.Tensor
    joint_limit_clipped: torch.Tensor
    rate_limited: torch.Tensor


def safe_direct_joint_target(
    glove_q: torch.Tensor,
    current_target: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    *,
    max_joint_delta_rad: float,
) -> DirectJointTarget:
    """Prepare an absolute RetargetX target for direct XHand PD control."""

    tensors = {
        "glove_q": glove_q,
        "current_target": current_target,
        "lower_limits": lower_limits,
        "upper_limits": upper_limits,
    }
    expected_shape = glove_q.shape
    if glove_q.ndim != 2 or glove_q.shape[-1] != 21:
        raise ValueError(f"glove_q must be shaped [B,21], got {tuple(glove_q.shape)}")
    for label, value in tensors.items():
        if value.shape != expected_shape:
            raise ValueError(
                f"{label} must have shape {tuple(expected_shape)}, got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} contains NaN or Inf")
    if not bool((lower_limits <= upper_limits).all()):
        raise ValueError("lower_limits must not exceed upper_limits")
    if max_joint_delta_rad <= 0.0:
        raise ValueError("max_joint_delta_rad must be positive")

    limited_to_joints = torch.clamp(glove_q, lower_limits, upper_limits)
    joint_limit_clipped = limited_to_joints.ne(glove_q)
    delta = limited_to_joints - current_target
    limited_delta = delta.clamp(-max_joint_delta_rad, max_joint_delta_rad)
    rate_limited = limited_delta.ne(delta)
    target = torch.clamp(current_target + limited_delta, lower_limits, upper_limits)
    return DirectJointTarget(
        target=target,
        joint_limit_clipped=joint_limit_clipped,
        rate_limited=rate_limited,
    )


@dataclass(frozen=True)
class RelativeCommand:
    """One calibrated goal and its residual from the executing robot hand."""

    motion: torch.Tensor
    temporal_motion: torch.Tensor
    goal_keypoints: torch.Tensor
    glove_keypoints: torch.Tensor
    keypoint_velocity: torch.Tensor


class RelativeKeypointCommand:
    """Map relative GeoRT motion onto the robot's calibration-time grasp.

    The calibration stores keypoints rather than joint offsets.  Consequently,
    different joint configurations with the same task-space motion do not leak
    GeoRT's absolute neutral-pose bias into the executing XHand grasp.
    """

    def __init__(
        self,
        joint_names: Sequence[str],
        *,
        gain: float = 0.35,
        max_motion_m: float = 0.03,
        velocity_ema: float = 0.5,
        max_keypoint_speed_m_s: float = 0.25,
    ) -> None:
        if gain <= 0.0:
            raise ValueError("MANUS command gain must be positive")
        if max_motion_m <= 0.0:
            raise ValueError("MANUS command limit must be positive")
        if not 0.0 <= velocity_ema < 1.0:
            raise ValueError("MANUS velocity EMA must be in [0,1)")
        if max_keypoint_speed_m_s <= 0.0:
            raise ValueError("MANUS keypoint speed limit must be positive")
        self.joint_names = tuple(joint_names)
        self.gain = float(gain)
        self.max_motion_m = float(max_motion_m)
        self.velocity_ema = float(velocity_ema)
        self.max_keypoint_speed_m_s = float(max_keypoint_speed_m_s)
        self._glove_reference: torch.Tensor | None = None
        self._robot_reference: torch.Tensor | None = None
        self._previous_glove_keypoints: torch.Tensor | None = None
        self._previous_timestamp_s: float | None = None
        self._keypoint_velocity: torch.Tensor | None = None
        self.disarm_reason = "not calibrated"

    @property
    def armed(self) -> bool:
        return self._glove_reference is not None and self._robot_reference is not None

    def arm(self, glove_q: torch.Tensor, robot_q: torch.Tensor) -> None:
        self._validate_q(glove_q, "glove_q")
        self._validate_q(robot_q, "robot_q")
        if glove_q.shape != robot_q.shape:
            raise ValueError("glove_q and robot_q must have the same batch shape")
        self._glove_reference = xhand_keypoints_fk(glove_q, self.joint_names).detach().clone()
        self._robot_reference = xhand_keypoints_fk(robot_q, self.joint_names).detach().clone()
        self._previous_glove_keypoints = self._glove_reference.clone()
        self._previous_timestamp_s = None
        self._keypoint_velocity = torch.zeros_like(self._glove_reference)
        self.disarm_reason = ""

    def disarm(self, reason: str) -> None:
        self._glove_reference = None
        self._robot_reference = None
        self._previous_glove_keypoints = None
        self._previous_timestamp_s = None
        self._keypoint_velocity = None
        self.disarm_reason = str(reason)

    def compute(
        self,
        glove_q: torch.Tensor,
        robot_q: torch.Tensor,
        *,
        timestamp_s: float | None = None,
        horizons_s: tuple[float, float] = (0.1, 0.2),
    ) -> RelativeCommand:
        if not self.armed:
            raise RuntimeError(f"MANUS command is disarmed: {self.disarm_reason}")
        self._validate_q(glove_q, "glove_q")
        self._validate_q(robot_q, "robot_q")
        glove_keypoints = xhand_keypoints_fk(glove_q, self.joint_names)
        robot_keypoints = xhand_keypoints_fk(robot_q, self.joint_names)
        goal = self._robot_reference + self.gain * (glove_keypoints - self._glove_reference)
        if min(horizons_s) <= 0.0 or horizons_s[1] <= horizons_s[0]:
            raise ValueError("MANUS temporal horizons must be positive and ordered")
        if timestamp_s is not None:
            timestamp_s = float(timestamp_s)
            if (
                self._previous_timestamp_s is not None
                and timestamp_s > self._previous_timestamp_s
            ):
                dt = timestamp_s - self._previous_timestamp_s
                instantaneous = self.gain * (
                    glove_keypoints - self._previous_glove_keypoints
                ) / dt
                speed = instantaneous.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
                instantaneous = instantaneous * (
                    self.max_keypoint_speed_m_s / speed
                ).clamp_max(1.0)
                self._keypoint_velocity = (
                    self.velocity_ema * self._keypoint_velocity
                    + (1.0 - self.velocity_ema) * instantaneous
                )
            if (
                self._previous_timestamp_s is None
                or timestamp_s > self._previous_timestamp_s
            ):
                self._previous_glove_keypoints = glove_keypoints.detach().clone()
                self._previous_timestamp_s = timestamp_s
        velocity = self._keypoint_velocity
        motion = self._limit_motion(goal - robot_keypoints)
        temporal_motion = torch.stack(
            [
                self._limit_motion(
                    goal - robot_keypoints + velocity * float(horizon)
                )
                for horizon in horizons_s
            ],
            dim=1,
        )
        return RelativeCommand(
            motion=motion,
            temporal_motion=temporal_motion,
            goal_keypoints=goal,
            glove_keypoints=glove_keypoints,
            keypoint_velocity=velocity,
        )

    def _limit_motion(self, motion: torch.Tensor) -> torch.Tensor:
        norms = motion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return motion * (self.max_motion_m / norms).clamp_max(1.0)

    @staticmethod
    def _validate_q(q: torch.Tensor, label: str) -> None:
        if q.ndim != 2 or q.shape[-1] != 21:
            raise ValueError(f"{label} must be shaped [B,21], got {tuple(q.shape)}")
        if not bool(torch.isfinite(q).all()):
            raise ValueError(f"{label} contains NaN or Inf")
