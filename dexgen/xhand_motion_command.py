"""Convert XHand joint-space commands to DexGen keypoint motion commands.

The adapter uses PhysX geometric Jacobians at the current simulated state.  A
Jacobian is reported at each link centre of mass, so it is shifted to the PIP
actor origin or physical fingertip-pad centre before being rotated into the
wrist frame.  This avoids treating a 21-D joint action as keypoint motion.
"""

from __future__ import annotations

import torch

from dexgen.xhand_keypoints import FINGERTIP_OFFSETS, quat_rotate, quat_rotate_inverse


def _jacobian_body_ids(hand, body_ids: list[int]) -> list[int]:
    # PhysX omits the root-body Jacobian for fixed-base articulations.
    return [body_id - 1 for body_id in body_ids] if hand.is_fixed_base else body_ids


def _jacobian_joint_ids(hand, joint_ids: list[int]) -> list[int]:
    # Floating articulations prepend six root DOFs to Jacobian columns.
    return joint_ids if hand.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]


def _point_jacobians_world(
    base,
    body_ids: list[int],
    points_w: torch.Tensor,
    joint_ids: list[int],
) -> torch.Tensor:
    """Return point linear Jacobians with shape [N,P,3,J] in world axes."""
    hand = base.hand
    jacobians = hand.root_physx_view.get_jacobians()
    body_columns = _jacobian_body_ids(hand, body_ids)
    joint_columns = _jacobian_joint_ids(hand, joint_ids)
    body_jacobian = jacobians[:, body_columns, :, :][:, :, :, joint_columns]
    linear = body_jacobian[:, :, :3]
    angular = body_jacobian[:, :, 3:]

    # PhysX link linear velocity/Jacobian is expressed at the link COM.
    radius_w = points_w - hand.data.body_com_pos_w[:, body_ids]
    omega_columns = angular.transpose(-1, -2)  # [N,P,J,3]
    shifted = torch.cross(omega_columns, radius_w.unsqueeze(-2), dim=-1).transpose(-1, -2)
    return linear + shifted


def keypoint_jacobian_wrist(
    base,
    pip_body_ids: list[int],
    dip_body_ids: list[int],
    joint_ids: list[int],
) -> torch.Tensor:
    """Return interleaved PIP/pad Jacobians [N,10,3,J] in wrist axes."""
    hand = base.hand
    pip_w = hand.data.body_pos_w[:, pip_body_ids]
    dip_w = hand.data.body_pos_w[:, dip_body_ids]
    dip_quat_w = hand.data.body_quat_w[:, dip_body_ids]
    offsets = torch.as_tensor(FINGERTIP_OFFSETS, device=dip_w.device, dtype=dip_w.dtype)
    fingertip_w = dip_w + quat_rotate(dip_quat_w, offsets.unsqueeze(0).expand_as(dip_w))

    pip_jacobian_w = _point_jacobians_world(base, pip_body_ids, pip_w, joint_ids)
    tip_jacobian_w = _point_jacobians_world(base, dip_body_ids, fingertip_w, joint_ids)
    jacobian_w = torch.stack((pip_jacobian_w, tip_jacobian_w), dim=2).flatten(1, 2)

    vectors_w = jacobian_w.permute(0, 1, 3, 2)  # [N,10,J,3]
    root_quat_w = hand.data.root_quat_w[:, None, None, :].expand(
        -1, vectors_w.shape[1], vectors_w.shape[2], -1
    )
    return quat_rotate_inverse(root_quat_w, vectors_w).permute(0, 1, 3, 2)


def joint_delta_to_keypoint_motion(
    base,
    joint_delta: torch.Tensor,
    pip_body_ids: list[int],
    dip_body_ids: list[int],
    joint_ids: list[int],
    max_motion_m: float | None = 0.03,
) -> torch.Tensor:
    """Linearize a joint displacement into wrist-frame keypoint offsets."""
    jacobian = keypoint_jacobian_wrist(base, pip_body_ids, dip_body_ids, joint_ids)
    motion = torch.einsum("nkdj,nj->nkd", jacobian, joint_delta)
    if max_motion_m is not None:
        norms = motion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        motion = motion * (max_motion_m / norms).clamp_max(1.0)
    return motion


def measured_keypoint_velocity_wrist(
    base,
    pip_body_ids: list[int],
    dip_body_ids: list[int],
) -> torch.Tensor:
    """Compute instantaneous PIP/pad velocities directly from rigid-body data."""
    hand = base.hand
    pip_w = hand.data.body_pos_w[:, pip_body_ids]
    dip_w = hand.data.body_pos_w[:, dip_body_ids]
    dip_quat_w = hand.data.body_quat_w[:, dip_body_ids]
    offsets = torch.as_tensor(FINGERTIP_OFFSETS, device=dip_w.device, dtype=dip_w.dtype)
    tip_w = dip_w + quat_rotate(dip_quat_w, offsets.unsqueeze(0).expand_as(dip_w))

    def point_velocity(body_ids: list[int], point_w: torch.Tensor) -> torch.Tensor:
        com_w = hand.data.body_com_pos_w[:, body_ids]
        linear_w = hand.data.body_com_lin_vel_w[:, body_ids]
        angular_w = hand.data.body_com_ang_vel_w[:, body_ids]
        return linear_w + torch.cross(angular_w, point_w - com_w, dim=-1)

    velocity_w = torch.stack(
        (point_velocity(pip_body_ids, pip_w), point_velocity(dip_body_ids, tip_w)), dim=2
    ).flatten(1, 2)
    root_quat_w = hand.data.root_quat_w[:, None, :].expand(-1, velocity_w.shape[1], -1)
    return quat_rotate_inverse(root_quat_w, velocity_w)


def jacobian_velocity_error(
    base,
    pip_body_ids: list[int],
    dip_body_ids: list[int],
    joint_ids: list[int],
) -> tuple[float, float]:
    """Return mean/max |J qdot - measured point velocity| in m/s."""
    jacobian = keypoint_jacobian_wrist(base, pip_body_ids, dip_body_ids, joint_ids)
    qvel = base.hand_dof_vel[:, joint_ids]
    predicted = torch.einsum("nkdj,nj->nkd", jacobian, qvel)
    measured = measured_keypoint_velocity_wrist(base, pip_body_ids, dip_body_ids)
    error = (predicted - measured).norm(dim=-1)
    return float(error.mean()), float(error.max())
