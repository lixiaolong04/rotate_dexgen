"""XHand keypoint geometry shared by rollout collection and deployment.

DexterityGen represents hand motion with PIP joints and fingertips.  The XHand
USD exposes the distal-link origin, not the fingertip pad centre, so the latter
is reconstructed from the fixed ``*_dip_sensor_joint`` transforms in the URDF.
"""

from __future__ import annotations

import torch


FINGERS = ("thumb", "index", "mid", "ring", "pinky")
ACTUATED_JOINT_NAMES = (
    "right_hand_thumb_mcp0_joint",
    "right_hand_thumb_mcp1_joint",
    "right_hand_thumb_pip_joint",
    "right_hand_thumb_dip_joint",
    "right_hand_index_mcp0_joint",
    "right_hand_index_mcp1_joint",
    "right_hand_index_pip_joint",
    "right_hand_index_dip_joint",
    "right_hand_mid_mcp0_joint",
    "right_hand_mid_mcp1_joint",
    "right_hand_mid_pip_joint",
    "right_hand_mid_dip_joint",
    "right_hand_ring_mcp0_joint",
    "right_hand_ring_mcp1_joint",
    "right_hand_ring_pip_joint",
    "right_hand_ring_dip_joint",
    "right_hand_ltspalm_joint",
    "right_hand_pinky_mcp0_joint",
    "right_hand_pinky_mcp1_joint",
    "right_hand_pinky_pip_joint",
    "right_hand_pinky_dip_joint",
)
PIP_BODY_NAMES = tuple(f"right_hand_{finger}_pip_link" for finger in FINGERS)
DIP_BODY_NAMES = tuple(f"right_hand_{finger}_dip_link" for finger in FINGERS)
KEYPOINT_NAMES = tuple(
    name for finger in FINGERS for name in (f"{finger}_pip", f"{finger}_fingertip_pad")
)
KEYPOINT_FRAME = "articulation_root_wrist"
KEYPOINT_SEMANTICS = "five PIP origins interleaved with five fingertip pad centres"

# Translation of each DIP sensor/pad centre in its parent DIP-link frame.
# Source: assets/xhand_green/xhand_right/urdf/xhand_right.urdf.
FINGERTIP_OFFSETS = (
    (0.0, 0.0181728187019061, 0.00791114210898185),
    (0.0079111421089901, 0.0, 0.0181728187019025),
    (0.00791114210899072, 0.0, 0.0181728187019025),
    (0.00791114210899012, 0.0, 0.0181728187019025),
    (0.00791114209997283, 0.0, 0.018172818705828),
)


def representation_metadata(
    joint_names: list[str] | tuple[str, ...] = ACTUATED_JOINT_NAMES,
) -> dict[str, object]:
    """Return the serializable XHand representation contract."""
    return {
        "joint_names": list(joint_names),
        "keypoint_names": list(KEYPOINT_NAMES),
        "keypoint_frame": KEYPOINT_FRAME,
        "fingertip_offsets_in_dip_frame": [list(value) for value in FINGERTIP_OFFSETS],
        "keypoint_semantics": KEYPOINT_SEMANTICS,
    }


def validate_representation_metadata(
    metadata: dict[str, object], *, require_joint_names: bool = False
) -> None:
    """Reject shape-compatible data whose XHand semantics are incompatible."""
    expected = representation_metadata()
    required = ("keypoint_names", "keypoint_frame", "fingertip_offsets_in_dip_frame", "keypoint_semantics")
    for key in required:
        if key not in metadata:
            raise ValueError(f"DexGen metadata is missing {key!r}")
        if metadata[key] != expected[key]:
            raise ValueError(
                f"DexGen representation mismatch for {key}: expected {expected[key]!r}, "
                f"got {metadata[key]!r}"
            )
    if require_joint_names and "joint_names" not in metadata:
        raise ValueError("DexGen metadata is missing 'joint_names'")
    if "joint_names" in metadata:
        names = metadata["joint_names"]
        if not isinstance(names, (list, tuple)):
            raise ValueError("DexGen joint_names must be a list")
        if set(names) != set(ACTUATED_JOINT_NAMES) or len(names) != len(ACTUATED_JOINT_NAMES):
            raise ValueError("DexGen joint_names do not match the canonical XHand 21-DOF set")

# Joint origins and axes copied from xhand_right.urdf.  Each chain is ordered
# from the articulation root to the fingertip.  Keeping these constants beside
# KEYPOINT_NAMES gives collection, exact FK, and deployment one canonical
# representation.
_FINGER_CHAINS = (
    (
        ("right_hand_thumb_mcp0_joint", (0.020417, 0.028872, 0.035928), (0.61087, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ("right_hand_thumb_mcp1_joint", (0.0060305, 0.022506, 0.0055), (0.0, 0.0, -0.2618), (1.0, 0.0, 0.0)),
        ("right_hand_thumb_pip_joint", (0.0, 0.0345, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ("right_hand_thumb_dip_joint", (0.0, 0.037417, -0.0045), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ),
    (
        ("right_hand_index_mcp0_joint", (0.0045, 0.032961, 0.096293), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ("right_hand_index_mcp1_joint", (0.0055, 0.0, 0.0233), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_index_pip_joint", (0.0, 0.0, 0.0345), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_index_dip_joint", (-0.0045, 0.0, 0.037417), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ),
    (
        ("right_hand_mid_mcp0_joint", (0.0045, 0.0086396, 0.098793), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ("right_hand_mid_mcp1_joint", (0.0055, 0.0, 0.0233), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_mid_pip_joint", (0.0, 0.0, 0.0345), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_mid_dip_joint", (-0.0045, 0.0, 0.037417), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ),
    (
        ("right_hand_ring_mcp0_joint", (0.0045, -0.014087, 0.094293), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ("right_hand_ring_mcp1_joint", (0.0055, 0.0, 0.0233), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_ring_pip_joint", (0.0, 0.0, 0.0345), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_ring_dip_joint", (-0.0045, 0.0, 0.037417), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ),
    (
        ("right_hand_ltspalm_joint", (0.0115, -0.0265, 0.041), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("right_hand_pinky_mcp0_joint", (-0.007, -0.011478, 0.044292), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ("right_hand_pinky_mcp1_joint", (0.0055, 0.0, 0.0233), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_pinky_pip_joint", (0.0, 0.0, 0.0345), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right_hand_pinky_dip_joint", (-0.0045, 0.0, 0.037417), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ),
)


def quat_rotate(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized wxyz quaternions."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, xyz = quaternion[..., :1], quaternion[..., 1:]
    return vectors + 2.0 * (
        w * torch.cross(xyz, vectors, dim=-1)
        + torch.cross(xyz, torch.cross(xyz, vectors, dim=-1), dim=-1)
    )


def quat_rotate_inverse(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    return quat_rotate(conjugate, vectors)


def _rpy_matrix(rpy: tuple[float, float, float], reference: torch.Tensor) -> torch.Tensor:
    """Return the URDF fixed-axis roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = (torch.as_tensor(value, device=reference.device, dtype=reference.dtype) for value in rpy)
    sr, cr = roll.sin(), roll.cos()
    sp, cp = pitch.sin(), pitch.cos()
    sy, cy = yaw.sin(), yaw.cos()
    return torch.stack(
        (
            cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr,
        )
    ).reshape(3, 3)


def _axis_angle_matrix(axis: tuple[float, float, float], angle: torch.Tensor) -> torch.Tensor:
    """Batched Rodrigues rotation for a constant unit joint axis."""
    axis_tensor = torch.as_tensor(axis, device=angle.device, dtype=angle.dtype)
    axis_tensor = axis_tensor / axis_tensor.norm().clamp_min(1.0e-8)
    x, y, z = axis_tensor
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    outer = axis_tensor[:, None] * axis_tensor[None, :]
    identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
    cosine = angle.cos()[..., None, None]
    sine = angle.sin()[..., None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


def _canonical_joint_positions(
    joint_pos: torch.Tensor, joint_names: list[str] | tuple[str, ...] | None
) -> torch.Tensor:
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must have shape [B,J], got {tuple(joint_pos.shape)}")
    if joint_names is None:
        if joint_pos.shape[1] != len(ACTUATED_JOINT_NAMES):
            raise ValueError(f"Expected {len(ACTUATED_JOINT_NAMES)} joints, got {joint_pos.shape[1]}")
        return joint_pos
    if len(joint_names) != joint_pos.shape[1] or len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be unique and match the joint_pos width")
    missing = [name for name in ACTUATED_JOINT_NAMES if name not in joint_names]
    extra = [name for name in joint_names if name not in ACTUATED_JOINT_NAMES]
    if missing or extra:
        raise ValueError(f"XHand joint-name mismatch: missing={missing}, extra={extra}")
    indices = torch.tensor(
        [joint_names.index(name) for name in ACTUATED_JOINT_NAMES],
        device=joint_pos.device,
    )
    return joint_pos.index_select(1, indices)


def xhand_keypoints_fk(
    joint_pos: torch.Tensor,
    joint_names: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Exact URDF FK for the ten canonical PIP/fingertip keypoints.

    Args:
        joint_pos: Joint positions with shape ``[B,21]``.
        joint_names: Column names for ``joint_pos``.  If omitted, columns must
            already follow ``ACTUATED_JOINT_NAMES``.

    Returns:
        Wrist/root-frame points with shape ``[B,10,3]`` in ``KEYPOINT_NAMES`` order.
    """
    canonical = _canonical_joint_positions(joint_pos, joint_names)
    joint_index = {name: index for index, name in enumerate(ACTUATED_JOINT_NAMES)}
    batch = canonical.shape[0]
    keypoints = []
    for finger_index, chain in enumerate(_FINGER_CHAINS):
        position = torch.zeros(batch, 3, device=canonical.device, dtype=canonical.dtype)
        rotation = torch.eye(3, device=canonical.device, dtype=canonical.dtype).expand(batch, -1, -1).clone()
        pip_position = None
        for name, xyz, rpy, axis in chain:
            origin = torch.as_tensor(xyz, device=canonical.device, dtype=canonical.dtype)
            position = position + torch.einsum("bij,j->bi", rotation, origin)
            fixed_rotation = _rpy_matrix(rpy, canonical)
            rotation = rotation @ fixed_rotation @ _axis_angle_matrix(axis, canonical[:, joint_index[name]])
            if name.endswith("_pip_joint"):
                pip_position = position
        if pip_position is None:
            raise RuntimeError(f"Finger chain {finger_index} has no PIP joint")
        tip_offset = torch.as_tensor(
            FINGERTIP_OFFSETS[finger_index], device=canonical.device, dtype=canonical.dtype
        )
        fingertip = position + torch.einsum("bij,j->bi", rotation, tip_offset)
        keypoints.extend((pip_position, fingertip))
    return torch.stack(keypoints, dim=1)


def exact_keypoint_motion(
    current_joint_pos: torch.Tensor,
    target_joint_pos: torch.Tensor,
    joint_names: list[str] | tuple[str, ...] | None = None,
    max_motion_m: float | None = 0.03,
) -> torch.Tensor:
    """Compute ``FK(q_target) - FK(q_current)`` in the canonical wrist frame."""
    if current_joint_pos.shape != target_joint_pos.shape:
        raise ValueError("current_joint_pos and target_joint_pos must have identical shapes")
    motion = xhand_keypoints_fk(target_joint_pos, joint_names) - xhand_keypoints_fk(
        current_joint_pos, joint_names
    )
    if max_motion_m is not None:
        if max_motion_m <= 0.0:
            raise ValueError("max_motion_m must be positive or None")
        norms = motion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        motion = motion * (max_motion_m / norms).clamp_max(1.0)
    return motion


def resolve_body_ids(body_names: list[str]) -> tuple[list[int], list[int]]:
    required = PIP_BODY_NAMES + DIP_BODY_NAMES
    missing = [name for name in required if name not in body_names]
    if missing:
        raise RuntimeError(f"USD articulation is missing DexGen keypoint bodies: {missing}")
    return (
        [body_names.index(name) for name in PIP_BODY_NAMES],
        [body_names.index(name) for name in DIP_BODY_NAMES],
    )


def wrist_keypoints(base, pip_body_ids: list[int], dip_body_ids: list[int]) -> torch.Tensor:
    """Return interleaved PIP/pad-centre points in the wrist/root frame."""
    pip_w = base.hand.data.body_pos_w[:, pip_body_ids]
    dip_w = base.hand.data.body_pos_w[:, dip_body_ids]
    dip_quat_w = base.hand.data.body_quat_w[:, dip_body_ids]
    offsets = torch.as_tensor(FINGERTIP_OFFSETS, device=dip_w.device, dtype=dip_w.dtype)
    fingertip_w = dip_w + quat_rotate(dip_quat_w, offsets.unsqueeze(0).expand_as(dip_w))

    points_w = torch.stack((pip_w, fingertip_w), dim=2).flatten(1, 2)
    root_pos_w = base.hand.data.root_pos_w[:, None, :]
    root_quat_w = base.hand.data.root_quat_w[:, None, :].expand(-1, len(KEYPOINT_NAMES), -1)
    return quat_rotate_inverse(root_quat_w, points_w - root_pos_w)
