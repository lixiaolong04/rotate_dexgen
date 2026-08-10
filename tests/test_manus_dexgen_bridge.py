from __future__ import annotations

import socket
import threading
import time

import numpy as np

from dexgen.xhand_keypoints import ACTUATED_JOINT_NAMES
from manus_dexgen_bridge.command import (
    RelativeKeypointCommand,
    joint_reorder_indices,
    safe_direct_joint_target,
)
from manus_dexgen_bridge.dexgen_client import LatestCommandClient
from manus_dexgen_bridge.eval_manus_guided import EVALUATOR as GUIDED_EVALUATOR
from manus_dexgen_bridge.eval_manus_tabletop_pickup import (
    EVALUATOR as TABLETOP_EVALUATOR,
)
from manus_dexgen_bridge.open_inference import OpenPoseEstimator, build_open_profile
from manus_dexgen_bridge.protocol import RetargetFrame, decode_frame, encode_frame


def test_manus_launchers_target_categorized_evaluator():
    expected = (
        GUIDED_EVALUATOR.parents[3]
        / "dexgen"
        / "commands"
        / "eval"
        / "eval_guided.py"
    )
    assert GUIDED_EVALUATOR == expected
    assert TABLETOP_EVALUATOR == expected
    assert expected.is_file()


def test_protocol_round_trip_socketpair():
    left, right = socket.socketpair()
    try:
        expected = RetargetFrame.create(7, ACTUATED_JOINT_NAMES, np.arange(21, dtype=np.float32))
        left.sendall(encode_frame(expected))
        actual = decode_frame(right)
        assert actual.sequence == 7
        assert actual.timestamp_ns == expected.timestamp_ns
        assert actual.valid
        assert actual.joint_names == ACTUATED_JOINT_NAMES
        assert np.array_equal(actual.q_target, expected.q_target)
    finally:
        left.close()
        right.close()


def test_latest_client_keeps_newest_frame():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    def send_frames() -> None:
        connection, _ = listener.accept()
        try:
            for sequence in range(3):
                frame = RetargetFrame.create(sequence, ACTUATED_JOINT_NAMES, np.full(21, sequence, dtype=np.float32))
                connection.sendall(encode_frame(frame))
                time.sleep(0.01)
        finally:
            connection.close()
            listener.close()

    thread = threading.Thread(target=send_frames, daemon=True)
    thread.start()
    client = LatestCommandClient(f"tcp://{host}:{port}", stale_timeout_s=0.5, reconnect_delay_s=0.05)
    try:
        deadline = time.monotonic() + 1.0
        snapshot = client.snapshot()
        while time.monotonic() < deadline:
            snapshot = client.snapshot()
            if snapshot.frame is not None and snapshot.frame.sequence == 2:
                break
            time.sleep(0.01)
        assert snapshot.status == "recording"
        assert snapshot.frame is not None
        assert snapshot.frame.sequence == 2
        assert np.all(snapshot.frame.q_target == 2.0)
    finally:
        client.close()
        thread.join(timeout=1.0)


def test_relative_command_calibration_and_limit():
    import torch

    controller = RelativeKeypointCommand(
        ACTUATED_JOINT_NAMES, gain=1.0, max_motion_m=0.001
    )
    glove_reference = torch.zeros(1, 21)
    robot_reference = torch.full((1, 21), 0.1)
    controller.arm(glove_reference, robot_reference)
    at_reference = controller.compute(glove_reference, robot_reference)
    assert at_reference.motion.shape == (1, 10, 3)
    assert at_reference.temporal_motion.shape == (1, 2, 10, 3)
    assert torch.allclose(at_reference.motion, torch.zeros_like(at_reference.motion), atol=1e-7)

    moved_glove = glove_reference.clone()
    moved_glove[:, 2] = 0.5
    command = controller.compute(moved_glove, robot_reference)
    assert float(command.motion.norm(dim=-1).max()) <= 0.001001
    assert float(command.motion.abs().max()) > 0.0

    controller.disarm("reset")
    assert not controller.armed


def test_relative_command_builds_timestamped_two_horizon_motion():
    import torch

    controller = RelativeKeypointCommand(
        ACTUATED_JOINT_NAMES,
        gain=1.0,
        max_motion_m=1.0,
        velocity_ema=0.0,
        max_keypoint_speed_m_s=10.0,
    )
    glove = torch.zeros(1, 21)
    robot = torch.zeros(1, 21)
    controller.arm(glove, robot)
    controller.compute(glove, robot, timestamp_s=1.0)
    moved = glove.clone()
    moved[:, 2] = 0.1
    command = controller.compute(moved, robot, timestamp_s=1.1)
    assert bool((command.keypoint_velocity.abs() > 0).any())
    first = command.temporal_motion[:, 0]
    second = command.temporal_motion[:, 1]
    assert not torch.allclose(first, second)
    assert torch.allclose(
        second - first, 0.1 * command.keypoint_velocity, atol=1.0e-6
    )


def test_joint_contract_reorders_by_name():
    reversed_names = tuple(reversed(ACTUATED_JOINT_NAMES))
    indices = joint_reorder_indices(reversed_names, ACTUATED_JOINT_NAMES)
    source = np.arange(21)[::-1]
    assert np.array_equal(source[list(indices)], np.arange(21))


def test_direct_joint_target_applies_joint_and_transition_limits():
    import torch

    glove = torch.zeros((1, 21))
    glove[:, 0] = 2.0
    glove[:, 1] = 0.4
    current = torch.zeros_like(glove)
    lower = torch.full_like(glove, -1.0)
    upper = torch.full_like(glove, 1.0)
    direct = safe_direct_joint_target(
        glove,
        current,
        lower,
        upper,
        max_joint_delta_rad=0.25,
    )
    assert float(direct.target[0, 0]) == 0.25
    assert float(direct.target[0, 1]) == 0.25
    assert bool(direct.joint_limit_clipped[0, 0])
    assert not bool(direct.joint_limit_clipped[0, 1])
    assert bool(direct.rate_limited[0, 0])
    assert bool(direct.rate_limited[0, 1])


def test_direct_joint_target_rejects_nonfinite_input():
    import pytest
    import torch

    glove = torch.zeros((1, 21))
    glove[:, 0] = torch.nan
    current = torch.zeros_like(glove)
    lower = torch.full_like(glove, -1.0)
    upper = torch.full_like(glove, 1.0)
    with pytest.raises(ValueError, match="glove_q contains"):
        safe_direct_joint_target(
            glove,
            current,
            lower,
            upper,
            max_joint_delta_rad=0.25,
        )


def test_open_pose_profile_has_hysteresis_and_confirmation():
    import torch

    generator = torch.Generator().manual_seed(7)
    closed = 0.02 * torch.randn((30, 21), generator=generator)
    opened = torch.ones((30, 21)) + 0.02 * torch.randn((30, 21), generator=generator)
    profile = build_open_profile(
        {"open": [opened[:15], opened[15:]], "closed": [closed[:15], closed[15:]]},
        rate_hz=10.0,
    )
    assert float(profile["enter_threshold"]) > float(profile["exit_threshold"])
    estimator = OpenPoseEstimator(
        profile, enter_confirmation_seconds=0.2, exit_confirmation_seconds=0.2
    )
    assert not estimator.update(opened[0]).is_open
    assert estimator.update(opened[1]).is_open
    assert estimator.update(closed[0]).is_open
    assert not estimator.update(closed[1]).is_open


def test_open_pose_default_requires_one_second_before_hold():
    import torch

    closed = torch.zeros((25, 21))
    opened = torch.ones((25, 21))
    profile = build_open_profile({"open": [opened], "closed": [closed]}, rate_hz=10.0)
    estimator = OpenPoseEstimator(profile)
    for sample in opened[:9]:
        assert not estimator.update(sample).is_open
    assert estimator.update(opened[9]).is_open
