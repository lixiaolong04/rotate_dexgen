from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import socket
import threading

import numpy as np
import pytest
import torch

from dexgen.deploy.dexgen_runtime import DexGenRuntime, RLCommandRuntime
from dexgen.deploy.hardware_client import XHandHardwareClient
from dexgen.deploy.manus_auto_real import AutoGestureController
from dexgen.deploy.manus_dexgen_real import (
    Controller,
    JointTargetSmoother,
    arrays,
    format_joint_errors,
    joint_error_codes,
    pregrasp_stop_reason,
)
from dexgen.deploy.protocol import recv_message, send_message
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg


def hardware_state() -> dict:
    return {
        "status": "ok",
        "joint_pos": [0.0] * 21,
        "joint_target": [0.0] * 21,
        "tactile_raw": [0.0] * 5,
        "torque": [0.0] * 21,
        "temperature": [25.0] * 21,
        "error_code": [0.0] * 21,
    }


def test_protocol_socketpair_roundtrip() -> None:
    left, right = socket.socketpair()
    try:
        send_message(left, {"op": "target", "target": [0.0] * 21})
        assert recv_message(right)["op"] == "target"
        send_message(right, hardware_state())
        assert recv_message(left)["temperature"] == [25.0] * 21
    finally:
        left.close()
        right.close()


def test_hardware_client_validates_and_relaxes() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    operations: list[str] = []

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            while True:
                request = recv_message(connection)
                operations.append(request["op"])
                if request["op"] == "relax":
                    send_message(connection, {"status": "ok", "relaxed": True})
                    break
                send_message(connection, hardware_state())

    worker = threading.Thread(target=serve)
    worker.start()
    client = XHandHardwareClient("127.0.0.1", listener.getsockname()[1])
    with pytest.raises(ValueError):
        client.target(np.zeros(20))
    assert arrays(client.hold())[0].shape == (21,)
    client.close(relax=True)
    worker.join(timeout=2.0)
    listener.close()
    assert not worker.is_alive()
    assert operations == ["ping", "hold", "relax"]


def test_hardware_state_and_error_validation() -> None:
    state = hardware_state()
    q, target, tactile, temperature = arrays(state)
    assert (q.shape, target.shape, tactile.shape, temperature.shape) == (
        (21,),
        (21,),
        (5,),
        (21,),
    )
    assert not np.any(joint_error_codes(state))
    state["error_code"][7] = 3
    assert joint_error_codes(state)[7] == 3
    state["error_code"][7] = 1501028
    message = format_joint_errors(joint_error_codes(state), [f"joint_{i}" for i in range(21)])
    assert "joint_7" in message
    assert "过温" in message


def test_rl_observation_keeps_all_five_tactile_channels() -> None:
    runtime = RLCommandRuntime.__new__(RLCommandRuntime)
    runtime.device = torch.device("cpu")
    runtime.lower = torch.full((1, 21), -1.0)
    runtime.upper = torch.full((1, 21), 1.0)
    runtime.frames = deque(maxlen=30)
    tactile = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    runtime.update(torch.zeros(1, 21), torch.zeros(1, 21), tactile)
    assert len(runtime.frames) == 30
    torch.testing.assert_close(runtime.frames[-1][0, -5:], tactile[0])


def test_dexgen_state_contract_is_83_dimensions() -> None:
    cfg = DeployCfg()
    runtime = DexGenRuntime.__new__(DexGenRuntime)
    runtime.device = torch.device("cpu")
    runtime.joint_names = tuple(cfg.joint_names)
    runtime.history_steps = 4
    runtime.history = deque(maxlen=4)
    frame = runtime.append_state(
        torch.zeros(1, 21),
        torch.zeros(1, 21),
        torch.arange(5, dtype=torch.float32).reshape(1, 5),
    )
    assert frame.shape == (1, 83)
    torch.testing.assert_close(frame[0, -5:], torch.arange(5, dtype=torch.float32))


def test_joint_target_smoother_limits_step_and_acceleration() -> None:
    smoother = JointTargetSmoother(
        dt=0.05,
        max_step_rad=1.0 / 24.0,
        max_accel_rad_s2=8.0,
    )
    smoother.reset(np.zeros(21, dtype=np.float32))
    outputs = [
        smoother.step(
            np.ones(21, dtype=np.float32),
            np.full(21, -2.0, dtype=np.float32),
            np.full(21, 2.0, dtype=np.float32),
        )
        for _ in range(8)
    ]
    increments = np.diff(
        np.vstack((np.zeros((1, 21), dtype=np.float32), outputs)), axis=0
    )
    assert np.max(np.abs(increments)) <= 1.0 / 24.0 + 1.0e-6
    velocity = increments / 0.05
    acceleration = np.diff(
        np.vstack((np.zeros((1, 21), dtype=np.float32), velocity)), axis=0
    ) / 0.05
    assert np.max(np.abs(acceleration)) <= 8.0 + 1.0e-4
    assert increments[0, 0] < increments[1, 0]

    # Abrupt goal reversals must change velocity through the acceleration
    # limiter instead of snapping the command to each new goal.
    commands = []
    for index in range(40):
        goal = np.full(21, 0.5 if index % 4 < 2 else -0.5, dtype=np.float32)
        commands.append(
            smoother.step(
                goal,
                np.full(21, -2.0, dtype=np.float32),
                np.full(21, 2.0, dtype=np.float32),
            )
        )
    increments = np.diff(np.vstack((outputs[-1][None], commands)), axis=0)
    acceleration = np.diff(increments / 0.05, axis=0) / 0.05
    assert np.max(np.abs(acceleration)) <= 8.0 + 1.0e-4


def test_pregrasp_stops_on_contacts_or_single_finger_force_limit() -> None:
    kwargs = {
        "contact_threshold": 0.1,
        "min_fingers": 2,
        "force_limit": 1.0,
    }
    assert pregrasp_stop_reason(np.zeros(5), **kwargs) is None
    assert "2 contacts" in pregrasp_stop_reason(
        np.array([0.11, 0.0, 0.12, 0.0, 0.0]), **kwargs
    )
    assert "force limit" in pregrasp_stop_reason(
        np.array([1.2, 0.0, 0.0, 0.0, 0.0]), **kwargs
    )


def test_pregrasp_holds_immediately_without_two_extra_frames(monkeypatch) -> None:
    monkeypatch.setattr("dexgen.deploy.manus_dexgen_real.time.sleep", lambda _: None)
    cfg = DeployCfg()

    class FakeHardware:
        def __init__(self) -> None:
            self.target_value = np.zeros(21, dtype=np.float32)
            self.after_zero = False
            self.pregrasp_targets: list[np.ndarray] = []
            self.hold_calls = 0

        def state(self, tactile: list[float] | None = None) -> dict:
            result = hardware_state()
            result["joint_pos"] = self.target_value.tolist()
            result["joint_target"] = self.target_value.tolist()
            result["tactile_raw"] = tactile or [0.0] * 5
            return result

        def target(self, target) -> dict:
            self.target_value = np.asarray(target, dtype=np.float32).copy()
            tactile = None
            if self.after_zero:
                self.pregrasp_targets.append(self.target_value.copy())
                if len(self.pregrasp_targets) >= 2:
                    tactile = [400.0, 0.0, 400.0, 0.0, 0.0]
            return self.state(tactile)

        def zero_tactile(self) -> dict:
            self.after_zero = True
            return self.state()

        def hold(self) -> dict:
            self.hold_calls += 1
            return self.state()

    class Resettable:
        def reset(self) -> None:
            pass

    controller = Controller.__new__(Controller)
    controller.args = SimpleNamespace(
        temp_limit=68.0,
        pregrasp_start_temp_limit=55.0,
        pregrasp_force_limit_n=1.0,
    )
    controller.cfg = cfg
    controller.names = list(cfg.joint_names)
    controller.lower = np.asarray(cfg.dof_lower, dtype=np.float32)
    controller.upper = np.asarray(cfg.dof_upper, dtype=np.float32)
    controller.hw = FakeHardware()
    controller.command_target = np.zeros(21, dtype=np.float32)
    controller.desired_target = np.zeros(21, dtype=np.float32)
    controller.smoother = JointTargetSmoother(
        dt=0.05, max_step_rad=1.0 / 24.0, max_accel_rad_s2=8.0
    )
    controller.smoothed_tactile = np.zeros(5, dtype=np.float32)
    controller.inference_lock = threading.Lock()
    controller.dexgen = Resettable()
    controller.rl = Resettable()

    controller.pregrasp(controller.hw.state())

    assert len(controller.hw.pregrasp_targets) == 2
    assert controller.hw.hold_calls == 1
    np.testing.assert_allclose(
        controller.command_target, controller.hw.pregrasp_targets[-1]
    )


def test_auto_open_gesture_holds_measured_pose_then_resumes() -> None:
    class FakeHardware:
        def __init__(self) -> None:
            self.hold_calls = 0
            self.target = np.linspace(0.0, 0.2, 21, dtype=np.float32)

        def hold(self) -> dict:
            self.hold_calls += 1
            result = hardware_state()
            result["joint_pos"] = self.target.tolist()
            result["joint_target"] = self.target.tolist()
            return result

    class FakeWorker:
        def __init__(self) -> None:
            self.control_mode = "hold"
            self.desired_mode = "cw"
            self.set_calls = []

        def snapshot(self):
            return SimpleNamespace(
                source_status="recording",
                error=None,
                control_mode=self.control_mode,
                desired_mode=self.desired_mode,
                decision=None,
                open_decision=None,
                samples=20,
            )

        def set_current_mode(self, mode, *, reset):
            self.set_calls.append((mode, reset))

    class Resettable:
        def __init__(self) -> None:
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

    controller = AutoGestureController.__new__(AutoGestureController)
    controller.hw = FakeHardware()
    controller.auto_worker = FakeWorker()
    controller.auto_enabled = True
    controller.manual_paused = False
    controller.auto_hold_reason = None
    controller.auto_resume_control = "dexgen_rl"
    controller.control = "dexgen_rl"
    controller.resume_control = "dexgen_rl"
    controller.mode = "cw"
    controller.command_target = np.zeros(21, dtype=np.float32)
    controller.desired_target = np.zeros(21, dtype=np.float32)
    controller.smoother = JointTargetSmoother(
        dt=0.05, max_step_rad=1.0 / 24.0, max_accel_rad_s2=8.0
    )
    controller.inference_lock = threading.Lock()
    controller.dexgen = Resettable()
    controller.rl = Resettable()
    controller.last_auto_report = float("inf")

    state = controller.update_external_control(hardware_state())
    assert controller.control == "hold"
    assert controller.auto_hold_reason == "open palm"
    assert controller.hw.hold_calls == 1
    np.testing.assert_allclose(controller.command_target, controller.hw.target)

    controller.update_external_control(state)
    assert controller.hw.hold_calls == 1

    controller.auto_worker.control_mode = "cw"
    controller.update_external_control(state)
    assert controller.control == "dexgen_rl"
    assert controller.auto_hold_reason is None
    assert controller.dexgen.resets == 1
    assert controller.rl.resets == 1

    controller.auto_worker.desired_mode = "ccw"
    controller.auto_worker.control_mode = "ccw"
    controller.update_external_control(state)
    assert controller.mode == "ccw"
    assert controller.dexgen.resets == 2
    assert controller.rl.resets == 2
