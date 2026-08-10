import numpy as np
import os
import threading

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv
from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
from rl_isaaclab.deploy_xhand.warmup_traj import N1, N2, SPREAD_JOINTS
from rl_isaaclab.utils.keyboard_listener import KeyboardListener
from rl_isaaclab.utils.misc import ThreadSafeValue


def _listener(initial=0):
    state = ThreadSafeValue(initial)
    tactile = ThreadSafeValue(0)
    return KeyboardListener(state, tactile), state


def test_enter_starts_policy_and_space_does_not():
    listener, state = _listener(1)
    listener._handle_key(" ")
    assert state.get() == 1
    listener.last_deploy_state_flag = 3
    listener._handle_key("\n")
    assert state.get() == 3


def test_enter_before_warmup_completion_is_ignored():
    listener, state = _listener(1)
    listener.last_deploy_state_flag = 0
    listener._handle_key("\n")
    assert state.get() == 1


def test_w_freezes_and_resumes_policy_without_opening():
    listener, state = _listener(3)
    listener._handle_key("w")
    assert state.get() == 1
    listener._handle_key("w")
    assert state.get() == 3


def test_w_after_warmup_never_falls_back_to_home():
    listener, state = _listener(1)
    listener.last_deploy_state_flag = 0
    listener._handle_key("w")
    assert state.get() == 1


def test_q_requests_slow_home_state():
    listener, state = _listener(3)
    listener._handle_key("q")
    assert state.get() == 4


def test_go_home_reverses_warmup_slowly(monkeypatch):
    cfg = DeployCfg()
    cfg.keyboard_listen = False
    cfg.warm_up = False
    start = np.linspace(-0.4, 1.0, 21, dtype=np.float32)
    hw = FakeHWInterface(pos=start)
    sent = []

    original_send = hw.set_joint_targets

    def record(targets, kp, kd, ff_torque=None):
        sent.append(np.asarray(targets, np.float32).copy())
        original_send(targets, kp, kd, ff_torque)

    hw.set_joint_targets = record
    monkeypatch.setattr("rl_isaaclab.deploy_xhand.deploy_env.time.sleep", lambda _: None)

    env = XhandInhandRotateDeployEnv(cfg, hw)
    assert env.go_home()

    spread = np.array([name in SPREAD_JOINTS for name in cfg.joint_names])
    assert len(sent) == N2 + N1
    assert np.allclose(sent[0], start)
    assert np.allclose(sent[-1], 0.0)
    assert np.allclose(sent[N2 - 1][~spread], 0.0)
    assert np.allclose(sent[N2 - 1][spread], start[spread])
    assert np.allclose(env.cur_targets.cpu().numpy(), 0.0)
    assert np.allclose(env.prev_targets.cpu().numpy(), 0.0)


def test_native_sdk_info_can_be_silenced_without_hiding_later_output(capfd):
    hw = XhandHWInterface.__new__(XhandHWInterface)
    hw.cfg = type("Cfg", (), {"sdk_quiet": True})()
    hw._native_log_lock = threading.Lock()

    with hw._quiet_native_sdk():
        os.write(1, b"native info should be hidden\n")
        os.write(2, b"native error should be hidden\n")
    print("application output remains visible")

    captured = capfd.readouterr()
    assert "native info" not in captured.out
    assert "native error" not in captured.err
    assert "application output remains visible" in captured.out
