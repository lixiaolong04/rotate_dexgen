# tests/deploy_xhand/test_deploy_env_obs.py
import numpy as np
import torch
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv, unscale


def _env(pos=None, tac=None):
    cfg = DeployCfg()
    cfg.keyboard_listen = False   # 离线: 不起 pynput
    cfg.warm_up = False           # 离线: 不回放/不 sleep(3)
    return XhandInhandRotateDeployEnv(cfg, FakeHWInterface(pos=pos, tactile=tac)), cfg


def test_spaces_for_proprioadapt():
    env, _ = _env()
    assert env.observation_space.shape[1] == 141   # 2026-06-17 对齐训练v2: 丢15维零 contact_pos (186→141)
    assert env.action_space.shape[1] == 21
    assert env.prop_hist_len == 30
    assert env.num_obs == 141


def test_reset_fills_history_no_zeros_obs():
    env, cfg = _env(pos=np.full(21, 0.1, np.float32), tac=np.full(5, 2.0, np.float32))
    obs = env.reset()
    assert obs["obs"].shape == (1, 141)
    assert obs["proprio_hist"].shape == (1, 30, 47)
    # reset 后所有帧被当前帧填满 -> dof_pos 槽 == unscale(0.1)
    lo = torch.tensor(cfg.dof_lower_raw)   # 2026-06-17 obs 归一化用 raw 限位(对齐训练)
    hi = torch.tensor(cfg.dof_upper_raw)
    expect_pos = unscale(torch.full((21,), 0.1), lo, hi).numpy()
    frame0 = obs["proprio_hist"][0, 0].numpy()
    assert np.allclose(frame0[0:21], expect_pos, atol=1e-5)


def test_frame_layout():
    # 用真实量级假力(原始 nf), 期望按 cfg.force_scale/contact_threshold 算(不硬编码标定常量)
    cfg0 = DeployCfg()
    above = cfg0.contact_threshold / np.asarray(cfg0.force_scale, np.float32)  # 每指刚好到阈值的原始力
    tac_raw = (np.array([2, 3, 4, 5, 6], np.float32)) * above    # 都明显 > 阈值
    env, cfg = _env(pos=np.full(21, 0.2, np.float32), tac=tac_raw)
    obs = env.reset()
    frame = obs["proprio_hist"][0, -1].numpy()   # 末帧(reset 后所有帧相同)
    lo = torch.tensor(cfg.dof_lower_raw)   # 2026-06-17 obs 归一化用 raw 限位(对齐训练)
    hi = torch.tensor(cfg.dof_upper_raw)
    assert np.allclose(frame[0:21], unscale(torch.full((21,), 0.2), lo, hi).numpy(), atol=1e-5)
    # at_reset 填充: 目标槽 = hand_dof_pos 原始(0.2), 照搬 sharpa
    assert np.allclose(frame[21:42], np.full(21, 0.2, np.float32), atol=1e-5)
    # contact 槽 = force_scale 后(噪声门未触发); 期望按 cfg 算
    expect = tac_raw * np.asarray(cfg.force_scale, np.float32)
    expect[expect < cfg.contact_threshold] = 0.0
    assert np.allclose(frame[42:47], expect, atol=1e-4)
    # 2026-06-17 对齐训练v2: 丢掉 contact_pos[47:62], 帧到 47 为止(无 contact_pos 槽)
    assert frame.shape[0] == 47


def test_continuous_no_threshold_floor():
    # 2026-06-17 对齐训练连续模式: 【无阈值地板】(原噪声门清零是 sim2real gap, 已删). 低于阈值的力照传(clamp+平滑后).
    cfg0 = DeployCfg()
    half = (cfg0.contact_threshold / np.asarray(cfg0.force_scale, np.float32)) * 0.5  # 半阈值的原始力
    tac = np.full(5, float(half), np.float32)
    env, cfg = _env(pos=np.zeros(21, np.float32), tac=tac)
    sensed, _ = env.get_tactile_info()
    expect = np.clip(tac * np.asarray(cfg.force_scale, np.float32), 0.0, 4.0)  # 无地板照传(首帧 prev=current 故 sensed=current)
    assert np.allclose(sensed.numpy().reshape(-1), expect, atol=1e-5)
