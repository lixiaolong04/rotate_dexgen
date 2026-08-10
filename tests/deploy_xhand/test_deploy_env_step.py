# tests/deploy_xhand/test_deploy_env_step.py
import numpy as np
import torch
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv, saturate


def _env(pos=None):
    cfg = DeployCfg()
    cfg.keyboard_listen = False   # 自动跑策略分支
    cfg.warm_up = False
    return XhandInhandRotateDeployEnv(cfg, FakeHWInterface(pos=pos)), cfg


def _exp_targets(cfg, prev=0.0):
    lo = torch.tensor(cfg.dof_lower).reshape(1, -1)
    hi = torch.tensor(cfg.dof_upper).reshape(1, -1)
    t = torch.full((1, 21), prev + cfg.action_scale * cfg.action_scale_mul)
    return saturate(t, lo, hi)


def test_step_target_update_and_send():
    env, cfg = _env()           # pos=0 -> prev_targets=0
    env.reset()
    a = torch.ones(1, 21)       # action = +1
    obs = env.step(a)
    exp = _exp_targets(cfg)     # saturate(0 + (1/24)*1, lo, hi)
    assert torch.allclose(env.cur_targets, exp, atol=1e-6)
    assert np.allclose(env.hw.last_targets, exp.squeeze(0).numpy(), atol=1e-6)  # 已下发硬件
    assert torch.allclose(env.prev_targets, env.cur_targets)                     # prev<-cur
    assert obs["obs"].shape == (1, 141)   # 2026-06-17 对齐训练v2: 186→141


def test_action_clipped():
    env, cfg = _env()
    env.reset()
    env.step(torch.full((1, 21), 5.0))   # 超 clip_actions=1 -> clip 到 1 再 *scale
    assert torch.allclose(env.cur_targets, _exp_targets(cfg), atol=1e-6)


def test_history_advances():
    env, _ = _env()
    env.reset()
    before = env.obs_buf_lag_history[0, -1].clone()
    env.hw._set_state(pos=np.full(21, 0.3, np.float32))
    env.step(torch.zeros(1, 21))
    assert not torch.allclose(env.obs_buf_lag_history[0, -1], before)  # 新帧入栈
