# tests/deploy_xhand/test_policy_runner.py
import os, torch, pytest
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv
from rl_isaaclab.deploy_xhand.policy_runner import PolicyRunner

CKPT = "ckpts/unilab_v2/student_5001.pth"   # 2026-06-17 部署主力 v2 student (obs 141 + priv 9); 旧 t603 是 obs 186 不兼容

@pytest.mark.skipif(not os.path.exists(CKPT), reason="ckpt 不在")
def test_load_and_infer_shape():
    cfg = DeployCfg()
    cfg.keyboard_listen = False
    cfg.warm_up = False
    env = XhandInhandRotateDeployEnv(cfg, FakeHWInterface())
    runner = PolicyRunner(env, CKPT, device="cpu")
    obs = env.reset()
    a = runner.act(obs)
    assert a.shape == (1, 21)
    assert torch.isfinite(a).all()
    assert (a >= -1).all() and (a <= 1).all()  # clamp 到 [-1,1]
