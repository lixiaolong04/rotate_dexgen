# scripts/sim2real_obs_check.py
"""给定 dof_pos(21,IsaacLab序) + contacts(5), 用 XhandInhandRotateDeployEnv 构帧, 打印 62 维分解。
B5 时: 把真机同姿态 obs 与 laptop 仿真(play.py 同姿态 obs)逐位对比。
用法: python scripts/sim2real_obs_check.py            # 用零姿态自检(FakeHW)
真机: 把 FakeHWInterface 换成 XhandHWInterface(cfg) + hw.connect(), pos/contacts 即真机读数。
"""
import numpy as np
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv

cfg = DeployCfg()
cfg.keyboard_listen = False
cfg.warm_up = False
pos = np.zeros(21, np.float32)        # 替换为真机 hw.get_joint_pos()
contacts = np.zeros(5, np.float32)    # 替换为真机 hw.get_tactile_raw()
env = XhandInhandRotateDeployEnv(cfg, FakeHWInterface(pos=pos, tactile=contacts))
obs = env.reset()
f = env.obs_buf_lag_history[0, -1].numpy()   # 最近一帧 62 维(reset 后全帧相同)
print("frame[0:21]  unscale(dof_pos):", np.round(f[0:21], 3))
print("frame[21:42] targets:        ", np.round(f[21:42], 3))
print("frame[42:47] contacts:       ", np.round(f[42:47], 3))
print("frame[47:62] contact_pos(=0):", np.round(f[47:62], 3))
print("obs(186) finite:", bool(np.isfinite(obs["obs"].numpy()).all()))
