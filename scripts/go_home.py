"""恢复初始姿态(安全): 把手从当前姿态【反向回放暖启动轨迹】张开回 home(全 0)。
反向 = 先松开弯曲(un-flex)、再收侧摆(un-abduct), 避免拇指卡死。
⚠️ 会动手、会松开物体(物体会掉/释放)。停掉 B8 后跑这个复位。
用法(容器内, privileged+host网络): python scripts/go_home.py [--step-dt 0.02]
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface

p = argparse.ArgumentParser()
p.add_argument("--step-dt", type=float, default=0.02)
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
traj = np.load(cfg.init_traj_path)        # (60,21) 两段式 open->grasp
hw = XhandHWInterface(cfg)
hw.connect()
try:
    print("[home] 反向回放(先松弯曲再收侧摆, 不卡拇指)张开到 home...")
    for j in range(len(traj) - 1, -1, -1):
        hw.set_joint_targets(traj[j], cfg.kp, cfg.kd)
        time.sleep(args.step_dt)
    print("[home] 已到张开姿态(全 0)。")
finally:
    hw.disconnect()
print("[home] 完成。")
