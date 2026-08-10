"""验证 cmd.kp/kd 到底有没有用(直接, 不依赖物体):
命令手保持当前姿态, 你【用手推一根手指】, 脚本测它被推动多少(挠度 rad)。
软(低kp)=推得动挠度大; 硬(高kp)=推不动挠度小。
跑两次对比: --kp 1 和 --kp 100。挠度差很多=kp 有用; 几乎一样=cmd.kp 被 SDK 忽略(得走固件寄存器)。
用法(容器内): python scripts/kp_feel_test.py --kp 1 [--kd 0.4] [--joint 5] [--secs 6]
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface

p = argparse.ArgumentParser()
p.add_argument("--kp", type=float, required=True)
p.add_argument("--kd", type=float, default=0.4)
p.add_argument("--joint", type=int, default=5, help="IsaacLab 关节下标(默认5=食指指根弯曲)")
p.add_argument("--secs", type=float, default=6.0)
p.add_argument("--mode", type=int, default=4, help="joint_cmd_mode: 4=kp/kd生效(官方工具用), 3=忽略kp/kd")
p.add_argument("--speed", type=float, default=0.0, help="mode=4(MIT)下 speed=目标速度, 必须0(否则驱动关节运动→握紧)!")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
cfg.kp = args.kp
cfg.kd = args.kd
cfg.joint_cmd_mode = args.mode
cfg.cmd_speed = args.speed
hw = XhandHWInterface(cfg)
hw.connect()
JN = cfg.joint_names[args.joint]

home = hw.get_joint_pos()
start = float(home[args.joint])
print(f"\n[kp_feel] kp={args.kp} kd={args.kd}  测试关节 {JN}  当前角={start:.3f}")
print(f">>> 现在【用手把这根手指推/掰】{args.secs:.0f} 秒, 用差不多的力, 感受软硬...")

defl = []
t0 = time.perf_counter()
try:
    while time.perf_counter() - t0 < args.secs:
        hw.set_joint_targets(home, cfg.kp, cfg.kd)   # 持续命令保持 home
        cur = float(hw.get_joint_pos()[args.joint])
        defl.append(abs(cur - start))
        time.sleep(0.02)
finally:
    hw.disconnect()

defl = np.array(defl)
print(f"\n[kp_feel] kp={args.kp}: 被推动【最大挠度={defl.max():.3f} rad】 均值={defl.mean():.3f}")
print("  对比: 跑 --kp 1 和 --kp 100, 若最大挠度差很多 -> cmd.kp 有用(软硬可调);")
print("        若两者几乎一样 -> cmd.kp 被 SDK 忽略, 得走固件寄存器(问厂商 kp 寄存器地址)。")
