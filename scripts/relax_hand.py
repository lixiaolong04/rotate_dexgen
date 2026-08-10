"""让手松开/失能: 给所有关节发 mode=0(官方工具每个测试收尾用它, 推断=失能/松手)。
最安全设计: kp/kd/speed/torque 全 0(不产生力), position=当前角度(万一 mode=0 是位置控制也不动)。
⚠️ mode=0 含义是【推断】的(非官方文档), 第一次试请【手放电源开关上】, 若手乱动立刻断电。
用法(容器内, privileged+host网络): python scripts/relax_hand.py [--secs 10]
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
from rl_isaaclab.deploy_xhand.joint_map import isaaclab_to_xhand
import hand_controller as hc

p = argparse.ArgumentParser()
p.add_argument("--secs", type=float, default=10.0, help="保持 mode=0 多久(秒); 期间你徒手扳手指试松紧")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
hw = XhandHWInterface(cfg)
hw.connect()
xt = isaaclab_to_xhand(hw.get_joint_pos())   # 当前角度(Xhand 序)


def send_relax():
    cmds = []
    for i in range(21):
        c = hc.Green_JointCommand_User()
        c.position = float(xt[i])   # 当前角度(万一 mode=0 是位置控制 -> 不动)
        c.kp = 0.0; c.kd = 0.0; c.speed = 0.0; c.torque = 0.0   # 全 0: 不产生任何力
        c.mode = 0
        cmds.append(c)
    hw.hand.set_joint_cmd_list(cmds)


print(f"[relax] 发 mode=0 + 全 0 力 (失能/松手), 保持 {args.secs:.0f}s...")
print("[relax] ⚠️ 手放电源开关上! 手乱动立刻断电。现在【徒手扳手指】试试松不松...")
try:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.secs:
        send_relax()
        time.sleep(0.1)
    print(f"[relax] 已保持 mode=0 {args.secs:.0f}s。手指是松的吗? 松=mode=0 就是失能/松手。")
finally:
    hw.disconnect()
print("[relax] 完成(已断开)。")
