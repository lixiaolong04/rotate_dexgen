"""B2: 逐个 Xhand SDK 关节 id 发【小幅低速】指令, 人工观察哪根手指哪节动 + 读回数值确认,
反推 Xhand id -> IsaacLab 关节, 回填 joint_map.ISAACLAB2XHAND_IDX。

⚠️⚠️ 这是【第一次主动发运动指令】, 手会动! 跑前确保: 手固定好、工作区清空、电源开关在手边能随时断。
   每个关节: 先等你回车 -> 才动 +0.2rad(低速) -> 读回看哪个 id 变 -> 你肉眼记哪根指动 -> 归零。
   (顺带验证我们的指令路径: 若发了指令关节【不动】, 说明 joint_cmd_mode 取值不对, 需调。)

用法(容器内, privileged+host网络): python scripts/inspect_xhand_joints.py [--angle 0.2] [--speed 100]
"""
import argparse
import time

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
import hand_controller as hc

p = argparse.ArgumentParser()
p.add_argument("--angle", type=float, default=0.05, help="单关节小幅测试角度(rad); 侧摆行程小, 默认 0.05 最稳; 弯曲想看明显可 --angle 0.15")
p.add_argument("--speed", type=float, default=100.0, help="首次运动用低速(cfg 默认 350 偏快)")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
hw = XhandHWInterface(cfg)
hw.connect()   # 已含正确顺序(open 后建 GreenHand) + 沉降


def read_xhand_raw():
    """直接按 Xhand id 顺序读 21 关节(绕过 joint_map), per-id。"""
    return [float(hw.hand.get_joint_state(i, True)[2].position) for i in range(21)]


def send_xhand_order(vec):
    """直接按 Xhand id 顺序下发(绕过 joint_map, 用于发现映射), 低速。"""
    cmds = []
    for i in range(21):
        c = hc.Green_JointCommand_User()
        c.position = float(vec[i]); c.kp = cfg.kp; c.kd = cfg.kd
        c.speed = float(args.speed); c.torque = cfg.cmd_torque; c.mode = cfg.joint_cmd_mode
        cmds.append(c)
    hw.hand.set_joint_cmd_list(cmds)


print("=" * 64)
print("IsaacLab 关节顺序(对照填 joint_map 用):")
for i, n in enumerate(cfg.joint_names):
    print(f"  isaaclab[{i:2d}] = {n}")
print("=" * 64)
print(f"参数: angle={args.angle}rad  speed={args.speed}  kp={cfg.kp} kd={cfg.kd} mode={cfg.joint_cmd_mode}")
print("Xhand id 解剖分组(预期): 0-3拇 / 4-7食 / 8-11中 / 12-15无名 / 16-19小 / 20=ltspalm")

base = read_xhand_raw()   # 当前姿态作基准, 测完每个关节回到这
print(f"\n当前 21 关节(Xhand 序): {[round(x,3) for x in base]}")

mapping = {}
try:
    for xid in range(21):
        input(f"\n>>> 按回车: 让 Xhand id={xid} 动 +{args.angle}rad (低速). 准备好看手 / 手按急停...")
        v = list(base)
        v[xid] = base[xid] + args.angle
        before = read_xhand_raw()
        send_xhand_order(v)
        time.sleep(1.2)
        after = read_xhand_raw()
        deltas = [after[i] - before[i] for i in range(21)]
        moved = max(range(21), key=lambda i: abs(deltas[i]))
        print(f"    数值上动最多: id={moved}  Δ={deltas[moved]:+.3f}  (命令的是 id={xid})")
        if abs(deltas[moved]) < 0.02:
            print("    ⚠️ 几乎没动! 可能 joint_cmd_mode 不对 / 该关节已在限位 / 速度太低。")
        ans = input(f"    你【肉眼】看是哪根手指哪节在动?(可直接填 IsaacLab 关节名或描述, 回车跳过): ")
        if ans.strip():
            mapping[xid] = ans.strip()
        send_xhand_order(base)   # 归零(回基准)
        time.sleep(0.8)
finally:
    hw.disconnect()

print("\n" + "=" * 64)
print("你记录的 Xhand id -> 手指/关节:")
for xid, name in mapping.items():
    print(f"  xhand id {xid} -> {name}")
print("据此 + 上面 IsaacLab 顺序, 填 joint_map.ISAACLAB2XHAND_IDX (长度21的0..20排列)。")
