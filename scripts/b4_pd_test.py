"""B4: PD 辨识 —— 调真机 kp/kd/cmd_speed 让位置跟踪"像仿真那样合理"。
仿真是显式力矩 PD(p_gain=10,d_gain=0.3) 且【PD增益域随机化 ×0.5~2】=> 策略对 4 倍 PD 范围鲁棒,
真机只要: ① 保位稳不抖 ② 阶跃在 ~1-2 控制周期到位、不超调太多。

⚠️ 会发指令、手会动(只动一个手指, 其余保持当前姿态)。手放稳、可断电。结果写 b4_result.txt(防 SDK 日志刷屏)。
用法(容器内, -it):
  单次: python scripts/b4_pd_test.py [--kp 100] [--kd 3] [--speed 350] [--joint 5] [--step 0.2]
  扫描: python scripts/b4_pd_test.py --sweep [--joint 5] [--step 0.3]   # 一次测多组 kp/speed, 出对比表
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
from rl_isaaclab.deploy_xhand.joint_map import ISAACLAB2XHAND_IDX

p = argparse.ArgumentParser()
p.add_argument("--kp", type=float, default=100.0)
p.add_argument("--kd", type=float, default=3.0)
p.add_argument("--speed", type=float, default=350.0)
p.add_argument("--joint", type=int, default=5, help="IsaacLab 关节下标(默认5=食指指根弯曲)")
p.add_argument("--step", type=float, default=0.2, help="阶跃幅度(rad)")
p.add_argument("--sweep", action="store_true", help="扫 kp∈{10,50,100,300}+speed∈{700,1500}, 出对比表")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
hw = XhandHWInterface(cfg)
hw.connect()
XID = ISAACLAB2XHAND_IDX[args.joint]
JN = cfg.joint_names[args.joint]


def read_test_joint():
    return float(hw.hand.get_joint_state(XID, True)[2].position)


def run_phase(target, kp, kd, duration, cmd_hz=20.0):
    t0 = time.perf_counter(); traj = []; last = -1.0; dt = 1.0 / cmd_hz
    while time.perf_counter() - t0 < duration:
        now = time.perf_counter() - t0
        if now - last >= dt:
            hw.set_joint_targets(target, kp, kd); last = now
        traj.append((now, read_test_joint())); time.sleep(0.004)
    return traj


def measure(kp, kd, speed, step):
    """从当前姿态做 保位+阶跃, 返回指标 dict。"""
    cfg.cmd_speed = speed
    home = hw.get_joint_pos(); start = float(home[args.joint])
    hold = np.array([v for _, v in run_phase(home, kp, kd, 1.0)])
    target = home.copy(); target[args.joint] = start + step
    tr = run_phase(target, kp, kd, 1.5)
    t = np.array([x for x, _ in tr]); y = np.array([v for _, v in tr])
    goal = start + step
    ss = y[t > 1.0].mean() if (t > 1.0).any() else y[-1]
    peak = y.max() if step > 0 else y.min()
    over = (peak - goal) / step * 100 if step else 0.0
    thr = start + 0.9 * step
    idx = np.where(y >= thr)[0] if step > 0 else np.where(y <= thr)[0]
    rise = (t[idx[0]] * 1000) if len(idx) else float("nan")
    run_phase(home, kp, kd, 1.0)  # 归位
    return dict(kp=kp, kd=kd, speed=speed, jitter=hold.std(), rise=rise, over=over, sserr=goal - ss)


lines = [f"测试关节 isaaclab[{args.joint}]={JN}(xhand {XID})  阶跃={args.step}rad"]
try:
    if args.sweep:
        combos = [(kp, args.kd, 350.0) for kp in (10, 50, 100, 300)] + \
                 [(100.0, args.kd, 700.0), (100.0, args.kd, 1500.0)]
        lines.append(f"{'kp':>5}{'kd':>5}{'speed':>7} | {'抖动std':>9}{'上升ms':>8}{'超调%':>7}{'稳态err':>9}")
        lines.append("-" * 60)
        for kp, kd, sp in combos:
            m = measure(kp, kd, sp, args.step)
            lines.append(f"{kp:>5.0f}{kd:>5.1f}{sp:>7.0f} | {m['jitter']:>9.4f}{m['rise']:>8.0f}{m['over']:>7.0f}{m['sserr']:>+9.3f}")
        lines.append("判读: 若各行上升ms/超调差不多 => kp/kd/speed 对 SDK 内部环无效, 留默认; 若某组明显更快更稳 => 用它")
    else:
        m = measure(args.kp, args.kd, args.speed, args.step)
        lines.append(f"kp={args.kp} kd={args.kd} speed={args.speed}")
        lines.append(f"[保位] 抖动std={m['jitter']:.4f}")
        lines.append(f"[阶跃] 上升时间={m['rise']:.0f}ms  超调={m['over']:+.0f}%  稳态误差={m['sserr']:+.3f}rad")
        lines.append("好的标准: 上升<100ms & 超调<~20% & 稳态误差小")
finally:
    hw.disconnect()

report = "\n".join(lines)
with open("b4_result.txt", "w") as f:
    f.write(report + "\n")
print("\n" + "=" * 64 + "\n========== B4 结果 (也已写入 b4_result.txt) ==========\n" + "=" * 64)
print(report)
print("=" * 64)
