"""E1: 作动器开环特性(无物体) —— 隔离 PD/作动器/延迟 gap。
命令确定激励(阶跃 + 0.5/1/2Hz 正弦), 记 目标/实际/力矩, 算: 延迟ms、上升时间、超调、稳态误差、各频幅比(带宽)。
仿真侧是近理想 PD(秒到位), 真机这里量出来的偏差就是 sim2real 的作动器缺口。

⚠️ 会让手指【空转】! 跑前: 取下物体、手指周围无遮挡、电源急停在手边。用户自己跑(发控制命令)。
容器内(-it): python scripts/e1_actuator_char.py [--joints 5,7,8,11] [--amp 0.5] [--hz 50] [--log e1.npz]
急停: Ctrl-C 或另开终端 touch STOP。
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface

JN = ["i_mcp0", "ltspalm", "m_mcp0", "r_mcp0", "t_mcp0", "i_mcp1", "p_mcp0", "m_mcp1",
      "r_mcp1", "t_mcp1", "i_pip", "p_mcp1", "m_pip", "r_pip", "t_pip", "i_dip",
      "p_pip", "m_dip", "r_dip", "t_dip", "p_dip"]
SINE_FREQS = (0.5, 1.0, 2.0)

p = argparse.ArgumentParser()
p.add_argument("--joints", default="5,7,8,11", help="激励关节IsaacLab序号(逗号); 默认四指mcp1屈曲(旋转gait主用)")
p.add_argument("--amp", type=float, default=0.5, help="阶跃幅度rad")
p.add_argument("--hz", type=float, default=50.0, help="命令+采样目标率")
p.add_argument("--kp", type=float, default=10.0)
p.add_argument("--kd", type=float, default=0.3)
p.add_argument("--log", default="e1.npz")
args = p.parse_args()

joints = [int(x) for x in args.joints.split(",")]
dt = 1.0 / args.hz

# 时间驱动的激励 s(t)(按真实时间生成 -> 频率不受循环抖动影响)
segs = [("base0", 2.0, lambda tl: 0.0),
        ("step_up", 2.0, lambda tl: args.amp),
        ("step_dn", 1.0, lambda tl: 0.0)]
for f in SINE_FREQS:
    segs.append((f"sine{f}", 4.0 / f, (lambda ff: lambda tl: 0.4 + 0.3 * np.sin(2 * np.pi * ff * tl))(f)))
segs.append(("end0", 1.0, lambda tl: 0.0))
bnd = []
acc = 0.0
for label, dur, fn in segs:
    bnd.append((label, acc, acc + dur, fn))
    acc += dur
TOTAL = acc


def target_at(t):
    for label, a, b, fn in bnd:
        if a <= t < b:
            return float(fn(t - a)), label
    return 0.0, "end0"


cfg = DeployCfg()
cfg.keyboard_listen = False
cfg.joint_cmd_mode = 4          # MIT: kp/kd 生效, 软
cfg.cmd_speed = 0.0             # 保持(非0会驱动)
cfg.cmd_torque = 0.0
hw = XhandHWInterface(cfg)
hw.connect()
print(f"[E1] mode4 kp={args.kp} kd={args.kd}; 激励={[JN[j] for j in joints]}; 总时长~{TOTAL:.0f}s @{args.hz:.0f}Hz")
input(">>> ⚠️ 确认【已取下物体】+手指周围无遮挡+急停在手边, 回车开始空转激励(Ctrl-C/touch STOP 急停)...")
if os.path.exists("STOP"):
    os.remove("STOP")

logt, lcmd, lact, ltq, lph = [], [], [], [], []
t0 = time.time()
try:
    while True:
        t = time.time() - t0
        if t >= TOTAL or os.path.exists("STOP"):
            break
        val, label = target_at(t)
        vec = np.zeros(21, np.float32)
        for j in joints:
            vec[j] = val
        hw.set_joint_targets(vec, args.kp, args.kd)
        pos = hw.get_joint_pos()
        tq, _, _ = hw.read_joint_diag()
        logt.append(time.time() - t0)
        lcmd.append(val); lact.append(pos[joints].copy()); ltq.append(tq[joints].copy()); lph.append(label)
        sl = t0 + (len(logt)) * dt - time.time()
        if sl > 0:
            time.sleep(sl)
except KeyboardInterrupt:
    print("\n[E1] 急停")
finally:
    for _ in range(15):       # 缓慢回零
        hw.set_joint_targets(np.zeros(21, np.float32), args.kp, args.kd)
        time.sleep(0.03)
    hw.disconnect()

logt = np.array(logt); lcmd = np.array(lcmd); lact = np.array(lact, np.float32)
ltq = np.array(ltq, np.float32); lph = np.array(lph)
np.savez(args.log, t=logt, cmd=lcmd, actual=lact, torque=ltq, phase=lph, joints=joints, hz=args.hz, amp=args.amp)
print(f"[E1] 存 {args.log} ({len(logt)}步, 实测均率 {1.0/np.mean(np.diff(logt)):.0f}Hz)")


def xcorr_lag(cmd, act, tt):
    """互相关求 act 相对 cmd 的滞后(s); 用真实采样间隔。"""
    a = cmd - cmd.mean(); b = act - act.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return np.nan
    c = np.correlate(b, a, "full")
    k = c.argmax() - (len(a) - 1)
    return k * np.mean(np.diff(tt))


print("\n=== 作动器特性(每激励关节) ===")
for ji, j in enumerate(joints):
    cmd = lcmd; act = lact[:, ji]
    su = np.where(lph == "step_up")[0]
    lat = rise = ov = sse = np.nan
    if len(su) > 5:
        i0 = su[0]; base = act[max(0, i0 - 2)]; final = args.amp
        a = act[i0:su[-1] + 1]; tt = logt[i0:su[-1] + 1] - logt[i0]
        t10 = base + 0.1 * (final - base); t90 = base + 0.9 * (final - base)
        w10 = np.where(a >= t10)[0]; w90 = np.where(a >= t90)[0]
        if len(w10):
            lat = tt[w10[0]]
        if len(w90) and not np.isnan(lat):
            rise = tt[w90[0]] - lat
        ov = (a.max() - final) / (final - base) * 100 if final > base else np.nan
        sse = final - a[-1]
    sl = []
    for f in SINE_FREQS:
        m = np.where(lph == f"sine{f}")[0]
        if len(m) > 10:
            lg = xcorr_lag(cmd[m], act[m], logt[m])
            ar = (act[m].max() - act[m].min()) / max(1e-6, (cmd[m].max() - cmd[m].min()))
            sl.append(f"{f}Hz:延迟{lg * 1000:+.0f}ms 幅比{ar:.2f}")
    print(f" {JN[j]:8s} 阶跃: 延迟{lat * 1000:4.0f}ms 上升{rise * 1000:4.0f}ms 超调{ov:5.0f}% 稳态误差{sse:+.3f}rad | " + "  ".join(sl))
print("\n判读: 延迟>~50ms 或 2Hz幅比<0.7 = 作动器跟不上快步态(旋转就靠这些屈曲关节高频换抓);")
print("      稳态误差大 = 力矩不足/卡; 仿真近理想(延迟~0,幅比~1), 这里的偏差=作动器sim2real缺口。")
