"""E2: 触觉特性 —— 隔离 触觉延迟/迟滞/噪声/线性度 gap。
单指【点按 + 慢斜坡压硬面 + 松】, 同步记 关节力矩(=真实接触力的地面真值) 和 触觉 nf, 算:
  - 延迟: 每次点按, 力矩起跳 ↔ nf 起跳 的时间差(ms)。
  - 迟滞: 斜坡压下 vs 松开, 同力矩下 nf 差(%)。
  - 噪声: 未接触时 nf 抖动。 线性度: nf 与力矩相关性。 刷新率: nf 实际多久变一次。
仿真触觉是【瞬时、无迟滞、无噪声、严格随力】, 这里量出来的就是触觉 sim2real 缺口。

⚠️ 会让一根手指【反复压】! 跑前: 指尖正下方放硬平面(书/木块), 调高度让"压"时指尖压到、"抬"时离开; 急停在手边。用户自己跑。
容器内(-it): python scripts/e2_tactile_char.py [--finger mid] [--log e2.npz]
急停: Ctrl-C 或另开终端 touch STOP。
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface

# finger -> (触觉索引, [mcp1, pip, dip] 屈曲关节 IsaacLab序)
FMAP = {"thumb": (0, [9, 14, 19]), "index": (1, [5, 10, 15]), "mid": (2, [7, 12, 17]),
        "ring": (3, [8, 13, 18]), "pinky": (4, [11, 16, 20])}
LIFT, PRESS, RMAX = 0.3, 0.9, 1.0   # 抬起/点按/斜坡最深 屈曲rad

p = argparse.ArgumentParser()
p.add_argument("--finger", default="mid", choices=list(FMAP))
p.add_argument("--hz", type=float, default=50.0)
p.add_argument("--kp", type=float, default=12.0, help="点按要顶得动+顶得住, kp 比抓握稍大")
p.add_argument("--kd", type=float, default=0.4)
p.add_argument("--taps", type=int, default=8)
p.add_argument("--log", default="e2.npz")
args = p.parse_args()

tac_idx, jts = FMAP[args.finger]
dt = 1.0 / args.hz

# 时间驱动激励
segs = [("noise", 2.5, lambda tl: LIFT)]
for k in range(args.taps):                       # 点按: 0.3s 压 + 0.3s 抬
    segs.append((f"tap{k}_press", 0.3, lambda tl: PRESS))
    segs.append((f"tap{k}_lift", 0.3, lambda tl: LIFT))
segs.append(("ramp_up", 4.0, lambda tl: LIFT + (RMAX - LIFT) * tl / 4.0))
segs.append(("ramp_dn", 4.0, lambda tl: RMAX - (RMAX - LIFT) * tl / 4.0))
segs.append(("end", 1.0, lambda tl: LIFT))
bnd = []
acc = 0.0
for label, dur, fn in segs:
    bnd.append((label, acc, acc + dur, fn)); acc += dur
TOTAL = acc


def target_at(t):
    for label, a, b, fn in bnd:
        if a <= t < b:
            return float(fn(t - a)), label
    return LIFT, "end"


cfg = DeployCfg()
cfg.keyboard_listen = False
cfg.joint_cmd_mode = 4; cfg.cmd_speed = 0.0; cfg.cmd_torque = 0.0
hw = XhandHWInterface(cfg)
hw.connect()
print(f"[E2] 指={args.finger}(触觉idx{tac_idx}, 屈曲关节{jts}) mode4 kp={args.kp}; 总~{TOTAL:.0f}s")
print("[E2] 采触觉零漂基线(指尖此刻【别碰】硬面, 抬起状态), 等1s...")
base = hw.zero_tactile()
print(f"[E2] 基线nf={[round(float(x)) for x in base]}")
input(">>> ⚠️ 硬平面放在指尖正下方(压到/抬开), 急停在手边, 回车开始点按+斜坡(Ctrl-C/touch STOP 急停)...")
if os.path.exists("STOP"):
    os.remove("STOP")

logt, lcmd, lact, ltq, lnf, lph = [], [], [], [], [], []
t0 = time.time()
try:
    while True:
        t = time.time() - t0
        if t >= TOTAL or os.path.exists("STOP"):
            break
        val, label = target_at(t)
        vec = np.zeros(21, np.float32)
        for j in jts:
            vec[j] = val
        hw.set_joint_targets(vec, args.kp, args.kd)
        pos = hw.get_joint_pos()
        tq, _, _ = hw.read_joint_diag()
        nf = hw.get_tactile_raw()[tac_idx]
        logt.append(time.time() - t0); lcmd.append(val); lact.append(pos[jts].copy())
        ltq.append(tq[jts].copy()); lnf.append(float(nf)); lph.append(label)
        sl = t0 + len(logt) * dt - time.time()
        if sl > 0:
            time.sleep(sl)
except KeyboardInterrupt:
    print("\n[E2] 急停")
finally:
    for _ in range(15):
        hw.set_joint_targets(np.zeros(21, np.float32), args.kp, args.kd)
        time.sleep(0.03)
    hw.disconnect()

logt = np.array(logt); lcmd = np.array(lcmd); lact = np.array(lact, np.float32)
ltq = np.array(ltq, np.float32); lnf = np.array(lnf, np.float32); lph = np.array(lph)
teff = np.abs(ltq).sum(axis=1)   # 三屈曲关节力矩绝对值和 = 接触effort地面真值
np.savez(args.log, t=logt, cmd=lcmd, actual=lact, torque=ltq, nf=lnf, teff=teff, phase=lph, finger=args.finger)
print(f"[E2] 存 {args.log} ({len(logt)}步, 实测均率 {1.0/np.mean(np.diff(logt)):.0f}Hz)")


def onset(sig, base_mean, base_std, t_arr, k=5.0):
    """首次超过 base+k*std 的时刻; 无则 nan。"""
    thr = base_mean + k * max(base_std, 1e-6)
    w = np.where(sig > thr)[0]
    return t_arr[w[0]] if len(w) else np.nan


# 噪声底(noise 段)
nm = np.where(lph == "noise")[0]
nf_noise_mean, nf_noise_std = (lnf[nm].mean(), lnf[nm].std()) if len(nm) else (0, 0)
tq_noise_mean, tq_noise_std = (teff[nm].mean(), teff[nm].std()) if len(nm) else (0, 0)

# 延迟: 每个 tap 的 press 段。接触时刻=【实际角卡住】(撞到面停住), 不能用力矩(起步加速也会跳=误判).
lats = []
for k in range(args.taps):
    m = np.where(lph == f"tap{k}_press")[0]
    if len(m) < 4:
        continue
    tt = logt[m]; ang = lact[m, 0]      # mcp1 实际角
    amax = ang.max()
    ci = np.where(ang >= amax - 0.02)[0]   # 接触=首次到达本次角度平台
    t_contact = tt[ci[0]] if len(ci) else np.nan
    t_nf = onset(lnf[m], nf_noise_mean, nf_noise_std, tt)
    if not (np.isnan(t_contact) or np.isnan(t_nf)):
        lats.append((t_nf - t_contact) * 1000)

# 迟滞 + 线性度(ramp)
ru = np.where(lph == "ramp_up")[0]; rd = np.where(lph == "ramp_dn")[0]
hyst = lin = np.nan
if len(ru) > 5 and len(rd) > 5:
    tqmid = 0.5 * (teff[ru].max() + teff[ru].min())
    nf_up = np.interp(tqmid, teff[ru], lnf[ru])
    nf_dn = np.interp(tqmid, teff[rd][::-1], lnf[rd][::-1])
    denom = max(abs(nf_up), abs(nf_dn), 1e-6)
    hyst = abs(nf_up - nf_dn) / denom * 100
    allm = np.concatenate([ru, rd])
    if teff[allm].std() > 1e-6 and lnf[allm].std() > 1e-6:
        lin = float(np.corrcoef(teff[allm], lnf[allm])[0, 1])

# 刷新率: 只在【力持续上升(ramp_up 且 nf>0)】时数 nf 变化才有意义(持稳压时力不变 nf 本就不变, 不算慢)。
ruc = ru[lnf[ru] > 0] if len(ru) else np.array([], int)
chg = np.mean(np.abs(np.diff(lnf[ruc])) > 1e-3) if len(ruc) > 2 else np.nan

print("\n=== 触觉 sim2real 特性 ===")
print(f"  延迟(接触[角度卡住]→nf起跳): 中位 {np.median(lats):.0f}ms  各次={[round(x) for x in lats]}  (含部分机械就位; nf一旦起跳很快)" if lats else "  延迟: 未测到有效点按(检查硬面高度/kp)")
print(f"  迟滞: {hyst:.0f}%   (压下vs松开同力矩下nf差; 仿真=0)")
print(f"  线性度: nf~力矩 相关 r={lin:.2f}   (仿真接近1; nan=斜坡段没压到面, 重摆硬面位置再测)")
print(f"  噪声底: nf std={nf_noise_std:.0f} (均值{nf_noise_mean:.0f}); 真机噪声门 contact_threshold 应盖住它")
print(f"  nf刷新(力上升段): {chg*100:.0f}% 步在变" if not np.isnan(chg) else "  nf刷新: 斜坡段没接触, 无法测(重摆硬面)")
print("\n判读: 延迟≈100ms(≈2控制步)含机械就位, 不是大缺口; 真要测纯传感器延迟需直接敲指尖。")
