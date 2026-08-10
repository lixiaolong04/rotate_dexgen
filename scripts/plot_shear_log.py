# scripts/plot_shear_log.py
"""画 deploy --log 的触觉时间序列: 法向力 nf vs 剪切力 tf, 验证"物体爬升时剪切升、法向不动"。

deploy --log 已记每指【指尖】 shear_nf / shear_tf / shear_tfDir (deploy_xhand.py:134-136),
指序 = [thumb, index, mid, ring, pinky]; nf/tf=first+second pad 之和(原始单位), tfDir 65535=无接触。
analyze_deploy_log.py 只看法向 contact 槽、没碰剪切 —— 本脚本补上剪切这一半。

用法:
  python scripts/plot_shear_log.py s9402_real.npz                  # 全程
  python scripts/plot_shear_log.py s9402_real.npz --t0 30 --t1 50  # 放大 30~50s
  python scripts/plot_shear_log.py s9402_real.npz --marks 12,34    # 在你观察到物体爬升的秒数画竖线
出: <npz>_shear.png + 控制台每指 nf/tf 统计 + 两个判读。
(只需 numpy+matplotlib, 不需硬件/Isaac; 可把 npz scp 到任意机器画。缺 matplotlib: pip install matplotlib)
"""
import argparse
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--rate", type=float, default=20.0, help="控制频率 Hz (deploy 20Hz)")
p.add_argument("--t0", type=float, default=None, help="放大起始秒")
p.add_argument("--t1", type=float, default=None, help="放大结束秒")
p.add_argument("--marks", default=None, help="逗号分隔的秒数, 在图上画竖线(标你看到物体爬升的时刻)")
args = p.parse_args()

d = np.load(args.npz)
FING = ["thumb", "index", "mid", "ring", "pinky"]
need = ["shear_nf", "shear_tf", "shear_tfdir"]
miss = [k for k in need if k not in d.files or len(d[k]) == 0]
if miss:
    raise SystemExit(
        f"❌ 这个 log 没有剪切字段 {miss}。\n"
        f"   说明是旧版 deploy 跑的。用当前 deploy_xhand.py(含[剪切验证], :134-136)重新 --log 跑一遍即可。")

nf = np.asarray(d["shear_nf"], float)       # (T,5) 原始法向力 (未扣基线)
tf = np.asarray(d["shear_tf"], float)       # (T,5) 原始剪切力
tfd = np.asarray(d["shear_tfdir"], float)   # (T,5) 原始剪切方向, 65535=无接触
T = len(nf)
t = np.arange(T) / args.rate

# ---- 控制台统计 ----
contact = tfd != 65535.0
print(f"\n=== {os.path.basename(args.npz)} | {T} 步 / {t[-1]:.1f}s @ {args.rate:.0f}Hz ===\n")
print(f"{'finger':7s} {'接触%':>6s} {'nf均':>9s} {'nf峰':>9s} {'tf均':>9s} {'tf峰':>9s}")
for i, f in enumerate(FING):
    print(f"{f:7s} {contact[:, i].mean() * 100:5.0f}% "
          f"{nf[:, i].mean():9.0f} {nf[:, i].max():9.0f} {tf[:, i].mean():9.0f} {tf[:, i].max():9.0f}")

nf_lo = nf.mean(0).max() * 0.08             # 相对最大指的 8% 算"几乎没法向力"
dead = [FING[i] for i in range(5) if contact[:, i].mean() < 0.1 or nf[:, i].mean() < nf_lo]
print(f"\n判读①(Q2 sim2real 缺口): 接触%≈0 或 nf≈0 的指 = 真机无触觉信号 → {dead if dead else '无'}")
print("  (sim 里这些指照样有合力进 obs → 这就是 sim 有、real 没有的 OOD 通道)")
print("判读②(治盲依据): 看下图 —— 物体爬升时若某指 nf 平 / tf 升, 说明剪切携带了爬升信号(=方案③值得上)。")
print("  用 --marks <看到爬升的秒数> 把那一刻标到图上, 对照该指 tf 是否抬头。\n")

# ---- 画图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m = np.ones(T, bool)
if args.t0 is not None:
    m &= t >= args.t0
if args.t1 is not None:
    m &= t <= args.t1
if m.sum() == 0:
    m = np.ones(T, bool)
marks = [float(x) for x in args.marks.split(",")] if args.marks else []

fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, f in enumerate(FING):
    ax[0].plot(t[m], nf[m, i], color=colors[i], lw=1.2, label=f)
    ax[1].plot(t[m], tf[m, i], color=colors[i], lw=1.2, label=f)
ax[0].set_title("Normal force  nf   (policy SEES this -- real obs uses ONLY this)")
ax[1].set_title("Shear force  tf   (policy is BLIND to this -- creep signal should live here)")
ax[0].set_ylabel("nf (raw)")
ax[1].set_ylabel("tf (raw)")
ax[0].legend(ncol=5, fontsize=8, loc="upper right")
ax[1].legend(ncol=5, fontsize=8, loc="upper right")

# 拇指 nf vs tf 直接对照(双轴): "物体被拇指搓往上爬" 的最直接检验
ax2 = ax[2]
l1 = ax2.plot(t[m], nf[m, 0], color="tab:blue", lw=1.4, label="thumb nf (normal)")
ax2.set_ylabel("thumb nf (raw)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax2b = ax2.twinx()
l2 = ax2b.plot(t[m], tf[m, 0], color="tab:red", lw=1.4, label="thumb tf (shear)")
ax2b.set_ylabel("thumb tf (raw)", color="tab:red")
ax2b.tick_params(axis="y", labelcolor="tab:red")
ax2.set_title("Thumb  nf vs tf   (creep  =>  nf flat / tf rises)")
ax2.set_xlabel("time (s)")
ax2.legend(l1 + l2, [x.get_label() for x in l1 + l2], loc="upper left", fontsize=8)

for a in list(ax) + [ax2b]:
    for mk in marks:
        a.axvline(mk, color="k", ls="--", lw=0.8, alpha=0.6)
for a in ax:
    a.grid(alpha=0.3)

out = os.path.splitext(args.npz)[0] + "_shear.png"
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"图已存: {out}\n")
