"""验证每指【指尖 sensor id】+ 标定统一 force_scale（像 sharpa 单一 scale）。
原理: force_scale 把 nf 换成【真实牛顿】, 仿真接触力也是牛顿 -> obs 自动对齐训练。要个【已知真实力】做参照。
★没秤就用【已知重量的水】当砝码: 水=1g/ml, 装 510ml 水的瓶≈510g≈5N(加瓶皮约530g, 想精确就装500ml再算)。
  方法: 指尖朝上, 把这瓶水【竖放在指尖上】让它的重量压传感器(你扶着别让它倒, 但别下压加力), 回车采样。
  force_scale = target(N) / nf。按多大力都行(nf 与力成正比), 关键是【这个力要已知】(靠水的重量)。
⚠️ 只读不发动作(但 EtherCAT 连接时手可能微动)。容器内跑(privileged+host网络, -it 交互):
  python scripts/calib_fingertip.py [--target 5]   # target=水的重量(N)=克/102
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
from rl_isaaclab.deploy_xhand.tactile_map import FINGER_SENSOR_LAYOUT

FING = ["thumb", "index", "mid", "ring", "pinky"]
EXPECT = [3, 7, 11, 15, 19]

p = argparse.ArgumentParser()
p.add_argument("--target", type=float, default=5.0, help="标定参照力(N): 把指尖压到这么大力。5N≈510g, 落仿真活跃压区间")
p.add_argument("--finger", default="all", choices=["all", "thumb", "index", "mid", "ring", "pinky"],
               help="只标某一根(避免多指受力不均, 单指好压稳); 推荐 mid 单独标")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
hw = XhandHWInterface(cfg)
hw.connect()
print("[calib] 采触觉基线(手张开别碰东西), 等1s让零漂稳定再采...")
baseline = hw.zero_tactile()   # settle=1.0: 等零漂稳定后采; 返回每指(指尖)基线
print(f"[calib] 零漂基线 nf = {[round(float(x)) for x in baseline]}  (下面测得力会减掉它, 与 deploy 一致)")


def read_finger(sids):
    out = {}
    for sid in sids:
        _c, _m, s = hw.hand.get_sensor_state(sid, True)
        out[sid] = float(s.first_ydds_nf) + float(s.second_ydds_nf)
    return out


tips, firm, done_idx = [], [], []
try:
    for i, f in enumerate(FING):
        if args.finger != "all" and f != args.finger:
            continue
        input(f"\n>>> 【{f} 指尖朝上】, 把装 {args.target*102:.0f}g 水的瓶竖放指尖上(用其重量压, 扶稳别下压), 回车采样(约0.3s)...")
        agg = {sid: 0.0 for sid in FINGER_SENSOR_LAYOUT[i]}
        for _ in range(10):
            r = read_finger(FINGER_SENSOR_LAYOUT[i])
            for sid in agg:
                agg[sid] = max(agg[sid], r[sid])
            time.sleep(0.03)
        tip = max(agg, key=agg.get)
        tips.append(tip); firm.append(agg[tip]); done_idx.append(i)
        ok = "✓" if tip == EXPECT[i] else f"✗(预期{EXPECT[i]})"
        print(f"   {f}: 各 sensor 峰值={ {k: round(v) for k, v in agg.items()} }  -> 指尖 id={tip} {ok}  力={agg[tip]:.0f}")
finally:
    hw.disconnect()

firm = np.array(firm)
done_names = [FING[i] for i in done_idx]
firm_sub = np.maximum(0.0, firm - baseline[done_idx])   # 减零漂基线(只取标过的指), 和 deploy 一致
nz = firm_sub[firm_sub > 0]
med = float(np.median(nz)) if len(nz) else 0.0
fs = args.target / med if med > 0 else 0.0       # scale = 参照力 / (减基线后的 nf 中位数)
print("\n" + "=" * 56)
print("识别的指尖 id:", dict(zip(done_names, tips)), " (预期 3/7/11/15/19)")
print(f"压到 {args.target:.1f}N 时 nf(原始):", dict(zip(done_names, [round(float(x)) for x in firm])))
print(f"减零漂基线后 nf:", dict(zip(done_names, [round(float(x)) for x in firm_sub])))
print(f"减基线后 nf 中位数 = {med:.0f}")
print(f"-> force_scale = {args.target:.1f}/{med:.0f} = {fs:.5f}   (统一一个, 像 sharpa)")
print("=" * 56)
print("启用步骤:")
print(f"  1. deploy_cfg.py 设 tactile_fingertip_only = True")
print(f"  2. deploy_cfg.py 设 force_scale = {fs:.5f}")
print("  3. 若上面指尖 id 有 ✗, 把实际 id 填进 tactile_map.FINGERTIP_SENSOR_LAYOUT")
