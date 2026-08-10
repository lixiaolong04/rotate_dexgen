"""实时显示 5 指【指尖】压力: 减零漂后的 nf 原始值 + 换算成牛顿(×force_scale)。
压一压指尖就能看到对应数跳, 用于: 标定核对 / 看抓握时哪几指在受力 / 调 force_scale。
容器内跑(privileged+host网络, -it): python scripts/tac_monitor.py [--hz 10] [--raw]
  --raw: 额外显示不减基线的原始 nf。  Ctrl-C 退出。
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface

FING = ["拇", "食", "中", "无名", "小"]

p = argparse.ArgumentParser()
p.add_argument("--hz", type=float, default=10.0, help="刷新频率")
p.add_argument("--raw", action="store_true", help="也显示不减基线的原始 nf")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
fs = cfg.force_scale if np.isscalar(cfg.force_scale) else float(np.mean(cfg.force_scale))

hw = XhandHWInterface(cfg)
hw.connect()
print("[mon] 采零漂基线(手张开别碰东西), 等1s...")
baseline = hw.zero_tactile()
print(f"[mon] 零漂基线 nf = {[round(float(x)) for x in baseline]}")
print(f"[mon] force_scale={fs:.5f} (指尖-only={getattr(cfg,'tactile_fingertip_only',False)}); 实时显示中, 压指尖看跳。Ctrl-C 退出。\n")

dt = 1.0 / max(1.0, args.hz)
peak = np.zeros(5, np.float32)
try:
    while True:
        nf = np.asarray(hw.get_tactile_raw(), np.float32)   # 已减基线、负截0 (5,)
        peak = np.maximum(peak, nf)
        cells = []
        for i, f in enumerate(FING):
            bar = "█" * min(20, int(nf[i] * fs / 0.3))       # 每 0.3N 一格, 最多20格
            cells.append(f"{f}{nf[i]*fs:4.1f}N({nf[i]:5.0f}) {bar:<20}")
        line = " | ".join(cells)
        if args.raw:
            rawnf = hw._finger_nf_raw()
            line += "  原始nf=" + str([round(float(x)) for x in rawnf])
        print("\r" + line + " ", end="", flush=True)
        time.sleep(dt)
except KeyboardInterrupt:
    print(f"\n[mon] 退出。本次峰值 nf={[round(float(x)) for x in peak]} -> N={[round(float(x*fs),1) for x in peak]}")
finally:
    hw.disconnect()
