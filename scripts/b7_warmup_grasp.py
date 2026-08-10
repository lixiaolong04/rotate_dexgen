"""B7: 暖启动抓取(独立安全版) —— 验证两段式闭合 + 接触即停 + 不卡拇指。
流程: 张开手 -> 你把物体放掌心(手指先别碰) -> 采触觉基线 -> 两段式回放(先张侧摆后合弯曲),
      指尖接触>阈值即停(抓住) -> 保持 5s -> 回车松开。

⚠️⚠️ 手会闭合抓握! 跑前: 物体放好、手固定/拿稳、电源急停在手边。慢速默认 0.1s/步(整条~6s, 可随时断电)。
用法(容器内, -it): python scripts/b7_warmup_grasp.py [--step-dt 0.1] [--thresh 1.0]
  --thresh 抓住判定(get_tactile_info 量级, cfg.grasp_contact_thresh; force_scale 后, 1.0≈原始nf400)
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface

FINGER = ["thumb", "index", "mid", "ring", "pinky"]

p = argparse.ArgumentParser()
p.add_argument("--step-dt", type=float, default=0.1, help="每步间隔(s), 0.1=慢(60帧~6s)更稳; 想快可调小")
p.add_argument("--thresh", type=float, default=None, help="抓住判定阈值(默认用 cfg.grasp_contact_thresh)")
p.add_argument("--min-fingers", type=int, default=None, help="要≥几指接触才算抓住(默认 cfg 2; 防拇指单点一碰就停)")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
thresh = args.thresh if args.thresh is not None else cfg.grasp_contact_thresh
min_fingers = args.min_fingers if args.min_fingers is not None else cfg.grasp_min_fingers
traj = np.load(cfg.init_traj_path)            # (60,21) 两段式: 前 20 张, 后 40 合
close_start = len(traj) // 3                   # 只在"合拢段"判定抓住(张开段不算)
hw = XhandHWInterface(cfg)
hw.connect()


def scaled_contact():
    """与 deploy get_tactile_info 一致: 减基线后 ×force_scale, <阈值清零。返回 (5,)。"""
    f = hw.get_tactile_raw() * cfg.force_scale
    f[f < cfg.contact_threshold] = 0.0
    return f


try:
    # 1. 张开手到轨迹起点(全0)
    print("[B7] 张开手(命令轨迹起点)...")
    hw.set_joint_targets(traj[0], cfg.kp, cfg.kd)
    time.sleep(2.0)

    # 2. 放物体 + 采基线(手指此刻应【空着】不碰物体)
    input("\n>>> 把水瓶放到掌心(手指先别碰到它), 确认安全后回车 -> 采触觉基线 + 开始抓取...")
    print(f"[B7] 采触觉基线: {hw.zero_tactile()}")

    # 3. 两段式回放, 合拢段【≥min_fingers 指接触】才停(防拇指单点一碰就停=夹不牢)
    print(f"[B7] 开始两段式闭合(前{close_start}帧张/后合), step-dt={args.step_dt}s, 阈值={thresh}, 要≥{min_fingers}指")
    grasped_j = None
    for j in range(len(traj)):
        hw.set_joint_targets(traj[j], cfg.kp, cfg.kd)
        time.sleep(args.step_dt)
        c = scaled_contact()
        n_contact = int((c >= thresh).sum())
        phase = "张侧摆" if j < close_start else "合弯曲"
        print(f"  step{j:>2}[{phase}] 接触={[round(float(x),2) for x in c]}  接触指数={n_contact}")
        if j >= close_start and n_contact >= min_fingers:
            grasped_j = j
            print(f"  ✅ 抓住! 第 {j} 帧, {n_contact} 指接触(≥{min_fingers}); 指={[FINGER[i] for i in range(5) if c[i] >= thresh]}")
            break

    # 4. 确定保持姿态
    if grasped_j is not None:
        hold = traj[min(grasped_j + 2, len(traj) - 1)]
    else:
        hold = traj[-1]
        print("  ⚠️ 走完整条轨迹仍未达接触阈值(没碰到/物体太软/阈值偏高), 停在最终 grasp 姿态。")

    # 5. 保持抓握 5s
    print("[B7] 保持抓握 5s(持续下发 grasp 姿态)...")
    for _ in range(50):
        hw.set_joint_targets(hold, cfg.kp, cfg.kd)
        time.sleep(0.1)
    c = scaled_contact()
    print(f"[B7] 保持中接触={[round(float(x),2) for x in c]}")

    # 6. 松开
    input("\n>>> 回车松开(张开手)...")
    print("[B7] 张开手...")
    hw.set_joint_targets(traj[0], cfg.kp, cfg.kd)
    time.sleep(2.0)
finally:
    hw.disconnect()
print("[B7] 完成。")
