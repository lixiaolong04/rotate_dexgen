"""分析A: 仿真接触力目标(给真机 KP 调参用)。
仿真 sensed_contacts = 指尖净接触力的模长(单位~牛顿, 平滑后), 从 t603 running_mean_std 提取其分布。
真机 analyze_deploy_log 的接触值经 force_scale=0.0025 标定后与此【同单位】, 可直接对照:
  KP 调参目标 = 让真机每指接触力落到下面的"均值~活跃压"区间(而非现在的 16)。
用法: python scripts/sim_contact_target.py [--ckpt ckpts/scale_1000M/student_t603.pth]
"""
import argparse
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpts/scale_1000M/student_t603.pth")
args = p.parse_args()

rms = torch.load(args.ckpt, map_location="cpu")["running_mean_std"]
mean = rms["running_mean"].numpy()
std = np.sqrt(rms["running_var"].numpy())
FING = ["thumb", "index", "mid", "ring", "pinky"]

print("=" * 60)
print("仿真接触力(单位~牛顿) —— 真机 KP 调参目标")
print("=" * 60)
print(f"{'指':>6} | {'均值(常态)':>10} | {'std':>6} | {'活跃压~均值+std':>14}")
print("-" * 60)
tot_m = 0.0
rows = []
for i, f in enumerate(FING):
    # contact 在每帧的 [42:47]; 三帧平均
    ms = np.mean([mean[fr * 62 + 42 + i] for fr in range(3)])
    ss = np.mean([std[fr * 62 + 42 + i] for fr in range(3)])
    rows.append((f, ms, ss))
    tot_m += ms
    print(f"{f:>6} | {ms:>10.2f} | {ss:>6.2f} | {ms + ss:>14.2f}")
print("-" * 60)
print(f"五指合计均值={tot_m:.1f}  全指平均={tot_m/5:.2f}N  拇指占比={rows[0][1]/tot_m*100:.0f}%")
print()
print("【KP 调参判据】真机跑 deploy --log 后看 analyze 的'各指均值力':")
print(f"  目标: 每指落到 ~1.5-4N(均值)、活跃压峰值 ~5-7N; 拇指占比 ~{rows[0][1]/tot_m*100:.0f}%; 力失衡<3")
print("  现状: 单指顶到 16(超 4 倍) = KP 太硬. JSON 调低 KP 直到接触力进入上面区间。")
print("  注: 接触力会随旋转起伏(松开时~0, 压住时~峰值), 看均值别看瞬时峰。")
