"""option C 前pad几何过滤 —— 标定 & 验证。
hold 住抓取(zero action), 打印每指【接触点在 dip_link 局部帧的平均坐标(x,y,z)】+ 净力,
用来验证 pad 法向轴(拇指+Y/其余+Z)与阈值(默认0.009): pad法向那一列应明显>0(≈pad偏移0.018),
阈值取其半值(~0.009)能留前pad、切掉侧/背。直接读原始传感器(不受 env 内部 filter 影响)。

Usage: python scripts/check_pad_contact.py --num_envs 20 --steps 30 --headless
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
p.add_argument("--num_envs", type=int, default=20)
p.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import rl_isaaclab.tasks.inhand_rotate  # noqa: F401,E402  registers tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rl_isaaclab.tasks.inhand_rotate.xhand_wave_env import transform_between_frames  # noqa: E402

fingers = ["thumb", "index", "mid", "ring", "pinky"]
env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg)
u = env.unwrapped

obs, _ = env.reset()
zero = torch.zeros(args.num_envs, env.action_space.shape[-1], device=u.device)

warm = 3
acc_p = torch.zeros(5, 3, device=u.device)   # 每指接触点局部帧坐标累加
acc_f = torch.zeros(5, device=u.device)      # 每指净力
seen = torch.zeros(5, device=u.device)       # 每指有接触的样本数(用于求均值)
frac_front = torch.zeros(5, device=u.device)  # 用当前阈值判为前pad的比例
acc_proj = torch.zeros(5, device=u.device)    # 接触点沿【精确3D法向】的投影
cnt = 0
normals = torch.tensor(u.cfg.frontpad_normal, dtype=torch.float, device=u.device)  # (5,3) 单位法向
th = torch.tensor(u.cfg.frontpad_thresh, device=u.device)
for step in range(args.steps):
    env.step(zero)
    if step >= warm:
        tf = u.hand.data.body_link_state_w[:, u.elastomer_ids, :7]
        tpos, tquat = tf[..., :3], tf[..., 3:7]
        wq = torch.zeros_like(tquat); wq[..., 0] = 1.0
        for i, sensor in enumerate(u._contact_sensor):
            f = torch.norm(sensor.data.net_forces_w_history[:, 0, 0, :], dim=-1)  # (N,)
            _, pts, _, _, c, s = sensor.contact_physx_view.get_contact_data(dt=u.cfg.sim.dt)
            mean_w, nct = u._unpack_contact_mean(pts, c, s, u.num_envs, u.device)
            p_local = transform_between_frames(mean_w - tpos[:, i, :], wq[:, i, :], tquat[:, i, :])
            has = nct > 0
            if has.any():
                acc_p[i] += p_local[has].sum(0)
                acc_f[i] += f[has].sum()
                seen[i] += has.sum()
                proj = (p_local * normals[i]).sum(-1)   # 接触点·精确法向
                acc_proj[i] += proj[has].sum()
                frac_front[i] += (has & (proj > th[i])).sum()
        cnt += 1

seen_safe = seen.clamp(min=1)
acc_p /= seen_safe.unsqueeze(-1)
acc_f /= seen_safe
acc_proj /= seen_safe
frac_front /= seen_safe
print(f"\n=== 每指 接触点(dip_link局部帧) + 沿【精确3D法向】投影 proj·n + 净力 (over ~{cnt} steps) ===")
print(f"{'finger':7} {'x':>8} {'y':>8} {'z':>8}  {'proj·n':>7}  {'force':>6}  thresh  %判前pad")
for i, fg in enumerate(fingers):
    print(f"{fg:7} {acc_p[i,0]:8.4f} {acc_p[i,1]:8.4f} {acc_p[i,2]:8.4f}  {acc_proj[i]:7.4f}  {acc_f[i]:6.2f}  {float(th[i]):.3f}   {frac_front[i]*100:5.1f}%")
print("\n判读: pad法向轴(拇指=y轴1/其余=z轴2)那一列应明显>0(≈pad偏移~0.018)。")
print("      阈值应落在'侧面接触值'与'前pad值'之间; %判前pad 太低=阈值太严, 太高=没过滤到侧面。")
import os, sys  # noqa: E402
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)   # 硬退出绕开 headless app.close() 偶发 hang
