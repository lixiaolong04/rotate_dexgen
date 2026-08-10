# scripts/dump_isaaclab_joint_meta.py
"""在 laptop env_isaaclab (py3.11 + Isaac) 下运行, 导出 articulation 关节顺序与限位。"""
import argparse, json, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch, gymnasium as gym
import rl_isaaclab.tasks.inhand_rotate  # 注册任务
from rl_isaaclab.tasks.inhand_rotate.xhand_wave_env_cfg import XhandWaveEnvCfg

cfg = XhandWaveEnvCfg()
cfg.scene.num_envs = 8  # 必须能被 scale_range[2]=8 整除
env = gym.make("Isaac-Inhand-Rotate-Xhand-Wave-v0", cfg=cfg, render_mode=None)
hand = env.unwrapped.hand
limits = hand.root_physx_view.get_dof_limits()[0].cpu().numpy()  # (21,2), 未乘 scale
meta = {
    "joint_names": list(hand.joint_names),
    "dof_lower": limits[:, 0].tolist(),
    "dof_upper": limits[:, 1].tolist(),
    "dof_limits_scale": cfg.dof_limits_scale,
    "actuated_joint_names": list(cfg.actuated_joint_names),
}
os.makedirs("cache", exist_ok=True)
with open("cache/xhand_isaaclab_joint_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("joint_names:", meta["joint_names"])
print("wrote cache/xhand_isaaclab_joint_meta.json")
env.close(); app.close()
