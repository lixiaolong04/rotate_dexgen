"""Phase1 快验: enable_contact_force_vec=True 时 obs=186 且末15维剪切槽有信号。本地 Isaac, 轻量。
用法: python scripts/check_shear_obs.py --headless --num_envs 20
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=20)   # 须为 scale 桶数(10) 的倍数
parser.add_argument("--task", default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
from isaaclab.envs import DirectRLEnvCfg  # noqa: E402

import rl_isaaclab.tasks.inhand_rotate  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.enable_contact_force_vec = True   # ★ 开剪切 obs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    u = env.unwrapped
    print("[check] cfg.observation_space =", u.cfg.observation_space)
    obs = env.reset()
    z = env.zero_actions()
    for _ in range(20):
        obs, _, _, _ = env.step(z)
    ob = obs["obs"] if isinstance(obs, dict) else obs
    print("[check] obs shape =", tuple(ob.shape))
    F = u.cfg.observation_space // 3
    shear = ob[:, 2 * F + 47:2 * F + 62]      # 末帧剪切 15 维
    nz = (shear.abs().sum(-1) > 1e-6).float().mean() * 100
    print(f"[check] 剪切槽15维: 非零env占比={nz:.0f}%  |均值|={shear.abs().mean():.4f}  范围[{shear.min():.3f},{shear.max():.3f}]")
    assert ob.shape[1] == 186, f"obs 不是 186 而是 {ob.shape[1]}!"
    print("✅ Phase1 sim 剪切 obs 接线正确: obs=186, 剪切槽有信号")
    env.close()


main()
app.close()
