"""云端快测(不改 IsaacLab): 在已建好的 contact view 上直接调 get_friction_data,
看 v2.2 + IsaacSim4.5 的 physx 是否真返回 friction 数据(buffer 配置对不对)。
测通 -> 移植那 ~80 行 friction 读取进云端 contact_sensor 就能训 shear; 测不通 -> 要走更重的升级路。

用法(云端): python scripts/test_cloud_friction.py --headless --num_envs 20 [--cache <grasp.npy>]
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=20)   # scale 桶数倍数
parser.add_argument("--task", default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
parser.add_argument("--cache", default=None)
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
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    u = env.unwrapped
    env.reset()
    z = env.zero_actions()
    for _ in range(30):   # 让暖抓握住物体 -> 指尖有接触
        env.step(z)

    cs = u._contact_sensor[0]
    view = getattr(cs, "contact_physx_view", None) or getattr(cs, "_contact_physx_view", None)
    print("[test] contact sensor:", type(cs).__name__, "| view:", type(view).__name__ if view else None)
    print("[test] filter_count =", getattr(view, "filter_count", "N/A"))
    print("[test] has get_friction_data =", hasattr(view, "get_friction_data"))
    print("[test] has get_contact_data =", hasattr(view, "get_contact_data"))

    import torch
    dt = float(getattr(u, "physics_dt", None) or u.cfg.sim.dt)
    try:
        out = view.get_friction_data(dt=dt)
        print(f"[test] get_friction_data 返回 {len(out)} 项:")
        for i, x in enumerate(out):
            if hasattr(x, "shape"):
                nz = int((x != 0).sum()) if x.numel() else 0
                sa = float(x.abs().sum()) if x.numel() else 0.0
                print(f"   [{i}] tensor shape={tuple(x.shape)} nonzero={nz} sum|·|={sa:.4f}")
            else:
                print(f"   [{i}] {x}")
        fforce = out[0]
        if hasattr(fforce, "numel") and fforce.numel() and float(fforce.abs().sum()) > 1e-6:
            print("✅ 云端 get_friction_data 真返回非零 friction 力 -> 移植可行, 能在云端训 shear")
        else:
            print("⚠️ get_friction_data 可调用但返回空/全零 -> 多半 view 创建时没配 friction buffer(需更深改 view 创建); 或当前没接触")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 云端 get_friction_data 调用失败: {e} -> 走升级路, 不走小补丁")
    env.close()


main()
app.close()
