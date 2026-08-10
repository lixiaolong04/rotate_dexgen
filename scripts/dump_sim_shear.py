"""Phase2 验证门(sim 侧): 在本地 IsaacLab 2.3.2 开 track_friction_forces, dump 每指【指尖帧真·切向力(剪切)】。

背景: 云端 v2.2 没有 friction_forces_w(只有法向), 本地 2.3.2 有 -> 先在本地验证"真剪切对不对得上真机 tf",
门过了再决定怎么在云端重训。配 scripts/compare_shear_sim2real.py 出 C1-C5 判据。

受控场景 S-A: 零动作 + grasp 缓存握住 + 对物体施加"沿轴向上(+z)"恒定外力做幅值扫描, 模拟"被搓往上爬"。
关随机化 + force_scale=0(关 env 自带随机扰动力, 否则会覆盖我们施加的上推力)。

用法(本地, 需 Isaac Sim, headless):
  python scripts/dump_sim_shear.py --headless --num_envs 128
出: sim_shear_dump.npz (friction_tip / nf / up_tip / push / cmask), 给 compare_shear_sim2real.py。

⚠️ 首版, 需本地实跑一次校错(friction_forces_w 形状 / 外力是否被 env 覆盖 / quat 工具名)。报错发我即改。
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Xhand-Wave-v0")
parser.add_argument("--cache", type=str, default=None, help="grasp 缓存路径(默认用 cfg 的 pose5)")
parser.add_argument("--out", type=str, default="sim_shear_dump.npz")
parser.add_argument("--settle", type=int, default=40, help="每个力档先跑多少步让接触稳定")
parser.add_argument("--record", type=int, default=20, help="稳定后记多少步")
parser.add_argument("--pushes", type=str, default="0,0.5,1,2,3,4", help="上推力档(×m·g)逗号分隔")
parser.add_argument("--tilt", action="store_true", help="S-B: 改用全SO(3)随机朝向(靠重力造滑移)而非上推力")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402

import rl_isaaclab.tasks.inhand_rotate  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

FING = ["thumb", "index", "mid", "ring", "pinky"]


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    # S-A: 标称朝向先排除 SO(3) 混淆; --tilt 时走 S-B
    env_cfg.reset_random_quat = bool(args_cli.tilt)
    env_cfg.gravity_curriculum = False
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.force_scale = 0.0  # 关 env 自带随机扰动力(否则覆盖我们的上推力)
    # 关全部随机化, 让信号干净
    for attr in ("randomize_pd_gains", "randomize_friction", "randomize_com",
                 "randomize_mass", "randomize_joint_pos_offset"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache
    # ★ 本地 2.3.2 开真切向力(云端 v2.2 没有此字段, 此脚本只在本地跑)
    for cs in env_cfg.contact_sensor:
        cs.track_friction_forces = True
        # max_contact_data_count_per_prim 默认 4 足够; filter_prim_paths_expr 已指向 object

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    u = env.unwrapped
    env.reset()
    z = env.zero_actions()

    pushes = [float(x) for x in args_cli.pushes.split(",")] if not args_cli.tilt else [0.0]
    # 物体质量(N,) 用于把"×m·g"换成牛顿
    obj_mass = u.object.root_physx_view.get_masses().reshape(u.num_envs).to(u.device)
    g = 9.81
    up_world = torch.zeros(u.num_envs, 3, device=u.device)
    up_world[:, 2] = 1.0  # 圆柱轴 = 世界 +z(rot_axis=(0,0,1)); 爬升 = +z

    rec = {k: [] for k in ["friction_tip", "nf", "up_tip", "push", "cmask"]}

    for p in pushes:
        F = (up_world * (p * obj_mass * g)[:, None]).reshape(u.num_envs, 1, 3)
        torq = torch.zeros(u.num_envs, 1, 3, device=u.device)
        for s in range(args_cli.settle + args_cli.record):
            if not args_cli.tilt and p > 0.0:
                u.object.set_external_force_and_torque(forces=F, torques=torq)  # 每步重施(env force_scale=0 不覆盖)
            env.step(z)
            if s < args_cli.settle:
                continue
            # --- 每指: 世界帧真切向力 friction_forces_w (N,1,1,3) -> (N,3) ---
            fr_w = torch.stack([
                torch.nan_to_num(u._contact_sensor[i].data.friction_forces_w[:, 0, 0, :], nan=0.0)
                for i in range(5)], dim=1)  # (N,5,3) world
            # 法向幅值(给 tf/nf 比) = net_forces 最新帧的 norm
            nf = torch.stack([
                torch.norm(u._contact_sensor[i].data.net_forces_w_history[:, 0, 0, :], dim=-1)
                for i in range(5)], dim=1)  # (N,5)
            # 指尖(elastomer)帧 quat (N,5,4); 与 obs 用的同帧(env L434)
            q_tip = u.hand.data.body_link_state_w[:, u.elastomer_ids, 3:7]  # (N,5,4)
            # world -> 指尖帧(力是自由矢量, 只旋不平移)
            fr_tip = quat_apply(quat_inv(q_tip.reshape(-1, 4)), fr_w.reshape(-1, 3)).reshape(u.num_envs, 5, 3)
            up_tip = quat_apply(quat_inv(q_tip.reshape(-1, 4)),
                                up_world[:, None, :].expand(-1, 5, -1).reshape(-1, 3)).reshape(u.num_envs, 5, 3)
            cmask = nf > 1.0e-3
            rec["friction_tip"].append(fr_tip.cpu())
            rec["nf"].append(nf.cpu())
            rec["up_tip"].append(up_tip.cpu())
            rec["push"].append(torch.full((u.num_envs,), p))
            rec["cmask"].append(cmask.cpu())
        print(f"[dump] push={p:.1f}×mg done, 累计样本 {len(rec['push']) * u.num_envs}")

    out = {k: torch.cat(v).numpy() for k, v in rec.items()}
    np.savez(args_cli.out, **out)
    print(f"[dump] 存 {args_cli.out}: friction_tip{out['friction_tip'].shape} "
          f"({'S-B倾斜' if args_cli.tilt else 'S-A上推'}); 接下来: python scripts/compare_shear_sim2real.py {args_cli.out}")
    env.close()


main()
simulation_app.close()
