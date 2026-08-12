# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys
import shutil

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--cache", type=str, default=None, help="Cache path.")
parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default=None, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
parser.add_argument("--diag_sq", action="store_true", default=False, help="Diagnostic: 复现 scale+quat 训练条件 (quat 开 + 低重力 -0.05), 用来看 quat 旋转后为何老掉.")
parser.add_argument("--scale8", action="store_true", default=False, help="scale-only 策略(seed400)验证: 8 尺寸 0.4~0.6 + 掌心朝上 + 真重力. num_envs 需 ÷8.")
parser.add_argument("--friction_mul", type=float, default=1.0, help="(配 --scale8) 摩擦 = 训练base(0.1/0.8/0.5) × 此倍数. 1.0=标称, 0.5=滑, 2.0=黏. 扫范围验证真机鲁棒性.")
parser.add_argument("--so3", action="store_true", default=False, help="全 SO(3) 随机朝向 + 当前 XHand 10 尺寸分布 + 真重力 -9.81. num_envs 需为 10 的倍数.")
parser.add_argument(
    "--showcase_camera",
    action="store_true",
    default=False,
    help="将 viewer 聚焦到 env 0 的 XHand 近景，便于录制演示素材；不改变策略或动力学。",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg

import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree('outputs/')
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["algorithm"]["max_agent_steps"] = args_cli.max_agent_steps if args_cli.max_agent_steps is not None else agent_cfg["algorithm"]["max_agent_steps"]
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.showcase_camera:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.45, 0.45, 0.75)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.52)
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    if args_cli.diag_sq:
        # 诊断: 复现 scale+quat 训练条件 —— quat 开 + 低重力(训练时重力卡在 -0.05) + scale 用 cfg 的 [0.4,0.6,8].
        # 看 quat 旋转后的抓取在 reset 时成不成立, 以及转动时物体被甩出/飘出窗口的情况.
        env_cfg.reset_random_quat = True
        env_cfg.scale_range = [0.4, 0.6, 8]   # scale+quat 新策略(seed 300-303): 8 尺寸, num_envs 需 ÷8
        env_cfg.randomize_pd_gains = False
        env_cfg.randomize_friction = False
        env_cfg.randomize_com = False
        env_cfg.randomize_mass = False
        env_cfg.randomize_joint_pos_offset = False
        env_cfg.sim.gravity = (0, 0, -0.05)
        env_cfg.gravity_curriculum = False
    elif args_cli.scale8:
        # scale 策略验证: 8 尺寸 + 掌心朝上 + 真重力 -9.81 + 训练 base 摩擦×friction_mul. num_envs 需 ÷8.
        env_cfg.reset_random_quat = False
        env_cfg.scale_range = [0.4, 0.6, 8]
        env_cfg.randomize_pd_gains = False
        env_cfg.randomize_friction = True   # 用训练 base 摩擦(0.1/0.8/0.5)×friction_mul, 而非 USD 默认 0.5
        env_cfg.randomize_friction_scale_lower = args_cli.friction_mul
        env_cfg.randomize_friction_scale_upper = args_cli.friction_mul
        env_cfg.randomize_com = False
        env_cfg.randomize_mass = False
        env_cfg.randomize_joint_pos_offset = False
        env_cfg.sim.gravity = (0, 0, -9.81)
        env_cfg.gravity_curriculum = False
    elif args_cli.so3:
        # 当前 XHand student (8500/8103 等): 全 SO(3) 随机朝向 + 10 尺寸 + 真重力 -9.81.
        # 与 cache/xhand_grasp_linspace_0.4-0.7-10.npy 及当前训练分布保持一致。
        env_cfg.reset_random_quat = True
        env_cfg.scale_range = [0.4, 0.7, 10]
        env_cfg.randomize_pd_gains = False
        env_cfg.randomize_friction = True       # 训练 base 摩擦(2.0/1.0/2.0)×1.0, 避开 friction=False→USD默认0.5 的滑
        env_cfg.randomize_friction_scale_lower = 1.0
        env_cfg.randomize_friction_scale_upper = 1.0
        env_cfg.randomize_com = False
        env_cfg.randomize_mass = False
        env_cfg.randomize_joint_pos_offset = False
        env_cfg.sim.gravity = (0, 0, -9.81)
        env_cfg.gravity_curriculum = False
    else:
        env_cfg.reset_random_quat = False
        env_cfg.scale_range = [0.5, 0.5, 1]   # 旧策略(单尺寸/掌心朝上): seed100/200-203 等, num_envs 任意
        env_cfg.randomize_pd_gains = False
        env_cfg.randomize_friction = False   # eval under fixed nominal friction. Randomization now works (auto-derived elastomer shape ids in xhand_wave_env.py); off here so all seeds compare under identical conditions.
        env_cfg.randomize_com = False
        env_cfg.randomize_mass = False
        env_cfg.randomize_joint_pos_offset = False
        env_cfg.sim.gravity = (0, 0, -9.81)
        env_cfg.gravity_curriculum = False
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)
    
    # load the checkpoint
    resume_path = agent_cfg["load_path"]
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent.restore_test(resume_path)
    agent.test()

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
