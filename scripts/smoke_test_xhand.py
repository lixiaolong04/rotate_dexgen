"""Spawn N Xhand envs, run M random-action steps, print diagnostics.

Purpose: M1 Gate. Verifies USD loads, num_joints matches expectation, and
obs remain free of NaN/Inf under random actions.

Default task is the Grasp variant because it does not require a pre-existing
grasp cache file (the Wave task asserts cache presence at reset time and is
unusable until M2 gen_grasp completes).

Usage:
    # M1 smoke (headless first):
    python scripts/smoke_test_xhand.py --num_envs 1 --steps 100 --headless

    # M1 V1 visual check (GUI):
    python scripts/smoke_test_xhand.py --num_envs 1 --steps 200

    # M2 final smoke after gen_grasp ran (use Wave task):
    python scripts/smoke_test_xhand.py --num_envs 8 --steps 200 \\
        --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless
"""
import argparse
from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--expected_dofs", type=int, default=21)
    p.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0",
                   help="Task id. Grasp variant is safe before M2; Wave variant requires grasp cache.")
    p.add_argument("--action_scale", type=float, default=0.1,
                   help="Scale random action by this factor")
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    launcher = AppLauncher(args)
    app = launcher.app

    import torch
    import gymnasium as gym
    import rl_isaaclab.tasks.inhand_rotate  # noqa: F401  registers tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    print(f"[smoke] task: {args.task}, num_envs={args.num_envs}")
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    print(f"[smoke] env created. observation_space={env.observation_space}, action_space={env.action_space}")

    unwrapped = env.unwrapped
    num_joints = unwrapped.hand.num_joints
    print(f"[smoke] hand.num_joints = {num_joints}, expected = {args.expected_dofs}")
    if num_joints != args.expected_dofs:
        raise SystemExit(f"FAIL: num_joints {num_joints} != expected {args.expected_dofs}")

    print(f"[smoke] joint names ({len(unwrapped.hand.joint_names)}):")
    for i, name in enumerate(unwrapped.hand.joint_names):
        print(f"  [{i:2}] {name}")

    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    print(f"[smoke] reset OK. obs.shape={tuple(obs_t.shape)}")

    nan_steps = 0
    for step in range(args.steps):
        action = torch.randn(args.num_envs, env.action_space.shape[-1], device=unwrapped.device) * args.action_scale
        obs, reward, terminated, truncated, info = env.step(action)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        if torch.isnan(obs_t).any() or torch.isinf(obs_t).any():
            nan_steps += 1
            print(f"[smoke] NaN/Inf in obs at step {step}!")
        if step % 20 == 0:
            # grasp env returns int reward; wave env returns tensor
            if torch.is_tensor(reward):
                rew_str = f"{reward.float().mean().item():+.4f}"
            else:
                rew_str = f"{float(reward):+.4f} (scalar)"
            print(f"[smoke] step {step:>3} | reward={rew_str} | obs.norm={obs_t.norm().item():.4f}")

    print(f"[smoke] done. nan_steps={nan_steps}/{args.steps}")
    if nan_steps > 0:
        raise SystemExit(f"FAIL: {nan_steps} steps had NaN/Inf in obs")
    print(f"[smoke] OK -- all checks passed")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
