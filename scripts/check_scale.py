"""Verify object scale randomization actually produces different-sized objects.

Launches the Grasp task (no grasp cache needed), resets, then for each env reads
the object prim's xformOp:scale AND its real world-space AABB size. If scale
randomization works, env 0..N should show DIFFERENT sizes spanning scale_range;
if they're all identical, the scale event silently no-op'd (e.g. on instanced prims).

Usage:
    python scripts/check_scale.py --num_envs 8 --headless
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
import sys
sys.argv = [sys.argv[0]]

app = AppLauncher(args).app

import os
import gymnasium as gym
import rl_isaaclab.tasks.inhand_rotate  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def out(*a):
    print(*a, flush=True)


cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
out(f"\n[check_scale] task={args.task}  num_envs={args.num_envs}  cfg.scale_range={cfg.scale_range}")
env = gym.make(args.task, cfg=cfg)
env.reset()

import omni.usd
from pxr import UsdGeom, Usd

stage = omni.usd.get_context().get_stage()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], useExtentsHint=False)

out("\n" + "=" * 72)
out(f"{'env':>4}  {'xformOp:scale':<28}  {'world AABB size (x,y,z)':<30}")
out("-" * 72)
sizes = []
for i in range(args.num_envs):
    path = f"/World/envs/env_{i}/object"
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        out(f"{i:>4}  <prim not found at {path}>")
        continue
    attr = prim.GetAttribute("xformOp:scale")
    sc = attr.Get() if attr and attr.IsValid() else None
    try:
        rng = bbox_cache.ComputeWorldBound(prim).GetRange()
        sz = rng.GetSize()
        szt = (round(sz[0], 4), round(sz[1], 4), round(sz[2], 4))
        sizes.append(sz[0])
    except Exception as e:
        szt = f"<bbox err: {e}>"
    out(f"{i:>4}  {str(sc):<28}  {str(szt):<30}")

out("=" * 72)
if sizes:
    out(f"AABB x-size: min={min(sizes):.4f}  max={max(sizes):.4f}  "
        f"spread={max(sizes)-min(sizes):.4f}")
    if max(sizes) - min(sizes) < 1e-3:
        out("=> ALL SAME SIZE: scale randomization NOT working (silently no-op'd).")
    else:
        out("=> sizes DIFFER: scale randomization IS working.")
out("", )
os._exit(0)
