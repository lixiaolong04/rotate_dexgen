"""Dump the hand articulation's PhysX material-shape structure at runtime.

Why: friction randomization sets per-SHAPE friction (not per-body). Xhand's URDF
has 75 collision links that USD merges into 22 bodies, so the material-shape count
and the shape indices of the 5 fingertip (dip_link) elastomers are unknown until
PhysX loads the asset. There is no shape->link API, so we must dump it once.

Usage:
    python scripts/inspect_material_shapes.py --headless
"""
import argparse
from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0",
                   help="Grasp variant needs no grasp cache.")
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    return args


def out(*a):
    print(*a, flush=True)


def main():
    args = parse_args()
    launcher = AppLauncher(args)
    app = launcher.app

    import os
    import gymnasium as gym
    import rl_isaaclab.tasks.inhand_rotate  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    # IMPORTANT: reset so PhysX views are fully initialized before we touch them
    env.reset()
    hand = env.unwrapped.hand

    out("\n" + "=" * 70)
    out("PART 1: PhysX material properties tensor shape")
    out("=" * 70)
    mat = hand.root_physx_view.get_material_properties()
    max_shapes = mat.shape[1]
    out(f"get_material_properties().shape = {tuple(mat.shape)}")
    out(f"  -> num_envs   = {mat.shape[0]}")
    out(f"  -> MAX_SHAPES = {max_shapes}   (set_friction must size to this)")
    out(f"  -> last dim   = {mat.shape[2]}  (static_fric, dynamic_fric, restitution)")
    out(f"  env0 friction (first 10 shapes) = {mat[0, :10, 0].tolist()}")

    out("\n" + "=" * 70)
    out("PART 2: body (rigid link) list")
    out("=" * 70)
    out(f"num_bodies = {hand.num_bodies}")
    for i, name in enumerate(hand.body_names):
        out(f"  body[{i:2}] {name}")

    out("\n" + "=" * 70)
    out("PART 3: USD CollisionAPI prims under env_0 Robot")
    out("=" * 70)
    from pxr import Usd, UsdPhysics, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    robot_prim = None
    for prim in stage.Traverse():
        if prim.GetPath().pathString.endswith("/env_0/Robot"):
            robot_prim = prim
            break
    robot_root = robot_prim.GetPath().pathString if robot_prim else None
    out(f"robot root prim = {robot_root}")

    # Collisions live inside INSTANCED prims; normal Traverse() skips them.
    # Use a PrimRange with instance-proxy traversal to descend into instances.
    colliders = []
    rng = Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())
    for prim in rng:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(prim.GetPath().pathString)

    out(f"\ncollision prim count under Robot = {len(colliders)}")
    out(f"max_shapes from PhysX            = {max_shapes}")
    match = (len(colliders) == max_shapes)
    out(f"MATCH? {'YES — stage order candidate for shape order' if match else 'NO — counts differ'}")

    out(f"\nall collision prims (index = stage traversal order):")
    dip_indices = []
    for idx, path in enumerate(colliders):
        short = path.replace(robot_root + "/", "") if robot_root else path
        is_dip = "_dip_link" in path
        if is_dip:
            dip_indices.append(idx)
        out(f"  shape[{idx:2}] {short}{'   <-- DIP' if is_dip else ''}")

    out("\n" + "=" * 70)
    out("PART 4: summary")
    out("=" * 70)
    out(f"MAX_SHAPES = {max_shapes}")
    out(f"collision prim count = {len(colliders)}")
    out(f"dip_link shape indices (stage order) = {dip_indices}  (expect 5)")
    out("NOTE: dip indices valid ONLY IF PART 3 MATCH=YES.")

    # Isaac Sim's app.close() segfaults on this box; skip it, force-exit cleanly.
    out("\n[done] forcing clean exit (skipping buggy app.close)")
    os._exit(0)


if __name__ == "__main__":
    main()
