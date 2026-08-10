"""Quick USD introspection: list rigid bodies + collision meshes of the sensor-links
USD, to confirm *_dip_sensor_link survived --no-merge and to derive the fingertip
sub-body list (dip/sensor/top_cos/top) for elastomer friction.

Usage: python scripts/inspect_sensor_bodies.py [usd_relpath] --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_args, _ = _p.parse_known_args()
_args.headless = True
_app = AppLauncher(_args).app

from pxr import Usd, UsdPhysics  # noqa: E402  (pxr available only after AppLauncher)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rel = next((a for a in sys.argv[1:] if not a.startswith("-")),
           "assets/xhand_green/xhand_right/urdf/xhand_right_sensor_links.usd")
usd = os.path.join(root, rel)
stage = Usd.Stage.Open(usd)
assert stage, f"cannot open {usd}"

bodies, colliders = [], []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        bodies.append(prim.GetName())
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        colliders.append(prim.GetPath().pathString)

print(f"=== {len(bodies)} rigid bodies ===")
for b in sorted(bodies):
    print(" ", b)

print(f"\n=== fingertip(dip) 相关 body(看 unmerge 拆出了哪些) ===")
for b in sorted(bodies):
    if "dip" in b or "top" in b:
        print(" ", b)

print(f"\n=== {len(colliders)} collision prims (前 40, 看 dip_sensor 有无collider) ===")
dip_cols = [c for c in colliders if "dip" in c or "top" in c]
for c in dip_cols:
    print(" ", c)

_app.close()
