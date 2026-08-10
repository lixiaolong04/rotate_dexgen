"""Print all revolute joints in the Xhand URDF with their limits and axes.

Usage:
    python scripts/inspect_joints.py [--side right|left]

Output goes to stdout. Intended as a sanity check before training.
Exits non-zero if revolute joint count != 21.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_urdf(urdf_path: Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype != "revolute":
            continue
        name = joint.get("name")
        axis_el = joint.find("axis")
        limit_el = joint.find("limit")
        axis = axis_el.get("xyz") if axis_el is not None else "?"
        lower = limit_el.get("lower") if limit_el is not None else "?"
        upper = limit_el.get("upper") if limit_el is not None else "?"
        effort = limit_el.get("effort") if limit_el is not None else "?"
        velocity = limit_el.get("velocity") if limit_el is not None else "?"
        joints.append((name, axis, lower, upper, effort, velocity))
    return joints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--names-only", action="store_true",
                        help="Print only joint names (for piping into cfg)")
    args = parser.parse_args()

    urdf = PROJECT_ROOT / "assets" / "xhand_green" / f"xhand_{args.side}" / "urdf" / f"xhand_{args.side}.urdf"
    assert urdf.exists(), f"URDF not found: {urdf}"

    joints = parse_urdf(urdf)

    if args.names_only:
        for name, *_ in joints:
            print(name)
        return

    print(f"URDF: {urdf}")
    print(f"Revolute joints: {len(joints)}")
    print(f"{'idx':>3}  {'name':<45} {'axis':<10} {'lower':>10} {'upper':>10} {'effort':>8} {'velocity':>8}")
    for i, (name, axis, lo, up, ef, vel) in enumerate(joints):
        print(f"{i:>3}  {name:<45} {axis:<10} {lo:>10} {up:>10} {ef:>8} {vel:>8}")

    if len(joints) != 21:
        raise SystemExit(f"FAIL: expected 21 revolute joints, got {len(joints)}")
    print(f"\nOK: Joint count == 21")


if __name__ == "__main__":
    main()
