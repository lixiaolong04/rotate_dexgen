"""Add damping + friction to all revolute joints in Xhand URDF.

Why: Xhand URDF currently has `damping="0"` on every joint and no friction.
This makes joints "frictionless springs" — they oscillate freely and can't
maintain contact force when zero-action. SharpaWave's USD has built-in
damping/stiffness (USD units 0.001-0.004 / 0.05-0.12) which is why same
gen_grasp code converges for SharpaWave but not Xhand.

This script modifies the URDF in-place to add SharpaWave-equivalent dynamics:
    damping = 0.1   N·m·s/rad   (≈ SharpaWave 0.001-0.004 USD × 180/π)
    friction = 0.01 N·m         (static joint friction)

Run before convert_xhand_urdf.py.

Usage:
    python scripts/modify_xhand_urdf.py             # modifies both right and left
    python scripts/modify_xhand_urdf.py --side right
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAMPING_NM_S_PER_RAD = 0.1
FRICTION_NM = 0.01


def modify_urdf(urdf_path: Path):
    print(f"\n=== Modifying {urdf_path.relative_to(PROJECT_ROOT)} ===")
    # backup original (rename to .urdf.orig if not already saved)
    backup = urdf_path.with_suffix(".urdf.orig")
    if not backup.exists():
        shutil.copy2(urdf_path, backup)
        print(f"backed up → {backup.name}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    rev_count = 0
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        rev_count += 1
        # find or create <dynamics>
        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        old_damping = dyn.get("damping")
        old_friction = dyn.get("friction")
        dyn.set("damping", str(DAMPING_NM_S_PER_RAD))
        dyn.set("friction", str(FRICTION_NM))
        if rev_count <= 3:
            print(f"  joint {joint.get('name')}: "
                  f"damping {old_damping!r} → {DAMPING_NM_S_PER_RAD}, "
                  f"friction {old_friction!r} → {FRICTION_NM}")

    # Write back. Use default encoding (preserve original look as much as possible).
    tree.write(urdf_path, xml_declaration=True, encoding="utf-8")
    print(f"modified {rev_count} revolute joints in {urdf_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["right", "left", "both"], default="both")
    args = p.parse_args()

    sides = ["right", "left"] if args.side == "both" else [args.side]
    for side in sides:
        urdf = PROJECT_ROOT / f"assets/xhand_green/xhand_{side}/urdf/xhand_{side}.urdf"
        if not urdf.exists():
            print(f"WARN: {urdf} not found, skipping")
            continue
        modify_urdf(urdf)

    print("\nDone. Next step: run convert_xhand_urdf.py to regenerate USD.")


if __name__ == "__main__":
    main()
