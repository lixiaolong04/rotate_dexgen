"""Convert modified Xhand URDF → USD using IsaacLab's URDF converter.

Run after modify_xhand_urdf.py. Will OVERWRITE existing USD assets at:
    assets/xhand_green/xhand_<side>/urdf/xhand_<side>.usd
    assets/xhand_green/xhand_<side>/urdf/configuration/*.usd

The previous USD assets are backed up to <side>_usd_backup/ on first run.

Usage:
    python scripts/convert_xhand_urdf.py --side right
"""
import argparse
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["right", "left", "both"], default="right")
    p.add_argument("--usd-suffix", default="",
                   help="追加到输出USD basename, 如 _sensor_links → xhand_right_sensor_links.usd (不覆盖base)")
    p.add_argument("--no-merge-fixed-joints", action="store_true",
                   help="不合并固定关节 → 保留 *_dip_sensor_link 等 pad 为独立 body(前pad触觉需要)")
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    args.headless = True
    return args


def main():
    args = parse_args()
    launcher = AppLauncher(args)
    app = launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    sides = ["right", "left"] if args.side == "both" else [args.side]

    for side in sides:
        urdf = PROJECT_ROOT / f"assets/xhand_green/xhand_{side}/urdf/xhand_{side}.urdf"
        usd_dir = urdf.parent
        usd_file = f"xhand_{side}{args.usd_suffix}.usd"

        assert urdf.exists(), f"URDF not found: {urdf}"

        # Backup existing USD assets (one-time). Skip for a suffixed variant: we write a
        # NEW file (xhand_<side>_sensor_links.usd) and never overwrite the base usd.
        backup_dir = usd_dir.parent / f"{side}_usd_backup"
        if not backup_dir.exists() and not args.usd_suffix:
            backup_dir.mkdir(parents=True)
            for f in usd_dir.glob(f"xhand_{side}.usd*"):
                shutil.copy2(f, backup_dir / f.name)
                print(f"backed up {f.name} → {backup_dir.relative_to(PROJECT_ROOT)}")
            config_dir = usd_dir / "configuration"
            if config_dir.exists():
                shutil.copytree(config_dir, backup_dir / "configuration")
                print(f"backed up configuration/ → {backup_dir.relative_to(PROJECT_ROOT)}")

        print(f"\n=== Converting {urdf.name} → {usd_file} ===")
        # 版本自适应: 本地 IsaacLab v2.3.2 与云端 v2.2 的 UrdfConverterCfg API 不同。
        #   v2.3.2: collider_type(默认 convex_hull) + joint_drive.gains 必填
        #   v2.2  : convex_decompose_mesh(旧, False=convex_hull) + joint_drive 可选
        import dataclasses as _dc
        _fields = {f.name for f in _dc.fields(UrdfConverterCfg)}
        _kw = dict(
            asset_path=str(urdf),
            usd_dir=str(usd_dir),
            usd_file_name=usd_file,
            fix_base=True,
            merge_fixed_joints=not args.no_merge_fixed_joints,
            force_usd_conversion=True,
        )
        if "collider_type" in _fields:            # v2.3.2+
            # PD gains 必填; env 用 torque_control 会清零 USD drive(自己算力矩), 设0=等价无drive
            _kw["joint_drive"] = UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            )
        elif "convex_decompose_mesh" in _fields:  # v2.2(云端): 旧API, False=convex_hull 与 base usd 一致
            _kw["convex_decompose_mesh"] = False
        cfg = UrdfConverterCfg(**_kw)
        converter = UrdfConverter(cfg)
        print(f"saved USD → {converter.usd_path}")

    print("\nDone. Now test with:")
    print("  python scripts/smoke_test_xhand.py --num_envs 1 --steps 100 --headless")
    print("  python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0 --headless --num_envs 256")
    app.close()


if __name__ == "__main__":
    main()
