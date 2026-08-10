"""measure_joint_friction.py — 真机逐关节静摩擦(breakaway)辨识，把仿真摩擦从"猜"变成"量"。

原理: mode4 下对【单个】关节设 kp=0 / kd=安全阻尼, 缓慢加前馈力矩(torque), 读实际角;
  关节"开始动"(角度变化>阈值)那一刻的力矩 = 静摩擦(Coulomb breakaway)。
  其余关节用 kp=hold_kp 稳稳保持半开手姿。顺带记录"动时速度" → 可后续估粘滞阻尼。

⚠️ 安全: 手必须【空着】(无物体) + 手指能自由动; 力矩缓升、一动(0.05rad)就停、有上限保护;
        每次只驱一个关节、动幅被限在 ~0.05rad。手放急停旁, 异常按 Ctrl-C / 断电。
用法:
  python scripts/measure_joint_friction.py --dry_run        # 离线 smoke(fake_hw 假静摩擦桩)
  python scripts/measure_joint_friction.py                  # 真机(全 21 关节)
  python scripts/measure_joint_friction.py --joints 14,19   # 只测某些关节(IsaacLab序)
"""
import argparse
import json
import time

import numpy as np


def build_hw(cfg, dry_run):
    if dry_run:
        from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
        hw = FakeHWInterface()
    else:
        from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
        hw = XhandHWInterface(cfg)
    hw.connect()
    return hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--joints", default=None, help="只测这些(IsaacLab序,逗号分隔); 默认全21")
    ap.add_argument("--max_torque", type=float, default=1.5, help="力矩上限Nm(到此没动=跳过)")
    ap.add_argument("--ramp_rate", type=float, default=0.2, help="力矩爬升 Nm/s(越慢越准)")
    ap.add_argument("--move_thresh", type=float, default=0.05, help="判定'动了'的角度阈值rad")
    ap.add_argument("--hold_kp", type=float, default=8.0, help="非测试关节的保持kp")
    ap.add_argument("--safe_kd", type=float, default=0.5, help="测试关节的安全阻尼kd(防失控)")
    ap.add_argument("--rate", type=float, default=50.0, help="控制频率Hz")
    ap.add_argument("--out", default="cache/joint_friction.json")
    args = ap.parse_args()

    from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
    cfg = DeployCfg()
    hw = build_hw(cfg, args.dry_run)
    N = 21
    JN = [n.replace("right_hand_", "").replace("_joint", "") for n in cfg.joint_names]
    joints = [int(x) for x in args.joints.split(",")] if args.joints else list(range(N))
    dt = 1.0 / args.rate
    lo, hi = np.asarray(cfg.dof_lower, np.float32), np.asarray(cfg.dof_upper, np.float32)
    neutral = np.clip(lo + 0.3 * (hi - lo), lo, hi).astype(np.float32)   # 半开手, 各关节留上行空间
    _sleep = (lambda t: None) if args.dry_run else time.sleep

    def hold(test_j=None, torque=0.0):
        """全关节 kp=hold_kp 保持 neutral; test_j 改 kp=0/kd=safe + 给前馈力矩(纯力矩控制)。"""
        kp = np.full(N, args.hold_kp, np.float32)
        kd = np.full(N, 0.4, np.float32)
        tq = np.zeros(N, np.float32)
        if test_j is not None:
            kp[test_j], kd[test_j], tq[test_j] = 0.0, args.safe_kd, float(torque)
        hw.set_joint_raw(neutral, kp, kd, tq, mode=4)

    print(f"=== 关节静摩擦辨识 | {'DRY-RUN(假桩)' if args.dry_run else '真机'} | 测 {len(joints)} 个关节 ===")
    print("⚠️ 确认: 手空着(无物体) + 手指能自由动 + 急停就绪。2秒后到半开手姿开始...")
    hold()
    _sleep(2.0)

    res = {}
    max_steps = int(args.max_torque / args.ramp_rate * args.rate) + 1
    for j in joints:
        hold()
        _sleep(0.4)
        p0 = float(hw.get_joint_pos()[j])
        prev, brk, vel_at, rd_at = p0, None, 0.0, 0.0
        for s in range(1, max_steps):
            T = s * args.ramp_rate * dt              # 前馈力矩缓升
            hold(test_j=j, torque=T)
            _sleep(dt)
            cur = float(hw.get_joint_pos()[j])
            rd = float(hw.read_joint_diag()[0][j])   # 读实际电机力矩
            if abs(cur - p0) > args.move_thresh:     # 动了 -> breakaway
                brk, vel_at, rd_at = T, (cur - prev) / dt, rd
                break
            prev = cur
        hold()                                       # 立即复位该关节
        if brk is not None:
            res[JN[j]] = {"breakaway_Nm": round(float(brk), 3), "read_Nm": round(float(rd_at), 3),
                          "vel_rad_s": round(float(vel_at), 3)}
            print(f"  {JN[j]:11s}(j{j:2d}): 静摩擦≈{brk:.3f}Nm (读{rd_at:.3f}) 动时速度{vel_at:+.2f}rad/s")
        else:
            res[JN[j]] = {"breakaway_Nm": None, "note": f"到{args.max_torque}Nm没动"}
            print(f"  {JN[j]:11s}(j{j:2d}): 到 {args.max_torque}Nm 仍没动 ⚠️(限位?太涩?)")
        _sleep(0.2)

    hold()
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    vals = [r["breakaway_Nm"] for r in res.values() if r.get("breakaway_Nm")]
    print()
    if vals:
        print(f"=== 汇总: 静摩擦 {min(vals):.3f}~{max(vals):.3f}Nm  均值 {np.mean(vals):.3f}Nm ===")
        print(f"→ 建议仿真 actuator friction ≈ {np.mean(vals):.2f}, 随机化 [{min(vals):.2f}, {max(vals):.2f}]")
        print("  (粘滞阻尼可从各关节'动时速度'后续细估; 当前主拿 Coulomb 静摩擦)")
    else:
        print("=== 所有关节都没动到上限 —— 检查手是否空着/限位/把 --max_torque 调大 ===")
    print(f"已存 {args.out}")
    if not args.dry_run:
        hw.disconnect()


if __name__ == "__main__":
    main()
