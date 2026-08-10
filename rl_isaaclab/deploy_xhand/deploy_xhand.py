# rl_isaaclab/deploy_xhand/deploy_xhand.py
"""Xhand 真机部署入口(逐行照搬 sharpa deploy.py 的思路: 组装 env + 循环 act/step)。

env 内部负责: 20Hz 节拍 + 键盘状态机
(e=预抓取 / Enter=启动策略 / w=冻结继续 / q=缓慢回家 / t=触觉标定)
+ 暖启动抓取回放(回放 cache/deploy_init_traj_xhand.npy 到指尖接触)。entry 只做组装 + 主循环 + 急停。

分级安全(Xhand 专属): --action_scale_mul 0.3(S2) / 1.0(S3+);  --kp/--kd 覆盖(B4 标定值)。
--dry_run: 用 FakeHWInterface 离线验证(自动关 keyboard + warm_up, 无硬件即可跑); --max_steps N 跑够即停。
"""
import argparse
import os

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv
from rl_isaaclab.deploy_xhand.policy_runner import PolicyRunner


def build_hw(cfg, dry_run):
    if dry_run:
        from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
        return FakeHWInterface()
    from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
    return XhandHWInterface(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--load_path",
        default="ckpts/overnight_pose5/student_9402.pth",
        help="ProprioAdapt checkpoint matching the current 141-D/47-D deployment contract.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--comm", default=None, choices=["serial", "ethercat"])
    p.add_argument("--action_scale_mul", type=float, default=None)
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--kd", type=float, default=None)
    p.add_argument("--thresh", type=float, default=None, help="暖启动抓住阈值(纸杯等软物调小, 默认 cfg 1.0)")
    p.add_argument("--min-fingers", type=int, default=None, help="暖启动要≥几指接触才算抓住(默认 cfg 2; 夹不牢调大, 软物/单指调1)")
    p.add_argument("--torque", type=float, default=None, help="Green_JointCommand_User.torque(疑似力矩上限); 调小试能否降夹持力解死锁")
    p.add_argument("--mode", type=int, default=None, help="joint_cmd_mode: 官方工具用 4(kp/kd 生效!), 我们默认 3(kp/kd 被忽略)。改 4 让 kp 生效")
    p.add_argument("--speed", type=float, default=None, help="cmd_speed: mode=4 下会生效(厂商用0); 默认350在mode4下动作猛 -> 调小(如30)更慢更安全")
    p.add_argument("--friction_comp", action="store_true", help="开关节摩擦前馈补偿(Plan B: 软kp+补偿抵消真机摩擦→用回UniLab软kp); 自动确保 mode4+speed0")
    p.add_argument("--friction_comp_scale", type=float, default=None, help="摩擦补偿系数(默认cfg 0.8=实测80%); 抖/冲过头调小, 还堵转调大")
    p.add_argument("--no_keyboard", action="store_true", help="关键盘状态机, 自动跑(真机慎用)")
    p.add_argument("--no_warm_up", action="store_true", help="跳过暖启动抓取回放")
    p.add_argument("--max_steps", type=int, default=0, help=">0 时跑够步数自动停 (dry_run 用)")
    p.add_argument("--temp_limit", type=float, default=0.0, help="关节温度超此值(℃)自动停机; 0=关(默认不查温度, 自己掌控散热)")
    p.add_argument("--log", default=None, help="记录每步(策略态) action/目标角/实际角/接触/obs 到 npz, 供 analyze_deploy_log.py 排查")
    p.add_argument("--random-grasp", action="store_true", help="暖启动从训练 grasp cache 随机抽一个抓取(更忠于训练, 每次按e不同)")
    p.add_argument("--grasp-cache", default="cache/xhand_grasp_linspace_0.5-0.5-1.npy", help="--random-grasp 用的 cache 路径")
    p.add_argument("--close-extra", type=float, default=None, help="暖启动合到grasp还没碰到时, 弯曲【最多】再多合rad(≥min-fingers指接触即停, 默认0.3); 物体小/远调大")
    p.add_argument("--disable-tactile-ids", default=None, help="逗号分隔要清零触觉的指id(0拇1食2中3无名4小), 如 1,4=食指小指; V1 mask 部署对齐(训练清零了这俩指触觉, 真机也须清零否则OOD)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    if os.environ.get("XHAND_DEPLOY_OFFLINE_ONLY") == "1" and not args.dry_run:
        p.error(
            "当前是 xhand-deploy:offline 镜像，不包含厂商 SDK；"
            "请添加 --dry_run，或构建 xhand-deploy:22.04 真机镜像"
        )

    cfg = DeployCfg()
    cfg.device = args.device
    if args.comm:
        cfg.comm_kind = args.comm
    if args.action_scale_mul is not None:
        cfg.action_scale_mul = args.action_scale_mul
    if args.kp is not None:
        cfg.kp = args.kp
    if args.kd is not None:
        cfg.kd = args.kd
    if args.thresh is not None:
        cfg.grasp_contact_thresh = args.thresh
    if args.min_fingers is not None:
        cfg.grasp_min_fingers = args.min_fingers
    if args.torque is not None:
        cfg.cmd_torque = args.torque
    if args.mode is not None:
        cfg.joint_cmd_mode = args.mode
    if args.friction_comp:
        cfg.friction_comp = True
        if cfg.joint_cmd_mode != 4:
            print("[friction_comp] ⚠️ 前馈力矩只在 mode4 生效, 自动切 joint_cmd_mode=4(软kp也需要 mode4)")
            cfg.joint_cmd_mode = 4
        if cfg.cmd_speed != 0.0:
            print("[friction_comp] mode4 下 cmd_speed 须0(非0→瞬间握紧), 自动设0; --speed 可覆盖")
            cfg.cmd_speed = 0.0
    if args.friction_comp_scale is not None:
        cfg.friction_comp_scale = args.friction_comp_scale
    if args.random_grasp:
        cfg.grasp_cache_path = args.grasp_cache
    if args.close_extra is not None:
        cfg.warmup_close_extra = args.close_extra
    if args.disable_tactile_ids is not None:
        cfg.disable_tactile_ids = [int(x) for x in args.disable_tactile_ids.split(",") if x.strip() != ""]
        print(f"[deploy] disable_tactile_ids={cfg.disable_tactile_ids} (清零这些指触觉, 对齐 V1 mask 训练)")
    if args.speed is not None:
        cfg.cmd_speed = args.speed
    elif args.mode == 4:
        cfg.cmd_speed = 0.0   # mode=4(MIT)下 speed=目标速度, 不给则强制 0(否则驱动关节→瞬间握紧)
        print("[deploy] mode=4(MIT): 自动设 cmd_speed=0(目标速度), 避免瞬间握紧")
    # dry_run / 显式开关: 关键盘 + 暖启动(离线无硬件、无 pynput 也能跑)
    if args.dry_run or args.no_keyboard:
        cfg.keyboard_listen = False
    if args.dry_run or args.no_warm_up:
        cfg.warm_up = False

    hw = build_hw(cfg, args.dry_run)
    hw.connect()
    env = XhandInhandRotateDeployEnv(cfg, hw)
    runner = PolicyRunner(env, args.load_path, device=args.device)

    _kp_show = "逐关节(指尖0.9~拇指13.2)" if hasattr(cfg.kp, "shape") else cfg.kp
    _kd_show = "逐关节" if hasattr(cfg.kd, "shape") else cfg.kd
    print(f"[deploy] start: comm={cfg.comm_kind} keyboard={cfg.keyboard_listen} "
          f"warm_up={cfg.warm_up} mul={cfg.action_scale_mul} kp={_kp_show} kd={_kd_show}")
    if cfg.keyboard_listen:
        print("[deploy] 键盘: e=缓慢预抓取  Enter=启动策略  w=冻结/继续  q=缓慢张开  t=触觉标定")
    print("[deploy] 停止: w=安全保持；Ctrl-C / `touch STOP`=断连；断电=硬急停")
    if os.path.exists("STOP"):
        os.remove("STOP")

    _fr = cfg.observation_space // 3              # 2026-06-17 帧宽自适应(obs141→47); 原硬编166:171是obs186的, obs141下越界→contact记成空
    CON = slice(2 * _fr + 42, 2 * _fr + 47)       # obs 末帧接触槽: 21jpos+21tgt 之后5维, [94+42:94+47]=[136:141]
    log = {"action": [], "target": [], "actual": [], "contact": [], "obs": [], "torque": [], "temp": [],
           "shear_nf": [], "shear_tf": [], "shear_tfdir": []} if args.log else None
    obs = env.reset()
    n = 0
    try:
        while True:
            if os.path.exists("STOP"):   # 信号无关的可靠软停
                print("[deploy] STOP 文件 -> 停")
                break
            a = runner.act(obs)
            obs = env.step(a)   # env.step 内含 20Hz 节拍
            # 记录(仅策略实际驱动时: 无键盘=一直驱动; 有键盘=state 3)
            if log is not None and (not cfg.keyboard_listen or env.deploy_state_flag.get() == 3):
                log["action"].append(a.reshape(-1).cpu().numpy().copy())
                log["target"].append(env.cur_targets.reshape(-1).cpu().numpy().copy())
                log["actual"].append(env.hand_dof_pos.reshape(-1).cpu().numpy().copy())
                log["contact"].append(obs["obs"][0, CON].cpu().numpy().copy())
                log["obs"].append(obs["obs"].reshape(-1).cpu().numpy().copy())
                _tq, _tp, _ = hw.read_joint_diag()
                log["torque"].append(_tq.copy()); log["temp"].append(_tp.copy())
                # [剪切验证] 读每指指尖 nf/tf/tfDir 入 log(不进 obs/不改策略); 看剪切是否随物体爬升变化
                _snf, _stf, _sdir = hw.get_shear_diag()
                log["shear_nf"].append(_snf.copy()); log["shear_tf"].append(_stf.copy()); log["shear_tfdir"].append(_sdir.copy())
                if n % 20 == 0:   # ~每秒一行拇指剪切, 边转边肉眼对照物体有没有往上爬
                    print(f"[shear] 拇指 nf={_snf[0]:7.0f} tf={_stf[0]:7.0f} dir={_sdir[0]:6.0f}   (dir=65535=无接触)")
            n += 1
            if args.temp_limit > 0 and n % 20 == 0:   # 仅 --temp_limit>0 时查温度自动停机(默认关)
                _, _tp, _ = hw.read_joint_diag()
                if float(_tp.max()) > args.temp_limit:
                    print(f"[deploy] 🔴 关节温度 {_tp.max():.0f}℃ > {args.temp_limit}℃ -> 自动停机! 歇凉再跑。")
                    break
            if args.max_steps and n >= args.max_steps:
                print(f"[deploy] reached {n} steps, stop")
                break
    except KeyboardInterrupt:
        print("\n[deploy] 停止: 释放舵机后断连")
    finally:
        try:
            print("[deploy] 所有关节切换到 mode=0（idle/松手）...")
            hw.relax()
        except Exception as e:
            print(f"[deploy] ⚠️ mode=0 松手失败，将继续断开通信: {e!r}")
        finally:
            hw.disconnect()
        if log is not None and len(log["action"]):
            np.savez(args.log, **{k: np.array(v) for k, v in log.items()})
            print(f"[deploy] 日志已存 {args.log} ({len(log['action'])} 策略步); 分析: python scripts/analyze_deploy_log.py {args.log}")


if __name__ == "__main__":
    main()
