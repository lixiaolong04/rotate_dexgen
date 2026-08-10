"""B8: 真机跑当前 student 策略做手内旋转(安全分级版)。
流程: 暖启动两段式抓取(采基线->先张后合->接触即停) -> 确认抓住 -> 分级跑策略 -> 掉物检测 -> 急停冻结。

⚠️⚠️ 策略会持续驱动手指旋转物体! 跑前: 物体放好、支架固定、电源急停在手边。首跑务必小 scale。
分级建议(每级重跑一次, 看不乱再升): --scale 0.3 -> 0.5 -> 0.7 -> 1.0(满速)。
用法(容器内, -it): python scripts/b8_run_policy.py [--scale 0.3] [--secs 15]
  离线自测(无真手): python scripts/b8_run_policy.py --dry_run --secs 2
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse
import signal
import sys
import time

import numpy as np

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv
from rl_isaaclab.deploy_xhand.policy_runner import PolicyRunner

p = argparse.ArgumentParser()
p.add_argument("--scale", type=float, default=0.3, help="action_scale_mul 安全分级(0.3起步->1.0)")
p.add_argument("--secs", type=float, default=8.0, help="跑多少秒(20Hz); 兜底上限, 首测别设太长")
p.add_argument("--ckpt", default="ckpts/unilab_v1_students_final_20260630/student_8103_best.pth")
p.add_argument("--no-warmup", action="store_true", help="跳过暖启动抓取(物体已在手里时)")
p.add_argument("--drop-stop", type=int, default=40, help="接触丢失多少步(20Hz)判掉物冻结")
p.add_argument("--thresh", type=float, default=None, help="暖启动抓住阈值(纸杯等软物调小防压扁, 默认 cfg 1.0)")
p.add_argument("--min-fingers", type=int, default=None, help="暖启动要≥几指接触才算抓住(默认 cfg 2)")
p.add_argument("--torque", type=float, default=None, help="Green_JointCommand_User.torque(疑似力矩上限); 调小试能否降夹持力")
p.add_argument("--log", default=None, help="记录每步 action/目标角/实际角/接触/obs 到 npz, 供 analyze_deploy_log.py 排查")
p.add_argument("--dry_run", action="store_true", help="FakeHW 离线自测")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
cfg.warm_up = not args.no_warmup
cfg.action_scale_mul = args.scale
if args.thresh is not None:
    cfg.grasp_contact_thresh = args.thresh
if args.min_fingers is not None:
    cfg.grasp_min_fingers = args.min_fingers
if args.torque is not None:
    cfg.cmd_torque = args.torque

if args.dry_run:
    from rl_isaaclab.deploy_xhand.fake_hw import FakeHWInterface
    hw = FakeHWInterface(tactile=np.full(5, 1000.0, np.float32))   # 非零触觉, 免误判掉物
else:
    from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
    hw = XhandHWInterface(cfg)
hw.connect()
env = XhandInhandRotateDeployEnv(cfg, hw)
runner = PolicyRunner(env, args.ckpt, "cpu")

frame_dim = cfg.observation_space // 3
CON = slice(2 * frame_dim + 2 * cfg.action_space,
            2 * frame_dim + 2 * cfg.action_space + 5)  # obs 末帧 contact 槽；兼容当前 141-D/47-D
print(f"[B8] scale={cfg.action_scale_mul} warm_up={cfg.warm_up} secs={args.secs} ckpt={args.ckpt}")

if not args.dry_run:
    input(">>> 物体放好、支架固定、电源急停在手边, 回车开始【暖启动抓取】...")
obs = env.reset()   # 暖启动: 采基线 + 两段式抓取(已修)
con = obs["obs"][0, CON].numpy()
print(f"[B8] 暖启动后接触={[round(float(x),2) for x in con]}  max={con.max():.2f}  (抓住阈值={cfg.grasp_contact_thresh})")
if not args.dry_run and con.max() < cfg.grasp_contact_thresh:
    if input("⚠️ 似乎没抓住(接触<阈值)! 继续跑可能甩飞物体。输入 y 仍跑, 其它退出: ").strip() != "y":
        hw.disconnect(); sys.exit(0)
if not args.dry_run:
    input(f">>> 已抓住。回车开始跑策略(scale={cfg.action_scale_mul}, {args.secs}s)... 随时 Ctrl-C 急停")


def freeze():
    """冻结: 持续下发当前姿态 1s, 不松手(避免掉物)。"""
    hold = hw.get_joint_pos()
    for _ in range(20):
        hw.set_joint_targets(hold, cfg.kp, cfg.kd)
        time.sleep(0.05)


# 抢回 SIGINT/SIGTERM(SDK 在 connect 时吃掉了 Ctrl-C) -> 置标志, 循环每步检查后冻结退出。
# ⚠️ 若 SDK 在底层线程里又重置了信号, 这仍可能失效 -> 真正可靠急停永远是【断电】或另开终端 docker kill。
_stop = {"f": False}
def _on_sig(s, fr):
    _stop["f"] = True
    print("\n[B8] 收到停止信号 -> 本步后冻结停止...")
signal.signal(signal.SIGINT, _on_sig)
signal.signal(signal.SIGTERM, _on_sig)
if os.path.exists("STOP"):
    os.remove("STOP")   # 清掉上次残留
print("[B8] 急停: Ctrl-C / 另开终端 `touch STOP` / docker kill / 断电(最稳)")

steps = int(args.secs * cfg.control_freq)
lost = 0
log = {"action": [], "target": [], "actual": [], "contact": [], "obs": [], "torque": [], "temp": []} if args.log else None
try:
    for t in range(steps):
        if _stop["f"] or os.path.exists("STOP"):   # STOP 文件: 信号无关的可靠软停
            print(f"[B8] 已停止(t={t})。")
            break
        a = runner.act(obs)
        obs = env.step(a)
        con = obs["obs"][0, CON].numpy()
        if log is not None:
            log["action"].append(a.reshape(-1).numpy().copy())
            log["target"].append(env.cur_targets.reshape(-1).cpu().numpy().copy())
            log["actual"].append(env.hand_dof_pos.reshape(-1).cpu().numpy().copy())
            log["contact"].append(con.copy())
            log["obs"].append(obs["obs"].reshape(-1).cpu().numpy().copy())
            _tq, _tp, _ = hw.read_joint_diag()
            log["torque"].append(_tq.copy()); log["temp"].append(_tp.copy())
        if t % 10 == 0:
            print(f"  t={t:>4} 接触={[round(float(x),1) for x in con]} max={con.max():>4.1f}  |a|max={float(a.abs().max()):.2f}")
        if con.max() <= 0.0:
            lost += 1
            if lost >= args.drop_stop:
                print(f"  ⚠️ 接触丢失 {lost} 步 -> 判定掉物, 冻结停止!")
                break
        else:
            lost = 0
    else:
        print(f"[B8] 跑满 {steps} 步, 正常结束。")
    freeze()
except KeyboardInterrupt:
    print("\n[B8] 急停: 冻结当前姿态(不松手)")
    freeze()
finally:
    hw.disconnect()
    if log is not None and len(log["action"]):
        np.savez(args.log, **{k: np.array(v) for k, v in log.items()})
        print(f"[B8] 日志已存 {args.log} ({len(log['action'])} 步); 分析: python scripts/analyze_deploy_log.py {args.log}")
print("[B8] 完成。")
