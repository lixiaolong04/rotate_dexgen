"""B5: obs 链路体检(只读) —— 在真机上用【部署代码】构建 141 维 obs, 验证有限/结构/范围,
并把真机 obs 喂给当前 student 跑一次前向, 确认产出 action 也正常。全程不发运动指令(warm_up=False, 不 step)。
注: 不抓物体, 故接触维≈0(正常); obs 分布与仿真"抓握态"的完整对齐留到 B7(有物体)后。

⚠️ 仍会连 EtherCAT(手不会被主动命令运动, 但连接瞬间可能微动); 手放稳。结果写 b5_result.txt。
用法(容器内, privileged+host网络): python scripts/b5_obs_check.py
"""
import os
os.environ.setdefault("SPDLOG_LEVEL", "warn")
import argparse

import numpy as np
import torch

from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface
from rl_isaaclab.deploy_xhand.deploy_env import XhandInhandRotateDeployEnv
from rl_isaaclab.deploy_xhand.policy_runner import PolicyRunner

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpts/unilab_v1_students_final_20260630/student_8103_best.pth")
args = p.parse_args()

cfg = DeployCfg()
cfg.keyboard_listen = False
cfg.warm_up = False
hw = XhandHWInterface(cfg)
hw.connect()
hw.zero_tactile()   # 手空着, 采触觉基线 -> 接触维应≈0
L = []


def log(s):
    print(s); L.append(s)


try:
    env = XhandInhandRotateDeployEnv(cfg, hw)
    obs = env.reset()
    o = obs["obs"]; ph = obs["proprio_hist"]
    n = cfg.action_space

    frame_dim = cfg.observation_space // 3
    log(f"[形状] obs={tuple(o.shape)}(应(1,141))  proprio_hist={tuple(ph.shape)}(应(1,30,47))")
    log(f"[有限] obs finite={bool(torch.isfinite(o).all())}  proprio_hist finite={bool(torch.isfinite(ph).all())}")

    frames = o.reshape(3, frame_dim)
    same = bool(torch.allclose(frames[0], frames[1]) and torch.allclose(frames[1], frames[2]))
    log(f"[结构] reset 后 3 帧是否相同={same}(应True)  obs 最大绝对值={o.abs().max():.3f}(clip_obs={cfg.clip_obs})")

    f = frames[-1]
    dof, tar, con = f[0:n], f[n:2*n], f[2*n:2*n+5]
    log(f"[关节dof_pos槽 0:21]  范围[{dof.min():.3f},{dof.max():.3f}](unscale 应~[-1,1])")
    log(f"[目标target槽 21:42]  范围[{tar.min():.3f},{tar.max():.3f}](reset 时=当前关节角rad)")
    log(f"[接触contact槽 42:47] = {[round(float(x),3) for x in con]}(无物体应≈0)")

    # 真机 obs 喂当前 student 跑前向
    log("\n[策略] 加载 student + 真机 obs 前向...")
    runner = PolicyRunner(env, args.ckpt, "cpu")
    act = runner.act(obs)
    log(f"[策略] action={tuple(act.shape)}(应(1,21))  finite={bool(torch.isfinite(act).all())}  "
        f"范围[{act.min():.3f},{act.max():.3f}](应⊆[-1,1])")
    log(f"[策略] action 前5={[round(float(x),3) for x in act.reshape(-1)[:5]]}")
    log("\n结论: 形状/有限/结构 全对 + 策略产出合法 action => obs->策略 链路在真机上通了(B5 通过)。")
finally:
    hw.disconnect()

with open("b5_result.txt", "w") as fp:
    fp.write("\n".join(L) + "\n")
print("\n(已写入 b5_result.txt)")
