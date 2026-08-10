"""B1: 只读连通性检查 —— 用【我们真实部署代码 XhandHWInterface】连 EtherCAT, 读 21 关节位置(IsaacLab 序)
+ 5 指触觉力, 不发任何动作指令。重跑 = 验证我们修好的代码路径(open()元组修复 + per-id 读)。
⚠️ EtherCAT connect 会启动主站, 手可能移到零位; 跑前确保手放稳、工作区清空、可随时断电。
用法(容器内, privileged + network host): python scripts/b1_read_check.py
"""
from rl_isaaclab.deploy_xhand.deploy_cfg import DeployCfg
from rl_isaaclab.deploy_xhand.hw_interface import XhandHWInterface

cfg = DeployCfg()
print(f"[B1] comm={cfg.comm_kind}  iface={cfg.ethercat_iface}  hand_id={cfg.hand_id}")

hw = XhandHWInterface(cfg)
hw.connect()
print("[B1] connect() OK (comm.open 成功)")
try:
    pos = hw.get_joint_pos()   # per-joint get_joint_state(id), 已转 IsaacLab 序
    print(f"[B1] 21 关节位置(IsaacLab 序, rad): {[round(float(x), 3) for x in pos]}")
    tac = hw.get_tactile_raw()  # per-sensor, 每指 3 块 nf 求和
    print(f"[B1] 5 指触觉力 [thumb,index,mid,ring,pinky]: {[round(float(x), 1) for x in tac]}")
finally:
    hw.disconnect()
    print("[B1] disconnect() done. 全程只读, 未发动作。")
