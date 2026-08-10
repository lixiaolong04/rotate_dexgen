# scripts/gen_deploy_init_traj_xhand.py
"""生成暖启动抓取轨迹(IsaacLab 顺序), 真机 _reset 按此逐步合拢、指尖触觉>阈值即停(抓住物体)。

抓取目标姿势两种来源:
  默认: cfg init_state.joint_pos(单一硬编码值)。
  --from-cache: 从【训练 grasp cache】抽一个真实抓取(更忠于训练 —— 训练每回合就是从这 5万个里随机采样)。

⚠️ 两段式(先张后合) —— 真机实测(B2): 大拇指【侧摆(mcp0)没张开时弯曲(mcp1)会机械卡死】。
   先张所有侧摆/掌部到位, 再合拢弯曲。四指侧摆收 0(防中指/无名指尖互顶堵转, B7 实测)。
"""
import argparse
import json
import numpy as np

GRASP_POSE = {   # pose4(2026-06-24 中指弯版 for Ø48, 重训中 seed9200); pose2(t701冠军)+历史见 doc/init_pose_history.md
    "right_hand_thumb_mcp0_joint": 0.880, "right_hand_thumb_mcp1_joint": 0.680,
    "right_hand_thumb_pip_joint": 0.190, "right_hand_thumb_dip_joint": 0.090,
    "right_hand_index_mcp0_joint": -0.010, "right_hand_index_mcp1_joint": 1.200,
    "right_hand_index_pip_joint": 0.300, "right_hand_index_dip_joint": 0.103,
    "right_hand_mid_mcp0_joint": 0.170, "right_hand_mid_mcp1_joint": 0.510,
    "right_hand_mid_pip_joint": 0.900, "right_hand_mid_dip_joint": 0.130,
    "right_hand_ring_mcp0_joint": 0.000, "right_hand_ring_mcp1_joint": 0.610,
    "right_hand_ring_pip_joint": 0.615, "right_hand_ring_dip_joint": 0.315,
    "right_hand_ltspalm_joint": 0.192, "right_hand_pinky_mcp0_joint": -0.000,
    "right_hand_pinky_mcp1_joint": 1.140, "right_hand_pinky_pip_joint": 0.220,
    "right_hand_pinky_dip_joint": 0.170,
}
SPREAD_JOINTS = {  # 第一段先动: 所有侧摆(mcp0) + 小指掌部
    "right_hand_thumb_mcp0_joint", "right_hand_index_mcp0_joint", "right_hand_mid_mcp0_joint",
    "right_hand_ring_mcp0_joint", "right_hand_pinky_mcp0_joint", "right_hand_ltspalm_joint",
}
DEPLOY_ABDUCT = {  # 2026-06-14 回退 pose2 四指侧摆
    "right_hand_index_mcp0_joint": -0.010, "right_hand_mid_mcp0_joint": 0.090,
    "right_hand_ring_mcp0_joint": 0.080, "right_hand_pinky_mcp0_joint": -0.000,
}
N1, N2 = 20, 40

p = argparse.ArgumentParser()
p.add_argument("--from-cache", action="store_true", help="从训练 grasp cache 抽一个真实抓取(更忠于训练)")
p.add_argument("--cache", default="cache/xhand_grasp_linspace_0.4-0.6-8.npy")
p.add_argument("--scale-bucket", type=int, default=4, help="scale 桶 0-(N-1); 10桶0.3-0.7 时 7≈Ø48(scale0.61)")
p.add_argument("--num-buckets", type=int, default=10, help="cache scale 桶数(_0.3-0.7-10→10, 旧_0.4-0.6-8→8)")
p.add_argument("--grasp-idx", type=int, default=0, help="桶内第几个抓取")
p.add_argument("--keep-abduct", action="store_true", help="保留 cache 原始四指侧摆(默认收0防互顶)")
args = p.parse_args()

with open("cache/xhand_isaaclab_joint_meta.json") as f:
    names = json.load(f)["joint_names"]                       # IsaacLab 顺序

if args.from_cache:
    c = np.load(args.cache)
    bucket = c.shape[0] // args.num_buckets
    pick = c[args.scale_bucket * bucket + args.grasp_idx]
    grasp = pick[:21].astype(np.float32)                     # 21 关节(IsaacLab 序)
    obj_pos, obj_quat = pick[21:24], pick[24:28]
    print(f"[from-cache] 桶{args.scale_bucket} idx{args.grasp_idx}: 该抓取对应【物体位置(相对env)】={[round(float(x),4) for x in obj_pos]} 四元数={[round(float(x),3) for x in obj_quat]}")
    print(f"             -> 真机放物体时让它大致在掌心这个相对位置(对齐仿真)。")
else:
    grasp = np.array([GRASP_POSE[n] for n in names], np.float32)

if not args.keep_abduct:   # 四指侧摆收 0(防互顶); 拇指侧摆保留
    for jn, v in DEPLOY_ABDUCT.items():
        grasp[names.index(jn)] = v

spread = np.array([n in SPREAD_JOINTS for n in names])

frames = []
for k in range(N1):                       # 第一段: 侧摆 0->grasp, 弯曲保持 0
    f = k / (N1 - 1)
    pose = np.zeros_like(grasp); pose[spread] = grasp[spread] * f
    frames.append(pose.copy())
for k in range(N2):                       # 第二段: 侧摆保持, 弯曲 0->grasp
    f = k / (N2 - 1)
    pose = grasp.copy(); pose[~spread] = grasp[~spread] * f
    frames.append(pose.copy())

traj = np.stack(frames).astype(np.float32)
np.save("cache/deploy_init_traj_xhand.npy", traj)
src = f"cache桶{args.scale_bucket}#{args.grasp_idx}" if args.from_cache else "cfg默认"
print(f"saved traj {traj.shape} (源={src}, 两段式 先张{N1}+后合{N2})")
print(f"  末帧拇指: 侧摆={traj[-1][names.index('right_hand_thumb_mcp0_joint')]:.3f} 弯曲={traj[-1][names.index('right_hand_thumb_mcp1_joint')]:.3f}")
