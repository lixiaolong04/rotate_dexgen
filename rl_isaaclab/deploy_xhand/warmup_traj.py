"""暖启动抓取轨迹: 从一个 grasp 姿势(21-dof, IsaacLab 序)构建【两段式 先张后合】轨迹,
或从训练 grasp cache 随机抽一个 grasp。两段式防大拇指卡(mcp0未张时mcp1卡死); 四指侧摆收0防互顶。
被 deploy_env._reset_idx(--random-grasp 时) 和 scripts/gen_deploy_init_traj_xhand.py 共用。"""
import numpy as np

# 第一段先动的"张开/定向"关节: 所有侧摆(mcp0) + 小指掌部(ltspalm)
SPREAD_JOINTS = {
    "right_hand_thumb_mcp0_joint", "right_hand_index_mcp0_joint", "right_hand_mid_mcp0_joint",
    "right_hand_ring_mcp0_joint", "right_hand_pinky_mcp0_joint", "right_hand_ltspalm_joint",
}
# 四指侧摆收 0 防中指/无名指尖互顶堵转(拇指侧摆保留)
DEPLOY_ABDUCT = {
    "right_hand_index_mcp0_joint": 0.0, "right_hand_mid_mcp0_joint": 0.0,
    "right_hand_ring_mcp0_joint": 0.0, "right_hand_pinky_mcp0_joint": 0.0,
}
N1, N2 = 20, 40   # 第一段(张)步数, 第二段(合)步数


def build_two_phase_traj(grasp, names, reduce_abduct=True):
    """grasp: (21,) IsaacLab 序的目标抓取姿势; names: 关节名(IsaacLab 序)。返回 (60,21) 两段式轨迹。"""
    grasp = np.array(grasp, np.float32).copy()
    if reduce_abduct:
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
    return np.stack(frames).astype(np.float32)


def random_cache_grasp(cache_path):
    """从训练 grasp cache 随机抽一个 grasp 的 21 关节(IsaacLab 序)。
    cache 每行 = 21 关节 + 3 物体位置 + 4 物体四元数 = 28。返回 (grasp21, obj_pos3)。"""
    c = np.load(cache_path)
    idx = int(np.random.randint(0, c.shape[0]))
    return c[idx, :21].astype(np.float32), c[idx, 21:24].astype(np.float32)
