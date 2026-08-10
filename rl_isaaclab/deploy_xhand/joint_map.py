# rl_isaaclab/deploy_xhand/joint_map.py
"""IsaacLab articulation 关节顺序 <-> Xhand SDK 关节 ID 顺序 的 21 维映射。

约定: ISAACLAB2XHAND_IDX[i] = 第 i 个 IsaacLab 关节 在 Xhand SDK 向量里的槽位下标 (记 p[i])。
  - isaaclab_to_xhand: scatter, out[p[i]] = v[i]   (关节 i 的值 -> xhand 槽 p[i])
  - xhand_to_isaaclab: gather,  out[i]  = v[p[i]]   (从 xhand 槽 p[i] 取回关节 i 的值)
  两者对任意排列互为逆。XHAND2ISAACLAB_IDX = p 的逆排列, 供需要"按 xhand 槽反查 isaaclab 关节"的消费者使用。

✅ ISAACLAB2XHAND_IDX 已由 B2 真机实测填入(2026-06-09, scripts/inspect_xhand_joints.py 逐关节)。
Xhand id 解剖分组(每指 侧摆mcp0/弯曲mcp1/中间pip/指尖dip 连续):
  0-3=拇  4-7=食  8-11=中  12-15=无名  16-19=小  20=ltspalm(小指掌部)
名->Xhand id: thumb(mcp0=0,mcp1=1,pip=2,dip=3) index(4,5,6,7) mid(8,9,10,11) ring(12,13,14,15) pinky(16,17,18,19) ltspalm=20
对照 IsaacLab 顺序(cache/xhand_isaaclab_joint_meta.json)得下表 ISAACLAB2XHAND_IDX[i]=IsaacLab 关节 i 的 Xhand id:
  i=0 index_mcp0->4   1 ltspalm->20   2 mid_mcp0->8    3 ring_mcp0->12  4 thumb_mcp0->0
  5 index_mcp1->5     6 pinky_mcp0->16 7 mid_mcp1->9    8 ring_mcp1->13  9 thumb_mcp1->1
  10 index_pip->6     11 pinky_mcp1->17 12 mid_pip->10  13 ring_pip->14  14 thumb_pip->2
  15 index_dip->7     16 pinky_pip->18  17 mid_dip->11  18 ring_dip->15  19 thumb_dip->3  20 pinky_dip->19
"""
import numpy as np

# B2 真机实测排列(长度 21 的 0..20 排列); ISAACLAB2XHAND_IDX[i] = IsaacLab 关节 i 对应的 Xhand id。
ISAACLAB2XHAND_IDX = [4, 20, 8, 12, 0, 5, 16, 9, 13, 1, 6, 17, 10, 14, 2, 7, 18, 11, 15, 3, 19]

# 逆排列（自动推导，勿手填）
XHAND2ISAACLAB_IDX = [0] * 21
for _i, _j in enumerate(ISAACLAB2XHAND_IDX):
    XHAND2ISAACLAB_IDX[_j] = _i


def isaaclab_to_xhand(vec_isaaclab: np.ndarray) -> np.ndarray:
    """输入 IsaacLab 顺序 (21,)，输出 Xhand SDK 顺序 (21,)。"""
    v = np.asarray(vec_isaaclab)
    out = np.empty_like(v)
    out[ISAACLAB2XHAND_IDX] = v
    return out


def xhand_to_isaaclab(vec_xhand: np.ndarray) -> np.ndarray:
    """输入 Xhand SDK 顺序 (21,)，输出 IsaacLab 顺序 (21,)。"""
    v = np.asarray(vec_xhand)
    return v[ISAACLAB2XHAND_IDX]
