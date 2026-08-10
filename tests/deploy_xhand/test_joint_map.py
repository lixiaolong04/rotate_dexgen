# tests/deploy_xhand/test_joint_map.py
import numpy as np
from rl_isaaclab.deploy_xhand.joint_map import isaaclab_to_xhand, xhand_to_isaaclab, ISAACLAB2XHAND_IDX

def test_roundtrip_identity():
    x = np.arange(21, dtype=np.float32)
    assert np.allclose(xhand_to_isaaclab(isaaclab_to_xhand(x)), x)
    assert np.allclose(isaaclab_to_xhand(xhand_to_isaaclab(x)), x)

def test_is_permutation_of_21():
    assert sorted(ISAACLAB2XHAND_IDX) == list(range(21))

def test_nonidentity_semantics(monkeypatch):
    """锁定映射方向: 关节 i 的值必须落在 xhand 槽 p[i]; 防止有人把方向改反。"""
    import rl_isaaclab.deploy_xhand.joint_map as jm
    p = [2, 0, 1] + list(range(3, 21))
    inv = [0] * 21
    for i, j in enumerate(p):
        inv[j] = i
    monkeypatch.setattr(jm, "ISAACLAB2XHAND_IDX", p)
    monkeypatch.setattr(jm, "XHAND2ISAACLAB_IDX", inv)
    v_lab = np.arange(100, 121, dtype=np.float32)   # 关节 i 的值 = 100+i
    x = jm.isaaclab_to_xhand(v_lab)
    for i in range(21):
        assert x[p[i]] == v_lab[i]                  # xhand 槽 p[i] 装关节 i 的值
    assert np.allclose(jm.xhand_to_isaaclab(x), v_lab)  # 能精确还原

def test_xhand2isaaclab_is_true_inverse():
    """XHAND2ISAACLAB_IDX 必须是 ISAACLAB2XHAND_IDX 的真逆排列。"""
    from rl_isaaclab.deploy_xhand.joint_map import ISAACLAB2XHAND_IDX, XHAND2ISAACLAB_IDX
    for i, j in enumerate(ISAACLAB2XHAND_IDX):
        assert XHAND2ISAACLAB_IDX[j] == i

# --- tactile_map: Xhand 每指 3-pad(tip/pad1/pad2) sensor 布局; 真力计算在 hw_interface ---
from rl_isaaclab.deploy_xhand.tactile_map import FINGER_SENSOR_LAYOUT

def test_layout_5_fingers_3_pads():
    assert len(FINGER_SENSOR_LAYOUT) == 5            # thumb,index,mid,ring,pinky
    for finger in FINGER_SENSOR_LAYOUT:
        assert len(finger) == 3                       # tip, pad1, pad2

def test_layout_ids_unique():
    ids = [sid for finger in FINGER_SENSOR_LAYOUT for sid in finger]
    assert len(ids) == len(set(ids)) == 15            # 5×3 唯一 sensor id
