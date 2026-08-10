# rl_isaaclab/deploy_xhand/fake_hw.py
"""测试/离线 dry-run 用的硬件桩, 接口与 XhandHWInterface 完全一致(IsaacLab 顺序)。
让整条部署链路在没有真手时也能跑(approach A 的离线可测性)。"""
import numpy as np


class FakeHWInterface:
    def __init__(self, n=21, pos=None, tactile=None):
        self._pos = np.zeros(n, np.float32) if pos is None else np.asarray(pos, np.float32)
        self._tac = np.zeros(5, np.float32) if tactile is None else np.asarray(tactile, np.float32)
        self.last_targets = None
        self.connected = False
        self.relaxed = False

    def connect(self): self.connected = True
    def disconnect(self): self.connected = False

    def get_joint_pos(self):
        return self._pos.copy()            # IsaacLab 顺序, 弧度

    def read_joint_diag(self):
        z = np.zeros(21, np.float32)
        return z.copy(), z.copy() + 25.0, z.copy()   # torque/temp(25℃)/error 桩

    def get_tactile_raw(self):
        return self._tac.copy()            # (5,) 原始力 [thumb,index,mid,ring,pinky]; 缩放/阈值在 env

    def set_joint_targets(self, targets, kp, kd, ff_torque=None):
        self.last_targets = np.asarray(targets, np.float32).reshape(-1).copy()

    def set_joint_raw(self, pos, kp, kd, torque, mode=4):
        """离线桩 + 简易静摩擦模型(让 measure_joint_friction 的 dry-run 能测 breakaway 逻辑):
        力矩超过假静摩擦(每关节 0.3~0.5Nm 略不同)就让该关节按超出量位移。"""
        self.last_targets = np.asarray(pos, np.float32).reshape(-1).copy()
        tq = np.asarray(torque, np.float32).reshape(-1)
        fake_stiction = 0.3 + 0.01 * np.arange(len(tq))    # 假每关节静摩擦 0.30~0.50
        over = np.maximum(0.0, tq - fake_stiction)
        self._pos = self._pos + over * 0.05                # 超出部分→位移(模拟动起来)

    def relax(self, repeats=3, interval=0.05):
        self.relaxed = True

    def set_current_coeff(self, c): pass
    def set_speed_coeff(self, s): pass
    def calib_tactile(self): return True
    def zero_tactile(self, settle=0.0): return np.zeros(5, np.float32)   # 离线桩: 软件基线空操作

    # 测试中可直接改 _pos/_tac 模拟手运动
    def _set_state(self, pos=None, tac=None):
        if pos is not None: self._pos = np.asarray(pos, np.float32)
        if tac is not None: self._tac = np.asarray(tac, np.float32)
