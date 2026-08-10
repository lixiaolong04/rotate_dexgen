# rl_isaaclab/deploy_xhand/hw_interface.py
"""XhandHWInterface: 唯一碰 hand_controller SDK 的单元(approach A 的"被替换的 Xhand 部分")。
对外暴露 IsaacLab 顺序的状态/指令, 与 FakeHWInterface 接口一致, 供 XhandInhandRotateDeployEnv 调用。

本文件已按【真实 Xhand SDK】(容器 introspect hand_controller 0.1.49)重写, 不再是 SharpaWave 接口:
  - 关节状态: get_joint_state_list(bool) -> (ErrorCode, str, list[Green_JointState_User]); 取 .position
  - 关节指令: Green_JointCommand_User(字段 kp/kd/position/speed/torque/mode) + set_joint_cmd_list
  - 触觉:     get_sensor_state_list(bool) -> (ErrorCode, str, list[Green_Sensor_User], 掌心)
              每指力 = 该指 tip/pad1/pad2 三块 pad 的法向力(first_ydds_nf+second_ydds_nf)聚合
  - 触觉清零: sensor_clear

⚠️ 仍需真机确认的项标 [B1](方法行为)/[B3](触觉布局与语义)——introspect 给到名字与签名, 但行为/数值要硬件验:
   [B1] comm.open() 的 ErrorCode 成功枚举名; Green_JointCommand_User.mode 的位置控制取值;
        set_joint_cmd_list 是否接受 list; get_joint_state_list 的 bool(是否 flush)语义。
   [B3] FINGER_SENSOR_LAYOUT 每指 tip/pad1/pad2 的真实 sensor_id; 法向力字段(first_ydds_nf/second_ydds_nf)
        语义与是否够用, 还是改用 HapticModel.compute_all_fingers(需 Haptic_Config 标定)。
"""
from contextlib import contextmanager
import os
import sys
import threading
import time

import numpy as np

from .joint_map import isaaclab_to_xhand, xhand_to_isaaclab
from .tactile_map import FINGER_SENSOR_LAYOUT, FINGERTIP_SENSOR_LAYOUT


class XhandHWInterface:
    def __init__(self, cfg):
        self.cfg = cfg
        import hand_controller as hc           # noqa: 仅此处 import
        self.hc = hc
        if cfg.comm_kind == "serial":
            self.comm = hc.serialCommunication(cfg.serial_device, cfg.serial_baud, 1000)  # pybind 名小写开头(已容器核对); [B1] timeout 参数
        elif cfg.comm_kind == "ethercat":
            self.comm = hc.etherCatCommunication(cfg.ethercat_iface)                       # pybind 名小写开头
        else:
            raise ValueError(cfg.comm_kind)
        self.hand = None  # 真机实测(2026-06-09): GreenHand 必须在 comm.open() 之后建, 否则 PDO 回调挂在从站映射前 -> 读全 0
        self._tac_baseline = np.zeros(5, dtype=np.float32)  # 软件触觉基线(zero_tactile 采集)
        self._native_log_lock = threading.Lock()

    @contextmanager
    def _quiet_native_sdk(self):
        """短暂屏蔽 SDK 高频 C++ info 输出，同时保留连接和异常阶段日志。

        hand_controller 没暴露 spdlog 级别 API，SPDLOG_LEVEL 环境变量对该
        预编译库也不生效，因此仅在批量状态读取期间重定向进程 fd 1/2。
        """
        if not getattr(self.cfg, "sdk_quiet", True):
            yield
            return
        with self._native_log_lock:
            saved_out = None
            saved_err = None
            null_fd = None
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                saved_out = os.dup(1)
                saved_err = os.dup(2)
                null_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(null_fd, 1)
                os.dup2(null_fd, 2)
            except Exception:
                if saved_out is not None:
                    os.dup2(saved_out, 1)
                    os.close(saved_out)
                if saved_err is not None:
                    os.dup2(saved_err, 2)
                    os.close(saved_err)
                if null_fd is not None:
                    os.close(null_fd)
                yield
                return
            try:
                yield
            finally:
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(saved_out)
                os.close(saved_err)
                os.close(null_fd)

    def connect(self):
        ret = self.comm.open()
        # 真机实测: open() 返回 (ErrorCode, msg, str) 元组, 不是裸 ErrorCode (2026-06-09 B1 实测修)
        ec = ret[0] if isinstance(ret, tuple) else ret
        assert ec == self.hc.ErrorCode.SUCCESS, f"comm open 失败: {ret}"
        # 关键顺序(对齐 C++ SDK 例子 + 官方工具): 先 open 扫从站, 再建 GreenHand 注册 PDO 回调。
        # 反了的话(我们之前在 __init__ 建)回调挂在映射前, 输入 PDO 不更新 -> 读全 0。
        self.hand = self.hc.GreenHand(self.cfg.hand_id, self.comm)
        # 再等几个 1ms PDO 周期让输入缓存填满(官方 get_pos 也是 INIT 后隔 ~1s 才读)。
        time.sleep(0.5)
        # Xhand 无 set_control_mode/current_coeff/speed_coeff(那是 SharpaWave); 控制模式/速度在每条指令里。
        # [B1] 触觉数据类型(0=raw/1=filtered/2=base/3=diff, 默认 diff); 如需指定:
        # for fid, layout in enumerate(FINGER_SENSOR_LAYOUT):
        #     for sid in layout: self.hand.set_sensor_data_type(sid, 3)

    def disconnect(self):
        self.comm.close()

    def get_joint_pos(self):
        # 真机实测(2026-06-09): get_joint_state_list 返回全 0(不解码位置)! 必须 per-joint get_joint_state(id)。
        xhand_pos = np.zeros(21, dtype=np.float32)
        with self._quiet_native_sdk():
            for i in range(21):
                _code, _msg, st = self.hand.get_joint_state(i, True)   # (ErrorCode, str, Green_JointState_User)
                if _code != self.hc.ErrorCode.SUCCESS:
                    raise RuntimeError(f"get_joint_state({i}) failed: {_msg}")
                xhand_pos[i] = st.position
        return xhand_to_isaaclab(xhand_pos)                            # -> IsaacLab 顺序, 弧度

    def read_joint_diag(self):
        """读每关节诊断: torque(Nm,±5=直接夹持力)/temperature(℃)/error_code, 返 IsaacLab 序 (3,21)。"""
        tq = np.zeros(21, np.float32); tp = np.zeros(21, np.float32); er = np.zeros(21, np.float32)
        with self._quiet_native_sdk():
            for i in range(21):
                _c, _m, st = self.hand.get_joint_state(i, True)
                if _c != self.hc.ErrorCode.SUCCESS:
                    raise RuntimeError(f"get_joint_state({i}) failed: {_m}")
                tq[i] = float(st.torque); tp[i] = float(st.temperature); er[i] = float(st.error_code)
        return xhand_to_isaaclab(tq), xhand_to_isaaclab(tp), xhand_to_isaaclab(er)

    def _finger_nf_raw(self):
        """每指法向力(first_ydds_nf+second_ydds_nf)之和, 未扣基线。
        tactile_fingertip_only=True: 只用指尖(dip)sensor, 对齐仿真(接触只在 *_dip_link 指尖); 否则 3 块 pad 求和。
        真机实测(2026-06-09): get_sensor_state_list 的 id/力不填, 必须 per-sensor get_sensor_state(id)。"""
        layout = FINGERTIP_SENSOR_LAYOUT if getattr(self.cfg, "tactile_fingertip_only", False) else FINGER_SENSOR_LAYOUT
        out = np.zeros(5, dtype=np.float32)
        with self._quiet_native_sdk():
            for k, sids in enumerate(layout):
                f = 0.0
                for sid in sids:
                    _code, _msg, s = self.hand.get_sensor_state(sid, True) # (ErrorCode, str, Green_Sensor_User)
                    if _code != self.hc.ErrorCode.SUCCESS:
                        raise RuntimeError(f"get_sensor_state({sid}) failed: {_msg}")
                    f += float(s.first_ydds_nf) + float(s.second_ydds_nf)  # 法向力(int 传感器单位)
                out[k] = f
        return out

    def zero_tactile(self, settle=1.0):
        """采当前(手指应空着)每指 nf 作软件基线, 之后 get_tactile_raw 减它。
        真机实测: 这是【零漂】, 每次上电偏置还不一样且连接后需 ~1s 才稳定/出现 ->
        先等 settle 秒让零漂稳定再采(用户实测建议)。SDK sensor_clear 对此无效, 故软件扣除。"""
        if settle > 0:
            time.sleep(settle)
        self._tac_baseline = self._finger_nf_raw()
        return self._tac_baseline

    def get_tactile_raw(self):
        """返回 (5,) 指尖力 [thumb,index,mid,ring,pinky], 已减软件基线、负的截 0;
        缩放/噪声门在 env.get_tactile_info(照搬 sharpa)。升级(B3): HapticModel(需 Haptic_Config)。"""
        return np.maximum(0.0, self._finger_nf_raw() - self._tac_baseline)

    def get_shear_diag(self):
        """[剪切验证用, 不进 obs] 每指【指尖】剪切诊断: 返回 (nf, tf, tfDir) 三个 (5,) 数组(IsaacLab 指序 thumb..pinky)。
        nf/tf = 指尖 first+second pad 的法向力/剪切力之和(原始传感器单位, nf 未扣基线);
        tfDir = tf 较大那个 pad 的剪切方向(原始 uint16, 未解码; 65535=无接触/默认值)。
        只读指尖 sensor(=接触主力 + 省 SDK 读, 不拖慢 20Hz)。
        用途: 真机跑策略时验证"物体被拇指搓往上爬"是否被剪切力感知到 → 决定要不要上方案 A。"""
        layout = FINGERTIP_SENSOR_LAYOUT
        nf = np.zeros(5, dtype=np.float32)
        tf = np.zeros(5, dtype=np.float32)
        tfdir = np.zeros(5, dtype=np.float32)
        with self._quiet_native_sdk():
            for k, sids in enumerate(layout):
                f_nf = 0.0
                f_tf = 0.0
                best_tf = -1.0
                best_dir = 65535.0
                for sid in sids:
                    _code, _msg, s = self.hand.get_sensor_state(sid, True) # Green_Sensor_User
                    if _code != self.hc.ErrorCode.SUCCESS:
                        raise RuntimeError(f"get_sensor_state({sid}) failed: {_msg}")
                    for tf_v, dir_v, nf_v in ((s.first_ydds_tf, s.first_ydds_tfDir, s.first_ydds_nf),
                                              (s.second_ydds_tf, s.second_ydds_tfDir, s.second_ydds_nf)):
                        f_nf += float(nf_v)
                        f_tf += float(tf_v)
                        if float(tf_v) > best_tf:       # 取剪切最大的 pad 的方向(=主接触面)
                            best_tf = float(tf_v)
                            best_dir = float(dir_v)
                nf[k] = f_nf
                tf[k] = f_tf
                tfdir[k] = best_dir
        return nf, tf, tfdir

    def set_joint_targets(self, targets_isaaclab, kp, kd, ff_torque=None):
        xt = isaaclab_to_xhand(np.asarray(targets_isaaclab, np.float32).reshape(-1))
        # 2026-06-17 逐关节 kp/kd: 数组→IsaacLab序转Xhand序(同 targets); 标量→None→广播(向后兼容 --kp/暖启动)
        xkp = isaaclab_to_xhand(np.asarray(kp, np.float32).reshape(-1)) if np.ndim(kp) > 0 else None
        xkd = isaaclab_to_xhand(np.asarray(kd, np.float32).reshape(-1)) if np.ndim(kd) > 0 else None
        # friction comp 前馈力矩(IsaacLab序数组)→Xhand序; None→用 cfg.cmd_torque(0, 旧行为)
        xff = isaaclab_to_xhand(np.asarray(ff_torque, np.float32).reshape(-1)) if ff_torque is not None else None
        cmds = []
        for i in range(21):
            c = self.hc.Green_JointCommand_User()
            c.position = float(xt[i])
            c.kp = float(xkp[i]) if xkp is not None else float(kp)
            c.kd = float(xkd[i]) if xkd is not None else float(kd)
            c.speed = float(self.cfg.cmd_speed)    # mode=4 生效=位置移动速度; mode=3 忽略
            c.torque = float(xff[i]) if xff is not None else float(self.cfg.cmd_torque)  # mode4 前馈力矩: friction_comp 时=逐关节摩擦补偿(有方向, 非恒定); 否则 cfg.cmd_torque(0)
            c.mode = self.cfg.joint_cmd_mode       # 3=kp/kd忽略, 4=kp/kd生效
            cmds.append(c)
        self.hand.set_joint_cmd_list(cmds)     # [B1] 批量下发(单关节用 set_joint_cmd(id, cmd))

    def set_joint_raw(self, pos_il, kp_il, kd_il, torque_il, mode=4):
        """逐关节 pos/kp/kd/torque(IsaacLab序数组)→ Xhand序 → SDK。摩擦辨识/底层控制用。
        mode4 下 torque=前馈力矩; 某关节设 kp=0/kd=安全阻尼 + torque 缓升 ≈ 纯力矩控制(测 breakaway)。"""
        conv = isaaclab_to_xhand
        xp = conv(np.asarray(pos_il, np.float32).reshape(-1))
        xkp = conv(np.asarray(kp_il, np.float32).reshape(-1))
        xkd = conv(np.asarray(kd_il, np.float32).reshape(-1))
        xtq = conv(np.asarray(torque_il, np.float32).reshape(-1))
        cmds = []
        for i in range(21):
            c = self.hc.Green_JointCommand_User()
            c.position = float(xp[i]); c.kp = float(xkp[i]); c.kd = float(xkd[i])
            c.speed = 0.0; c.torque = float(xtq[i]); c.mode = int(mode)
            cmds.append(c)
        self.hand.set_joint_cmd_list(cmds)

    def relax(self, repeats=3, interval=0.05):
        """将全部关节切到厂商定义的 mode=0（idle/释放电机控制）。

        position 使用退出瞬间的当前角度，其余增益、速度和力矩全部为 0；
        即使固件忽略 mode，也不会额外产生位置跳变。重复数帧后再断开
        EtherCAT，避免控制板保留最后一条 mode4 刚性命令。
        """
        try:
            pos = self.get_joint_pos()
        except Exception:
            pos = np.zeros(21, dtype=np.float32)
        zeros = np.zeros(21, dtype=np.float32)
        for _ in range(max(1, int(repeats))):
            self.set_joint_raw(pos, zeros, zeros, zeros, mode=0)
            time.sleep(max(0.0, float(interval)))

    # Xhand 无 current/speed coeff 全局系数(那是 SharpaWave); env 会调用这两个, 这里设为安全空操作。
    # 真机若需限速/限流, 走每条 Green_JointCommand_User 的 speed/torque 或 set_register_data。[B1]
    def set_current_coeff(self, c):
        pass

    def set_speed_coeff(self, s):
        pass

    def calib_tactile(self):
        # 真机实测: SDK sensor_clear 对 nf 偏置无效, 主用软件基线扣除; sensor_clear 仍调一下无害。
        try:
            self.hand.sensor_clear()
        except Exception:
            pass
        self.zero_tactile()                    # 软件基线(手指应空着时调用)
        return True                            # 必须返回 bool: env.calib_tactile 用 `if not self.hw.calib_tactile()` 判定, 返回数组会 ValueError
