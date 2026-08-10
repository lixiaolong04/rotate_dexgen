# rl_isaaclab/deploy_xhand/deploy_cfg.py
"""部署配置 —— 忠实照搬 sharpa_wave_deploy_env_cfg.py 的字段(22->21 适配),
外加 Xhand 专属的通信/PD/安全分级项。物理/数学常量来自训练 cfg, 不可改;
硬件相关项(kp/kd/force_scale/contact_threshold)由 Phase B 真机标定。"""
import json
import os

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class DeployCfg:
    # ---- 键盘状态机 (照搬 sharpa) ----
    keyboard_listen = True       # True: 不自动跑, 按 e 启动 / w 冻结 / q 回家 / t 触觉标定。
                                 # dry_run/离线测试时 entry 会改成 False(自动跑)。

    # ---- env (照搬 sharpa, 22->21 / 192->186) ----
    hand_side = 1                # 1=right (t603 是右手)
    action_space = 21
    observation_space = 141      # 2026-06-17 对齐训练v2: 丢15维零 contact_pos (帧 62→47); 与 student ckpt obs 一致
    prop_hist_len = 30
    priv_info_dim = 9            # 2026-06-17 对齐 UniLab: 8→9 (加物体 scale 通道, 与 student ckpt / policy_runner 一致)
    asymmetric_obs = False
    device = "cpu"

    # ---- 控制 (照搬 sharpa) ----
    warm_up = True               # 上策略前先回放抓取轨迹到指尖接触(抓住物体); dry_run 设 False
    control_freq = 20            # Hz (decimation 12 / dt 1/240)
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1.0 / 24
    current_coef = 0.3           # 电流系数, 保守起步
    speed_coef = 0.3             # 速度系数, 保守起步(sharpa 用 0.5, 我们更保守)
    dof_limits_scale = 0.9       # URDF 限位 × 该 scale (与训练一致)

    # ---- Xhand 专属: 通信 ----
    comm_kind = "ethercat"       # 真机 XH04 走以太网口 EtherCAT(2026-06-09 实机确认, 非串口)
    serial_device = "/dev/ttyUSB0"
    serial_baud = 3000000
    ethercat_iface = "enp129s0"  # 本机唯一有线网口(手插这); EtherCAT 需 sudo + 容器 privileged/host网络
    hand_id = 0                  # SDK 灵巧手 id, 默认 0
    sdk_quiet = True             # 屏蔽高频 get_joint_state 的 C++ info 刷屏；连接/错误信息仍保留
    # ⚠️ 重大修正(2026-06-10): mode=3 下 kp/kd/speed/torque 全被 SDK 忽略(B4 当时误以为"没用");
    #   真相: 官方 xhand1pro_test.py 用 mode=4, 这些字段才生效! 默认值改为厂商 JSON 的值(kp=12/kd=0.4/speed与torque=0),
    #   这样 mode=4 不加 flag 也是厂商默认(安全); mode=3 反正忽略。调软抓握: --mode 4 --kp 6/3。
    kp = 10.0                    # 2026-06-22 回退 uniform kp=10 (=kp=10 student 训练值, t701 proven); --kp 覆盖. (v2 逐关节软PD已弃, kp_per_joint 仍在__init__算但 kp!=None 故不用)
    kd = 0.3                     # 2026-06-22 uniform kd=0.3 (=训练); mode=4 生效
    joint_cmd_mode = 3           # 位置控制 mode; 3=kp/kd被忽略(B1-B8用), 4=kp/kd生效(官方工具用, 调软抓握用4)
    cmd_speed = 350.0            # speed; mode=3 忽略; ⚠️mode=4(MIT)下=【目标速度】, 必须0(非0驱动关节→瞬间握紧)! 见下方注释
    cmd_torque = 0.0             # ⚠️mode=4 下 torque 是【前馈力矩】, 非0 会让所有指恒定发力→瞬间握紧! 厂商用0。mode=3 忽略

    # ---- Xhand 专属: 安全分级 (S0->S4) ----
    action_scale_mul = 1.0       # S2 临时设 0.3, S3 恢复 1.0
    grasp_contact_thresh = 0.1   # 暖启动判定"抓住"的指尖力阈值(get_tactile_info 后的量级); 0.1=轻碰就算(物体被指节夹/指尖只蹭到也认)
    grasp_min_fingers = 2        # 暖启动要 ≥这么多指接触才算抓住(防只拇指碰到就停=夹不牢); 1=旧行为
    warmup_step_dt = 0.1         # 暖启动抓取每帧间隔(s); 60帧 ~6s, 慢慢合拢更稳更安全
    warmup_close_extra = 0.3     # 合到grasp还没碰到物体时, 弯曲【最多】再多合这么多rad; ≥min_fingers指接触即停; 0=关

    # ---- friction compensation (Plan B: 软kp + 前馈抵消关节摩擦 → 真机变低摩擦 → 用回 UniLab 软kp; 需 mode4) ----
    friction_comp = False         # True: mode4 前馈力矩 = scale×breakaway×tanh((target-current)/deadband) 抵消关节静摩擦
    friction_comp_scale = 0.8     # 实测摩擦的比例(保守); 抖/冲过头→调小, 还堵转→调大(≤1.2)
    friction_comp_deadband = 0.05 # rad; |target-current|<此值时前馈平滑趋0(防接近目标时方向抖)
    friction_comp_clip = 0.5      # Nm; 前馈力矩绝对值上限(安全, 远低于 5Nm 饱和)
    friction_json_path = os.path.join(_PROJECT_ROOT, "cache/joint_friction.json")

    # ---- 触觉 (照搬 sharpa get_tactile_info 语义) ----
    enable_tactile = True        # False -> 触觉/接触位置全 0
    force_scale = 0.00075        # nf->N 统一系数(像sharpa单一scale)。2026-06-10 用350ml水(3.4N)单标中指: nf~4500 -> 3.4/4500≈0.00075。
                                 #   目标: obs 出真实牛顿=对齐仿真(仿真接触也是N)。核对: tac_monitor 压水看是否~3.4N。详见 calib_fingertip
    binary_contact = False       # True 才按 contact_threshold 二值化(0/1)
    contact_smooth = 0.5         # 2026-06-17 对齐训练: 接触2帧平滑系数(smoothed = current×0.5 + prev×0.5)
    enable_contact_pos = False   # 接触位置(15维), 训练/部署均关 -> 全 0
    disable_tactile_ids = []     # 5 指触觉都好(实测), 无需屏蔽
    tactile_fingertip_only = True  # 每指只用指尖(dip)sensor, 对齐仿真(接触传感器只在 *_dip_link 指尖)。
                                 #   指尖 id 已验证 thumb3/index7/mid11/ring15/pinky19(calib_fingertip 2026-06-10); force_scale 已按指尖重标。
    contact_threshold = 0.1      # 真机噪声门: ×force_scale 后 <0.1(=原始 nf<40) 清零; B5 obs 对齐可再调

    # ---- 关节顺序+限位 (A1 产出, IsaacLab 顺序) ----
    joint_meta_path = os.path.join(_PROJECT_ROOT, "cache/xhand_isaaclab_joint_meta.json")
    init_traj_path = os.path.join(_PROJECT_ROOT, "cache/deploy_init_traj_xhand.npy")
    grasp_cache_path = None       # 非None: 暖启动时从此 grasp cache 随机抽一个grasp建两段式轨迹(更忠于训练, 每次reset不同)

    def __init__(self):
        with open(self.joint_meta_path) as f:
            meta = json.load(f)
        self.joint_names = meta["joint_names"]               # IsaacLab 顺序
        lo = np.array(meta["dof_lower"], dtype=np.float32)
        hi = np.array(meta["dof_upper"], dtype=np.float32)
        s = meta["dof_limits_scale"]
        self.dof_lower = lo * s                               # 已乘 0.9, IsaacLab 顺序; 目标裁剪(saturate)用
        self.dof_upper = hi * s
        self.dof_lower_raw = lo.copy()                        # 2026-06-17 修sim2real gap: 原始限位(未×0.9); obs 归一化用, 对齐训练(训练 obs 用 raw 归一化, xhand_wave_env:481)
        self.dof_upper_raw = hi.copy()
        assert self.dof_lower.shape == (21,), f"limits not 21: {self.dof_lower.shape}"

        # 2026-06-17 逐关节 PD (对齐训练v2 xhand_wave_env_cfg: kp=UniLab风格指尖软, kd=dampratio0.9; IsaacLab 顺序).
        #   SDK 范围 kp 0~2000 / kd 0~200, 以下值全在内. 默认用这套(self.kp/kd); --kp/--kd 覆盖成标量。
        def _pd(nm):
            if "thumb_mcp0" in nm: return 13.2, 0.358
            if "thumb_mcp1" in nm: return 6.95, 0.261
            if "thumb_pip"  in nm: return 4.76, 0.215
            if "thumb_dip"  in nm: return 0.9,  0.094
            if "ltspalm"    in nm: return 1.38, 0.116
            if "mcp0" in nm: return 6.62, 0.254
            if "mcp1" in nm: return 4.76, 0.216
            if "pip"  in nm: return 0.9,  0.094
            if "dip"  in nm: return 0.9,  0.094
            return 10.0, 0.3      # fallback (不该命中)
        _pds = [_pd(n) for n in self.joint_names]
        self.kp_per_joint = np.array([p[0] for p in _pds], dtype=np.float32)
        self.kd_per_joint = np.array([p[1] for p in _pds], dtype=np.float32)
        if self.kp is None:
            self.kp = self.kp_per_joint     # 默认逐关节; deploy_xhand 里 --kp 会覆盖成标量
        if self.kd is None:
            self.kd = self.kd_per_joint

        # friction comp: 各关节 breakaway 摩擦(IsaacLab序). json key(如 index_mcp0) 子串匹配 joint_names(如 right_hand_index_mcp0_joint)
        self.friction_breakaway = np.full(21, 0.14, dtype=np.float32)   # fallback = 实测均值
        if os.path.exists(self.friction_json_path):
            with open(self.friction_json_path) as f:
                _fric = json.load(f)
            for _i, _nm in enumerate(self.joint_names):
                for _k, _v in _fric.items():
                    if _k in _nm:
                        self.friction_breakaway[_i] = float(_v["breakaway_Nm"])
                        break
