# Xhand 灵巧手 Sim2Real 部署 —— 完整讲解

> 作者：hu5 + Claude（Opus 4.8）  日期：2026-06-04
> 范围：把仿真训好的手内旋转策略（ProprioAdapt student **t603**）部署到真实 **Xhand 1 Pro 右手（21 DOF）** 的全部工作：框架、流程、改了什么、为什么、原理。
> 配套文档：设计 `docs/superpowers/specs/2026-06-04-...-design.md`；计划 `docs/superpowers/plans/2026-06-04-...md`（含 AS-BUILT 修订）。

---

## 0. 一页速览（TL;DR）

- **目标**：仿真里训好的"手内绕 z 轴转圆柱"策略，搬到真手上跑。
- **部署权重**：`ckpts/scale_1000M/student_t603.pth`（ProprioAdapt student，只用本体感觉+触觉，无特权信息，真机可拿全）。
- **核心决策**：新建**独立轻量部署包** `rl_isaaclab/deploy_xhand/`，**完全脱离 Isaac Sim**，跑在 **Python 3.10 + Docker(Ubuntu 22.04)**，**CPU 推理**（实测 0.14ms ≪ 20Hz 预算）。
- **部署 env 做法**：**逐行照搬** SharpaWave 现成部署代码 `sharpa_wave_deploy_env.py`（approach A），**只替换属于 Xhand 的部分**（SDK、21 DOF、关节/触觉映射）。
- **最大的工程量**：把照搬来的 **SharpaWave SDK 调用全部改成真实 Xhand SDK 接口**（容器 introspect 验证），因为两套 SDK 名字/类型/返回完全不同。
- **抓到并修的关键坑**：失重蒸馏（训练侧）、CPU 加载崩、SharpaWave 接口在 Xhand 全不存在、触觉模型完全不同、speed=0 不动、pynput/evdev/X11。
- **待真机**：关节映射、触觉 sensor 布局、几个 SDK 行为值（mode/speed 语义/使能），都已标 `[B1]/[B3]` 并配了发现脚本。

---

## 1. 全局：训练 → 部署的 sim2real 闭环

```
仿真训练(Isaac Sim)                              真机部署(无 Isaac)
┌─────────────────────────────┐                ┌──────────────────────────┐
│ Stage1 PPO teacher          │                │ deploy_xhand 包          │
│   用 priv_info(物体真值8维)  │  蒸馏          │   只用 obs+本体感觉历史  │
│   学会旋转                   │ ──────────────▶│   加载 t603 跑前向       │
│ Stage2 ProprioAdapt student │  (t603)        │   → 21 关节位置指令      │
│   从本体感觉历史预测priv_info │                │   → Xhand SDK → 真手     │
└─────────────────────────────┘                └──────────────────────────┘
```

**为什么要蒸馏（teacher→student）**：teacher 用了"特权信息"priv_info（物体的位姿/质量/摩擦/质心等仿真真值），真机拿不到这些。student（ProprioAdapt）只用**本体感觉历史**（关节位置/目标的过去 30 帧）经一个 TConv 适配器**预测出** priv_info 的等价物，所以真机只靠关节读数+触觉就能跑。**t603 就是这个 student，是真正部署的策略。**

**sim2real gap 的来源**（部署要对付的）：仿真和真机在 ① 关节顺序/标定 ② 触觉传感器原理 ③ PD 增益量纲 ④ 摩擦 ⑤ 控制时序 上都有差异。部署代码的任务就是把策略和真手"对齐"，并把差异控制在策略训练时见过的域随机化范围内。

---

## 2. 训练框架与流程（仿真侧，理解部署的前提）

入口 `rl_isaaclab/scripts/train.py`。

### 2.1 怎么连 IsaacLab —— 先启动，再导入（头号规则）
```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)   # ← 这行才真正拉起 Omniverse/Isaac Sim
simulation_app = app_launcher.app
"""Rest everything follows."""
import gymnasium as gym                  # 之后才能 import isaaclab.* / 你的 env
```
**原理**：IsaacLab 所有模块依赖 Omniverse 内核，必须等 app 起来后才能 import。顺序写反直接崩。**这也是部署能脱离 Isaac 的原因**——deploy 根本不 import 这些。

### 2.2 你的任务怎么挂到 IsaacLab —— DirectRLEnv 钩子
`gym.make("Isaac-Inhand-Rotate-Xhand-Wave-v0")` 建 `XhandWaveInhandRotateEnv`（继承 IsaacLab `DirectRLEnv`）。你不写仿真循环，只**重写钩子**，IsaacLab 在物理循环里回调：

| 钩子（`xhand_wave_env.py`） | 作用 |
|---|---|
| `_setup_scene` | 建手(Articulation)+物体(RigidObject)+接触传感器+地面 |
| `_pre_physics_step(action)` | action → 目标 `cur_targets`（+随机力 DR） |
| `_apply_action` | 力矩 `kp*(target-pos)-kd*vel` 写进 PhysX（每物理子步） |
| `_get_observations` | 组 obs（62×3 + priv_info + proprio_hist） |
| `_get_rewards` | rotate/penalty 等 reward |
| `_get_dones` | 判掉落/超时 + **重力 curriculum 爬升** |
| `_reset_idx` | 从 grasp cache 摆姿态 + 域随机化 |

`decimation=12`：1 控制步 = 12 物理子步。

### 2.3 参数怎么加载 —— 两份配置 + 命令行覆盖
- **env_cfg = `XhandWaveEnvCfg` 类**（`xhand_wave_env_cfg.py`）：仿真/机器人/reward/域随机化全部参数。
- **agent_cfg = `ppo_cfg.yaml`**：PPO 超参（lr/horizon/网络层数…）。
- `@hydra_task_config(task,"agent_cfg_entry_point")` 从 gym 注册表取这两份并允许 Hydra CLI 覆盖；`main()` 再用 argparse 参数覆盖关键字段（num_envs/seed/device/algo/load_path…）。
- `ConfigWrapper(agent_cfg, env_cfg)` 打包成 `.train`/`.task` 传给算法。
- 每次 run 把 `env_cfg.py`+`ppo_cfg.yaml` **拷进 log 目录**，自带配置存档。

### 2.4 env 包装器
`GymStyleEnvWrapper`（`wrapper/sharpa_wave_env_wrapper.py`）把 DirectRLEnv 的 gym 五元组 `step` 适配成算法要的 `obs_dict["obs"]`+`dones`，clip action，构造时主动 `reset()` 一次。

### 2.5 PPO 训练循环（`algo/ppo/ppo.py`）
`__init__`：建 `ActorCritic` + `RunningMeanStd`(obs/value 归一化) + Adam + `ExperienceBuffer`。
`train()` 反复：`play_steps()` 采样 horizon=8 步 → 算 GAE return/advantage → `train_epoch()` 做 5 轮 minibatch（PPO 截断损失+value+entropy，自适应 KL 调 lr）→ 存 ckpt(`best`/定期/`last` 到 `stage1_nn/`)。ckpt = `{model, running_mean_std, value_mean_std}`。

### 2.6 两阶段
- **Stage1 PPO teacher**：用 priv_info(8 维)学旋转。
- **Stage2 ProprioAdapt student**（`algo/padapt/padapt.py`）：`restore_train` 载 teacher，**只训 TConv 适配器**让它从 proprio_hist 回归出 priv_info(`(e-e_gt)²`)，其余冻结。ckpt 多存 `sa_mean_std`。→ t603。

### 2.7 obs/action 数学（部署必须逐位一致）
```
每帧 62 = unscale(dof_pos,21) + cur_targets(21) + contacts(5) + contact_pos(15=0)
obs = 末 3 帧拼接 = 186 ;  proprio_hist = 末 30 帧 = (30,62)
action(21) ∈ [-1,1] → cur_targets = prev_targets + (1/24)*action, 再 saturate 到关节限位×0.9
unscale(x,lo,hi) = (2x-hi-lo)/(hi-lo)
控制 20Hz (= 240/decimation)
```

---

## 3. 部署框架与流程（真机侧，本次为 sim2real 新建）

### 3.1 总体决策
| 决策 | 内容 | 为什么 |
|---|---|---|
| 方案 A | 新独立包 `deploy_xhand/`，复用 `algo.padapt.ProprioAdapt` 加载权重 | 复用训练的归一化/网络加载，避免引入"加载 bug" |
| 脱离 Isaac | 部署不 import isaaclab | 推理只需 torch 前向 + SDK 发指令，不需要仿真器 |
| Python 3.10 | 独立 conda env / Docker | `hand_controller` whl 是 cp310 |
| Docker(22.04) | 容器跑 | Xhand SDK 是 jammy 编的，本机 24.04 装不上（libfmt8） |
| CPU torch | 不用 GPU | 策略 0.39M 参数、CPU 推理 0.14ms ≪ 50ms/步；真机板载常无 N 卡 |
| 忠实照搬 sharpa | deploy_env 逐行抄 `sharpa_wave_deploy_env.py` | 用真机验证过的成熟逻辑，少踩自创坑 |

### 3.2 架构四单元 + 数据流
```
deploy_xhand.py (entry: 20Hz 主循环 + 急停)
   ├── PolicyRunner      复用 ProprioAdapt 加载 t603 + 推理
   └── XhandInhandRotateDeployEnv   纯计算, 照搬 sharpa 的 obs/action 数学 (无 isaac/无硬件)
          └── XhandHWInterface / FakeHWInterface   唯一碰 SDK 的薄层(可换真/假手)
```
一个控制步（20Hz）：
```
真手 --SDK get_joint_state_list--> 21角(Xhand序) --joint_map--> 21角(IsaacLab序)
真手 --SDK get_sensor_state_list--> taxel→nf力 --tactile_map+force_scale+噪声门--> 5指力
   → XhandDeployEnv 组 obs(186)+proprio_hist(30,62)
   → PolicyRunner: running_mean_std + sa_mean_std → act_inference → clamp[-1,1]
   → cur_targets = prev + (1/24)*action, saturate → joint_map逆 → Green_JointCommand_User → set_joint_cmd_list
```

### 3.3 逐模块讲解（`rl_isaaclab/deploy_xhand/`）

**`deploy_cfg.py`** — 部署配置。忠实照搬 sharpa cfg 字段（22→21），外加 Xhand 通信/PD/安全项。
- 不可改常量（来自训练 cfg）：action_space=21、observation_space=186、prop_hist_len=30、action_scale=1/24、control_freq=20、dof_limits_scale=0.9。
- 硬件待标定占位：kp/kd、cmd_speed/cmd_torque、joint_cmd_mode、force_scale、contact_threshold。
- 安全：action_scale_mul（S2 设 0.3 限幅起步）、current/speed_coef 保守。
- `__init__` 从 `cache/xhand_isaaclab_joint_meta.json` 读 21 关节顺序+限位（×0.9）。**路径锚定项目根**（不依赖 cwd）。

**`joint_map.py`** — IsaacLab 关节顺序 ↔ Xhand SDK 关节 id 的 21 维映射。
- `isaaclab_to_xhand`(scatter) / `xhand_to_isaaclab`(gather) 互逆。
- 现为**恒等占位**，B2 用 `inspect_xhand_joints.py` 实测真实排列。
- 有"非恒等语义"测试锁死方向（防有人改反）。

**`tactile_map.py`** — 触觉 sensor 布局。
- `FINGER_SENSOR_LAYOUT` = 5 指 ×(tip,pad1,pad2) 的 sensor_id（占位，B3 实测）。
- **原理**：Xhand 每指 3 块触觉 pad，不是单点。

**`hw_interface.py`** — 唯一碰 `hand_controller` SDK 的单元（"被替换的 Xhand 部分"）。**已按真实 Xhand SDK 重写**（容器 introspect 验证）：
- 关节读：`get_joint_state_list(bool)→(ErrorCode,str,list[Green_JointState_User])`，取 `.position` → `xhand_to_isaaclab`。
- 关节写：`Green_JointCommand_User`(属性 position/kp/kd/speed/torque/mode) + `set_joint_cmd_list`。
- 触觉：`get_sensor_state_list`→`Green_Sensor_User`，每指 tip/pad1/pad2 的法向力 `first_ydds_nf+second_ydds_nf` 聚合。
- 通信类 pybind 名**小写开头** `serialCommunication`/`etherCatCommunication`；清零 `sensor_clear`；断连 `comm.close()`。
- 标 `[B1]/[B3]` 的是 introspect 给不了的"行为/数值"，待真机。

**`fake_hw.py`** — 假手桩，接口与 XhandHWInterface 完全一致。让整条链路**没真手也能 dry-run + 跑 14 个离线测试**（approach A 的核心好处）。

**`deploy_env.py`（`XhandInhandRotateDeployEnv`）** — **逐行照搬** `sharpa_wave_deploy_env.py`（22→21，槽位 44→42/49→47/64→62）：
- `compute_observations`（滑窗 80 帧、obs=末3帧、proprio=末30帧、reset 填满历史）。
- `get_tactile_info`（force_scale → **噪声门**`force<contact_threshold→0` → binary 才二值化）。
- `step`（键盘状态机 e=启动/w=冻结/q=回零/t=触觉标定）+ 20Hz 节拍。
- `_reset_idx`（暖启动回放抓取轨迹到指尖接触）。
- SDK 调用全走 `self.hw`（薄层），env 全程 IsaacLab 顺序。

**`policy_runner.py`** — 复用 `ProprioAdapt` 加载 t603 + 推理。
- **CPU-safe 加载**：`torch.load(ckpt, map_location=device)`（t603 存于 CUDA，部署 CPU torch 不带 map_location 会崩）。
- `act`：`running_mean_std(obs)` + `sa_mean_std(proprio_hist)` → `model.act_inference` → clamp[-1,1]。镜像训练 `padapt.test()` 推理路径。

**`deploy_xhand.py`** — entry。组装 cfg→hw→env→runner，20Hz 主循环（节拍在 env.step 内），Ctrl-C 急停。`--dry_run` 用 FakeHW 离线；`--no_keyboard` 自动跑；`--action_scale_mul` 分级；`--kp/--kd` 覆盖。

### 3.4 Docker（`deploy_docker/`）
借鉴 SharpaWave 现成的 `rl_isaaclab/utils/{Dockerfile,docker-compose.yml}`：
- `Dockerfile`：ubuntu:22.04 + Xhand `.deb`(系统 SDK) + `hand_controller`(Python 绑定) + **CPU torch** + pynput(借其 libevdev-dev 编译依赖装上)。
- `docker-compose.yml`：host 网络 + privileged(透串口) + **X11 挂载(键盘用)** + 挂仓库；**去掉** SharpaWave 的 nvidia runtime（Xhand 触觉非 GPU 视觉）。
- `build.sh`(规整带空格/(1) 的 SDK 文件名再构建) / `run.sh`(快速进容器，自动跳过不存在的串口) / `README.md`。

### 3.5 真机发现工具（`scripts/`，introspect 给不了行为，需真手）
- `inspect_xhand_sdk.py`（B1，容器免真手就能跑）：打印真实 SDK 接口名/字段/签名，核对回填 `[B1]`。
- `inspect_xhand_joints.py`（B2）：逐个 Xhand 关节发小幅指令，观察哪指动 → 填 `joint_map`。
- `inspect_xhand_tactile.py`（B3）：同时打印低层 nf/tf + 高层 HapticModel 每指力 → 填 `FINGER_SENSOR_LAYOUT`、选 API、标 `force_scale`。

---

## 4. 改动清单：改了什么 + 为什么 + 原理

### 4.1 新建独立部署包，脱离 Isaac Sim
- **改**：新建 `rl_isaaclab/deploy_xhand/`（8 个模块）+ 测试 + Docker + 脚本。
- **为什么**：现有 `deploy.py`/`sharpa_wave_deploy_env.py` 是 SharpaWave 22 DOF + 另一套 SDK，不能直接用；且推理不需要 Isaac。
- **原理**：`ProprioAdapt` 是纯 torch（只依赖 models/running_mean_std/tensorboardX），可在 py3.10 脱离 isaac 构造+加载+推理；env 做成普通对象暴露 `observation_space/action_space/prop_hist_len/num_obs` 供其读取。

### 4.2 忠实照搬 sharpa（approach A）+ 纠回两处自创偏差
- **改**：`deploy_env.py` 从"我自创的纯计算复刻训练"改成"逐行照搬 sharpa 部署 env"；SDK 走 hw_interface 薄层（保留离线测试）。
- **纠偏 1 触觉噪声门**：早先我以"对齐训练连续分支无阈值"为由删了噪声门——**错**。sharpa 部署故意保留噪声门（真机噪声清零≈仿真干净 0，反而更贴训练分布），已恢复。
- **纠偏 2 自创 EMA**：我曾加跨步 `contact_smooth` EMA，sharpa 没有、训练的 contact_smooth 是单步内子帧平滑（非跨步），已删。
- **原理/教训**：部署要尽量复刻"被真机验证过的"逻辑，少凭"看起来更对"的推理改动。

### 4.3 真 Xhand SDK 接口替换（最大工程量）
- **改**：`hw_interface.py` 的所有 SDK 调用从 SharpaWave 名字换成真实 Xhand 名字。
- **为什么**：照搬 sharpa 带来的 SDK 调用在 Xhand 上**全不存在**，真机必崩。
- **对照**：
  | SharpaWave（照搬来的，错） | 真实 Xhand |
  |---|---|
  | `get_joint_state_user()` | `get_joint_state_list(bool)→元组`，取 `.position` |
  | `JointCommand_User(...)` 位置参数 | `Green_JointCommand_User()` 属性式 + `set_joint_cmd_list` |
  | `get_sensor_state_user()`、6 轴 F6 `norm(f6[:3])` | `get_sensor_state_list`→`Green_Sensor_User`，taxel 推算的 nf/tf |
  | `SerialCommunication`(大写) | `serialCommunication`(小写) |
  | `set_control_mode/current_coeff/speed_coeff/stop/calib_tactile` | **不存在**；模式/速度在每条指令；清零 `sensor_clear`；断连 `comm.close` |
- **原理**：C++ 符号大写，但 pybind 暴露的 Python 名常不同——**必须容器 `import hand_controller` introspect 核对**，不能凭 C++ 文档猜。

### 4.4 CPU 加载修复
- **改**：`policy_runner` 用 `torch.load(ckpt, map_location=device)`。
- **为什么**：t603 在 CUDA 上保存，部署用 CPU-only torch，不带 map_location 会抛 "deserialize on CUDA but cuda not available"。
- **怎么发现的**：Docker(CPU) 实跑逼出来的（笔记本有 GPU 没暴露）。

### 4.5 cmd_speed=0 风险
- **改**：`set_joint_targets` 显式设 `Green_JointCommand_User.speed/torque`（cmd_speed=350 占位），加 cfg 项。
- **为什么**：实测 `Green_JointCommand_User` 默认 speed=0，伺服里 speed=0 常=速度上限 0=**发了位置指令也不动**。`[B1]` 真机验单位语义。

### 4.6 Docker + pynput/X11（借鉴 sharpa）
- **改**：Dockerfile 借 sharpa 做法装 `libevdev-dev`+编译依赖后 pip pynput；新增 docker-compose（X11 挂载支持键盘）。
- **为什么**：pynput 依赖 evdev 需编译；容器抓键需 X11。`--no_keyboard` 是无 X 时的退路；键盘 import 失败给清晰报错（不神秘崩）。

### 4.7 训练侧 sim2real 前提（让 t603 可部署，早前会话已修，列作上下文）
- **重力失重蒸馏修复**：`train.py` ProprioAdapt 时显式 `sim.gravity=(0,0,-9.81)`。原 sharpa 只关 curriculum 没设重力 → 蒸馏停在 cfg 默认 -0.05 近失重（潜伏 bug）。t603 是修复后蒸的，真重力货。
- **摩擦随机化 + scale 泛化 + friction_mul 扫描**：让策略对真机未知摩擦/尺寸鲁棒（见训练里程碑记忆）。

---

## 5. 触觉深入：sim 与真机不是一回事

| | 仿真（训练用） | 真机（Xhand） |
|---|---|---|
| 来源 | PhysX 指尖-物体**接触力矢量**，取**模长 ‖F‖** | taxel 阵列（每指 3 pad），SDK 推算 |
| 每指给策略的 | 1 个标量（接触力大小，≈牛顿） | 同样要给 1 个标量/指 |
| 单位 | 牛顿（物理真值） | 传感器单位 → **需 force_scale 标定** |
| 覆盖 | 整个 dip_link | 只有 pad 贴的区域，碰 pad 外读 0 |

**SDK 有直接返回力的 API（两层，容器实测确认）**：
1. 低层：`get_sensor_state_list`→`Green_Sensor_User.first_ydds_nf/second_ydds_nf`（法向力）+ `_tf`（切向力）+ `channelData`(11 通道原始阵列)。**SDK 已把阵列算成力，直接拿**（int 单位）。
2. 高层：`HapticModel.compute_all_fingers([5×Haptic_FingerSensors{tip,pad1,pad2}])→[5 float]`（官方每指综合力，需 `Haptic_Config` 标定）。

**当前实现**：用低层 nf 聚合（已是"拿 SDK 算好的力"）；HapticModel 是 B3 升级项。
**为什么这是头号 sim2real 风险**：真机触觉是 taxel 推算的法向力、和仿真的物理接触力**同类不同源**，量纲/覆盖/分布都有差。缓解：force_scale 标定 + 噪声门 + **策略以本体感觉为主、触觉为辅**（ProprioAdapt 特性，部分吸收触觉误差）。

---

## 6. 待真机回填（Phase B 占位清单）

| 占位 | 文件 | 真机定（工具） |
|---|---|---|
| `ISAACLAB2XHAND_IDX` 恒等 | joint_map.py | B2 `inspect_xhand_joints.py` |
| `FINGER_SENSOR_LAYOUT` | tactile_map.py | B3 `inspect_xhand_tactile.py` |
| `joint_cmd_mode` | deploy_cfg.py | B1 位置控制 mode 真值 |
| `cmd_speed/torque` | deploy_cfg.py | B1（speed=0 是否=不动） |
| `kp/kd` | deploy_cfg.py | B4 系统辨识 |
| `force_scale/contact_threshold` | deploy_cfg.py | B3 触觉标定 |
| `get_*_list(bool)` flush 语义/返回是否 21 按序 | hw_interface.py | B1 |
| `connect()` 是否需 `write_joint_start` 使能 | hw_interface.py | B1 |

**Phase B 入口（Docker）**：
```bash
cd deploy_docker && ./build.sh                 # 一次性
xhost +local:root && docker compose -f deploy_docker/docker-compose.yml up -d
docker exec -it xhand_deploy bash
python scripts/inspect_xhand_sdk.py            # B1 核对接口
python scripts/inspect_xhand_joints.py -d /dev/ttyUSB0    # B2 关节映射
python scripts/inspect_xhand_tactile.py -d /dev/ttyUSB0   # B3 触觉+布局+force_scale
python -m rl_isaaclab.deploy_xhand.deploy_xhand --action_scale_mul 0.3   # S2 低增益限幅起步
```
安全分级 S0→S4：回零→暖启动抓取→低增益+0.3 限幅→正常缩放→升增益持续旋转 ≥30s。

---

## 7. 原理与教训总结

1. **脱耦推理与仿真器**：策略推理只是 torch 前向，把它从 Isaac Sim 里剥出来，部署才轻、才可移植（CPU/任意板载机）。
2. **复刻 > 重写**：部署逻辑尽量逐行照搬真机验证过的代码（sharpa），只换硬件相关部分；自创"看起来更对"的改动容易引入静默 bug（我们就踩了噪声门/EMA）。
3. **SDK 必须 introspect 核对**：C++ 文档/符号 ≠ pybind Python 名。容器能 import SDK 时，提前把接口挖清楚，是真手到货前最值钱的事。
4. **离线可测性**：用 FakeHW 薄层让整条链路没真手也能 dry-run + 单元测试，90% 的问题在桌面就能排掉。
5. **先验证再信**：CPU 加载崩、speed=0、通信类名、失重蒸馏——都是"不验证就发现不了"的静默坑；坚持"在真实条件下评估/实跑"才抓得到。
6. **诚实标注未知**：introspect 给不了"行为/数值"的，老实标 `[B1]/[B3]` + 配发现工具，不假装"全对了"。

---

*配套：完整改动见 git `dev_evan` 分支 `deploy_xhand/` + `deploy_docker/` + `scripts/inspect_xhand_*`；记忆 `xhand-sim2real-deploy`。*
