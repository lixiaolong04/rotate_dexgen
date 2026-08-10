# XHand1 Pro 灵巧手 —— 手内旋转 Sim-to-Real 全流程

在 **NVIDIA Isaac Sim + IsaacLab** 中训练 Robotera **XHand1 Pro**（21 DOF 灵巧手）学会「手内物体旋转」（in-hand rotation），并 **sim2real 部署到真实硬件**。仿真里已达成任意朝向（全 SO(3)）旋转，真机已实现连续稳定旋转 2 分钟以上。

<p align="center">
  <img src="doc/截图/旋转.gif" width="30%" />
  <img src="doc/截图/image%20copy.png" width="63%" />
</p>

> 📖 **完整 RL DEMO 介绍**（团队学习 / 客户讲解用）：见 [docs/rl_demo_intro/RL_DEMO介绍.html](docs/rl_demo_intro/RL_DEMO介绍.html)，浏览器直接打开即可。

---

## 0. 这套流程是什么（30 秒版）

```
URDF/STL ──convert──▶ USD 资产
                        │
   gen_grasp（一次性，生成抓取初始姿态缓存）
                        │
   train teacher（PPO，能看到特权信息 priv_info）
                        │
   distill student（ProprioAdapt，盲策略，只看本体+触觉，TConv 时序网络）
                        │
   eval_batch（多条件量化挑权重：掉落率/转速/反转/抖动）
                        │
   deploy（真机 XHand1 Pro，EtherCAT，MIT 力矩模式 mode4）
```

- **teacher** 看得到物体位姿 / 摩擦等特权信息，学得快但真机拿不到这些。
- **student** 只吃真机也有的观测（关节角 + 目标角 + 指尖触觉），用时序卷积（TConv）从历史里回归出隐状态。**部署的是 student。**

---

## 1. 硬件 & 环境

### 1.1 训练机（本地 Blackwell 笔记本）

| 组件 | 版本 / 说明 |
|------|------|
| GPU | RTX 5070 Ti Laptop (Blackwell sm_120, 12GB) |
| NVIDIA Driver | **580.159.03**（⚠️ RTX 50 系 + 驱动 595 + Isaac Sim 5.x = 100% 崩溃，必须降到 580-open） |
| OS | Ubuntu 24.04 |
| conda env | `env_isaaclab`，Python 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| IsaacLab | v2.3.2（源码 clone 在 `~/IsaacLab`） |
| PyTorch | 2.7.0+cu128（支持 sm_120） |

### 1.2 云端大规模训练（可选）

百度 AIHC 4×L20 / IsaacSim 4.5 / IsaacLab 2.2 / Python 3.10。teacher 用 4 seed × 4 卡并行（无 DDP，各卡独立 seed），`num_envs=20480`。

### 1.3 安装踩坑

```bash
# IsaacLab ./isaaclab.sh --install 不会装 isaaclab 核心包，手动补：
pip install 'setuptools<81'                                    # 修 pkg_resources 缺失
pip install --no-build-isolation -e ~/IsaacLab/source/isaaclab
sudo apt install -y cmake build-essential                       # isaaclab.sh --install 需要

# 本仓库（editable）
conda activate env_isaaclab
cd era_dexhand_lab
pip install -e .
```

---

## 2. 资产：URDF → USD

真机厂商给的是 **URDF + STL 网格**（`assets/xhand_green/xhand_right/urdf/`）。Isaac Sim 的物理 / 渲染引擎吃的是 **USD**，所以要先转换（碰撞体、关节驱动、质量惯量都在转换时烘进 USD）。转换脚本：

```bash
python scripts/convert_xhand_urdf.py     # 版本自适应 IsaacLab 2.2/2.3 的 API 差异
```

转好的 USD 已随仓库提供（`assets/xhand_green/.../xhand_right.usd`），一般无需重跑。STL 网格也已入库，URDF 可直接加载。

> 手动调试可直接 `isaacsim` 打开 GUI，`File → Open` 选 `xhand_right.usd`。

---

## 3. 训练（headless）

```bash
conda activate env_isaaclab
cd era_dexhand_lab
export CUDA_VISIBLE_DEVICES=0
```

### 3.1 生成抓取初始姿态缓存（一次性，5–10 分钟）

抓取姿态决定每个 episode 手指从哪儿开始包住物体，是能不能转起来的关键。同一 scale 配置会覆盖旧缓存。

```bash
python rl_isaaclab/scripts/gen_grasp.py \
    --task Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0 --headless
```

缓存产物在 `cache/xhand_grasp_linspace_*.npy`（已随仓库提供 proven 版本，物理没变时可直接复用，不必重跑）。

### 3.2 训练 teacher（PPO）

```bash
# 本地 12GB 显存安全起步 num_envs=10；云端 L20 用 20480
python rl_isaaclab/scripts/train.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 10
```

### 3.3 蒸馏 student（部署用这个）

```bash
python rl_isaaclab/scripts/train.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless \
    --algorithm ProprioAdapt --load_path <stage1 teacher last.pth>
```

### 3.4 量化挑权重（eval_batch）

蒸馏 reward 高 ≠ 真机好。用 `eval_batch` 在多摩擦条件下量化 **掉落率 / 转速 / 每局圈数 / 反转率 / 动作抖动**，挑最鲁棒 + 最平滑的那个部署：

```bash
python rl_isaaclab/scripts/eval_batch.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --algorithm ProprioAdapt \
    --load_path ckpts/<训练目录>/
```

### 3.5 可视化策略

```bash
python rl_isaaclab/scripts/play.py \
    --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --so3 --num_envs 16 \
    --algorithm ProprioAdapt --load_path ckpts/<训练目录>/best.pth
```

---

## 4. 真机部署（XHand1 Pro）

真机用 **EtherCAT**（网口 `enp129s0`）通信，Docker 镜像 `xhand-deploy:22.04`。

### 4.1 核心：控制模式 mode4（MIT 力矩）

XHand1 Pro SDK 有两种关节控制模式（读 `.deb` C++ 头文件搞清的，见 `docs/xhand_control_manual.md`）：

- **mode 3**：三环位控，SDK **忽略 kp/kd**（早期 bring-up 一直没转就是踩这个坑）。
- **mode 4**：**MIT 力矩模式，kp/kd 生效**。力矩 = `kp·位置误差 + kd·速度误差 + 前馈`，与仿真里的显式力矩 PD **同公式、同单位**。

部署必须 `--mode 4 --kp 10 --kd 0.3 --speed 0`（speed = 目标速度，非 0 会瞬间猛动）。

### 4.2 部署命令（当前真机最佳：V1 mask 冠军 8103）

```bash
cd era_dexhand_lab
bash scripts/deploy_unilab_v1_student.sh 8103 --log v1_8103_real.npz
```

该脚本等价于：

```bash
python rl_isaaclab/deploy_xhand/deploy_xhand.py \
  --load_path ckpts/unilab_v1_students_final_20260630/student_8103_best.pth \
  --mode 4 --kp 10 --kd 0.3 --speed 0 \
  --action_scale_mul 1.0 --min-fingers 2 \
  --disable-tactile-ids 1,4          # ⚠️ 必须与训练对齐！见下
```

**部署键盘状态机**：

- `e`：先缓慢张开并标定触觉，再回放 `cache/deploy_init_traj_xhand.npy` 进行预抓取。
- `Enter`：抓取完成后启动旋转策略。
- `w`：冻结并保持当前姿态；再次按下继续策略，不会张开手。
- `q`：沿预抓取轨迹的逆序缓慢张开（约 6 秒），会释放物体。
- `t`：空载时重新标定触觉。
- `Ctrl-C` 或 `STOP`：先给全部关节发送 `mode=0`（idle/释放），再断开并退出。手会失去夹持力，物体下方必须有承接平台。

抓不牢可加 `--close-extra 0.4`。跑完使用
`python scripts/analyze_deploy_log.py <log>.npz` 复盘。

> ⚠️ **`--disable-tactile-ids` 必须与训练配置一致**，否则真机观测分布外（OOD）：
> - V1 训练时 mask 了食指(1)/小指(4)的触觉 → 部署 **必须** `--disable-tactile-ids 1,4`。
> - option C（前 pad 几何过滤）不 mask → 部署 **不加** 该参数（默认 `[]`）。

### 4.3 权重库（weight zoo）

| 权重 | 血缘 | 真机表现 | 部署要点 |
|------|------|----------|----------|
| **`ckpts/unilab_v1_students_final_20260630/student_8103_best.pth`** | V1 mask（食指/小指触觉清零对齐真机） | ⭐**全项目真机最佳**，360° 转 + 轻物体可倒置 | `--disable-tactile-ids 1,4` |
| `ckpts/frontpad_students_20260701/student_8500.pth` | option C 前 pad 几何过滤 | 仿真最优(19.91)，待真机 PK V1 | 不加 `--disable-tactile-ids` |
| `ckpts/overnight_pose5/student_9402.pth` | pose5 顺时针 | 真机能转，轻物体略往上爬 | 默认 |
| `ckpts/overnight_ccw/student_9503.pth` | 逆时针 CCW | 逆时针旋转 | 默认 |
| `ckpts/retrain_pose2/student_t701.pth` | pose2 retrain | proven，掉落 4.1% | 默认 |
| `ckpts/overnight_9000/student_9002.pth` | pose4b | 能转 | 默认 |
| `ckpts/unilab_kp10/student_7001.pth` | uniform kp=10 回退版 | 基线 | 默认 |
| `ckpts/overnight_pose6/student_9701.pth` | pose6 | 种子彩票 | 默认 |

能稳定旋转的物体范围：**Ø32–48mm × H26–38mm（中心 Ø40 最稳）**。

---

## 5. 关键坑（血泪总结）

1. **驱动**：RTX 50 系必须 NVIDIA 580，595 必崩。
2. **mode3 vs mode4**：位控模式 kp/kd 无效，真机再怎么调都不转 → 换 MIT 力矩模式 mode4。
3. **触觉对齐**：训练清零了哪几指触觉，部署就必须清零同样几指（`--disable-tactile-ids`），否则 OOD。
4. **选权重**：蒸馏 reward 高 ≠ 真机好；`best.pth` 常是低重力 / 后期 drift 假象，要在 -9.81 重力下用 `eval_batch` 评估掉落率 + 平滑度。
5. **种子彩票**：仿真几乎一样的兄弟种子，真机迁移可能天差地别 —— 仿真不预测真机迁移，多 seed 训 + eval 挑。
6. **摩擦**：`friction=False` 时摩擦走 USD 默认 0.5 ≠ 训练 base；真机指尖摩擦 0.08–0.10 是 UniLab 建模值的 10–25 倍，软 kp 驱不动 → 回退 uniform kp=10。
7. **散热**：真机连跑关节会到 86℃，`deploy --temp_limit 80` 自动停机。
8. **analyze 看不出转没转** —— 必须人眼看物体上画的线。


---

## 6. 目录导览

| 路径 | 内容 |
|------|------|
| `rl_isaaclab/tasks/inhand_rotate/` | 任务定义（`xhand_wave_env*.py` = 训练环境；`*_grasp_*` = 抓取生成） |
| `rl_isaaclab/scripts/` | `gen_grasp.py` / `train.py` / `play.py` / `eval_batch.py` |
| `rl_isaaclab/deploy_xhand/` | 真机部署栈（`deploy_xhand.py` 入口 + `deploy_cfg.py` + `deploy_env.py`） |
| `assets/xhand_green/` | XHand1 Pro 左右手 URDF + STL + 转好的 USD |
| `cache/` | 抓取姿态缓存 + 部署 init traj + 关节 / 摩擦标定 JSON |
| `ckpts/` | 各血缘的 teacher/student 权重（见 §4.3 权重库） |
| `scripts/` | 部署辅助、标定、诊断（`analyze_deploy_log.py` / `calib_fingertip.py` / `check_pad_contact.py` …） |
| `docs/rl_demo_intro/` | **RL DEMO 完整介绍**（HTML，团队学习 / 客户讲解用） |

---

## 7. 复刻这套流程（TL;DR）

```bash
# 1. 装环境（§1.3），驱动确认 580
# 2. 资产已随仓库（USD 已转好），可直接用
# 3. 抓取缓存已随仓库；要重生成：
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0 --headless
# 4. 训 teacher → 蒸 student → eval 挑权重（§3）
# 5. 真机部署 proven 冠军：
bash scripts/deploy_unilab_v1_student.sh 8103 --log v1_8103_real.npz
```

想直接上真机、跳过训练：仓库已带 §4.3 的全部权重，选一个 `--load_path` 即可（注意 §4.2 的触觉对齐）。

---

## 附录 A：最佳权重的训练参数

所有真机主力权重（**8103 / 9402 / 8500 / t701**）共享同一套「pose5 + uniform kp=10 + UniLab 对齐」基础配方，只在少数几处有差异。以下数值取自 `rl_isaaclab/tasks/inhand_rotate/xhand_wave_env_cfg.py` 与 `agents/ppo_cfg.yaml`。

### A.1 共享基础配方

**观测 / 动作 / 控制**

| 项 | 值 |
|---|---|
| action_space | 21（每关节目标）|
| observation_space | 141 =（21 关节角 + 21 目标角 + 5 指触觉法向力）× 3 帧 |
| 学生时序历史 `prop_hist_len` | 30 帧（TConv 输入）|
| `action_scale` | 1/24 |
| `torque_control` | True（显式力矩 PD，与真机 mode4 同公式）|
| kp / kd | **10 / 0.3**（uniform，armature 0.003，joint friction 0.1）|
| `decimation` / sim dt | 12 / (1/240) → 控制频率 **20 Hz** |
| episode | 20 s（≈400 控制步）|
| clip_obs / clip_actions | 5.0 / 1.0 |

**物体 & 任务**

| 项 | 值 |
|---|---|
| 物体 | 圆柱，质量 0.10 kg |
| `scale_range` | [0.4, 0.7, 10] → Ø32–56 mm（10 档）|
| 重力 | 满 **−9.81 从 step0**（无幅值课程 `gravity_curriculum=False`）|
| 朝向 | `reset_random_quat=True`，**全 SO(3)**（Shoemake 均匀）|
| `rot_axis` | (0,0,1)，angvel clip ±0.5 |
| `drop_dist_thresh` | 0.025 m（手帧判定，一离稳定区即终止）|

**Reward 权重**

| 项 | 值 |
|---|---|
| rotate_reward | **+2.5** |
| object_linvel_penalty | −0.3 |
| pos_diff_penalty | −0.4 |
| torque_penalty | −0.1 |
| work_penalty | −0.5 |
| object_pos_reward | +0.003 |

**域随机化（DR，全开）**

| 项 | 范围 |
|---|---|
| PD 增益 | p_gain ×[0.5, 2.0]，d_gain ×[0.5, 2.0] |
| 摩擦系数 | ×[0.5, 2.0]；base 弹性体 1.6 / 金属 0.2 / 物体 1.0 |
| 质心 COM | [−0.01, 0.01] m |
| 物体质量 | [0.01, 0.25] kg |
| 随机外力 | force_scale 2，prob 0.25，decay 0.9 / 0.08 s |
| 关节角噪声 | 0.02 |
| 触觉 | contact_smooth 0.5，latency 0.005 s |
| 关节限位 | `dof_limits_scale` 0.9（对齐真机）|

**PPO（teacher）— `ppo_cfg.yaml`**

| 项 | 值 |
|---|---|
| 策略网络 MLP | [512, 256, 128] |
| priv_mlp（特权编码）| [256, 128, 9] |
| num_envs | 20480（云端 L20）/ 本地 10 |
| γ / GAE λ | 0.99 / 0.95 |
| learning_rate | 1e-3 |
| kl_threshold | 0.02 |
| horizon / minibatch / mini_epochs | 8 / 32768 / 5 |
| critic_coef / entropy_coef / e_clip | 8（有效 4.0）/ 0.01 / 0.2 |
| grad_norm | 1.0 |
| max_agent_steps | 300 M（实际 ~300–500M 到顶，看 plateau 停）|
| priv_info_dim | 9 = [0:3]物体位移 [3]摩擦 [4]质量 [5:8]COM [8]物体scale |

**蒸馏（student, ProprioAdapt）**：teacher 冻结，student 用 **TConv** 从 30 帧本体 + 触觉历史回归 9 维特权隐状态，只吃真机可得观测。蒸馏约 480 M steps（手动看 plateau 停）。**部署的就是这个 student。**

### A.2 各最佳权重的差异（只有这几处不同）

| 权重 | 抓取姿态 | `disable_tactile_ids` | `frontpad_filter` | 旋向 | 备注 |
|---|---|---|---|---|---|
| **8103**（V1，真机冠军）| pose5 | **[1,4]**（清零食指/小指触觉）| False | CW | 部署必须 `--disable-tactile-ids 1,4` |
| 9402（CW 冠军）| pose5 | [] | False | CW | 纯 141 obs |
| 8500（option C，仿真最优）| pose5 | [] | **True** | CW | 前 pad 几何门控，法向阈值 0.014 |
| 9503（CCW）| pose5 | [] | False | **CCW** | rot_axis 反向 |
| t701（早期 proven）| pose2 | [] | False | CW | 掉落 4.1% |

> option C 的 `frontpad_normal`（每指精确 3D 法向，dip_link 局部帧）：拇指 (0, 0.917, 0.399)，其余四指 (0.399, 0, 0.917)；投影阈值 0.014 m。

---

## 附录 B：XHand 关节命名表（21 DOF，右手）

| Joint 名称 | 手指 | 位置 | 含义 |
|-----------|------|------|------|
| `right_index_MCP_FE` | 食指 | 掌指关节 | 指根弯曲/伸直 |
| `right_index_MCP_AA` | 食指 | 掌指关节 | 指根外展/内收 |
| `right_index_PIP` | 食指 | 近端指间 | 中间关节 |
| `right_index_DIP` | 食指 | 远端指间 | 指尖前关节 |
| `right_middle_*` (FE/AA/PIP/DIP) | 中指 | 同上 | 4 DOF |
| `right_ring_*` (FE/AA/PIP/DIP) | 无名指 | 同上 | 4 DOF |
| `right_pinky_CMC` | 小指 | 腕掌关节 | 小指根更靠掌部 |
| `right_pinky_*` (MCP_FE/MCP_AA/PIP/DIP) | 小指 | 同食指 | 4 DOF |
| `right_thumb_CMC_FE` / `right_thumb_CMC_AA` | 拇指 | 腕掌关节 | 根部 |
| `right_thumb_MCP_FE` / `right_thumb_MCP_AA` | 拇指 | 掌指关节 | 中段 |
| `right_thumb_IP` | 拇指 | 指间关节 | 末端 |

*本 README 记录的是一个真实 sim-to-real 项目从环境搭建到 2 分钟稳定旋转的完整路径。完整技术讲解见 `docs/rl_demo_intro/`。*

---

## DexGen：基于 RL 轨迹的生成式灵巧手控制

本仓库除上文的 RL teacher/student 流程外，还包含完整的 **DexGen** 系统。DexGen
使用冻结的 CW/CCW RL 专家采集轨迹，学习“状态 → 未来关键点运动 → 关节目标”的
生成式控制器，并支持 RL guidance、MANUS 手套遥操以及 XHand 真机部署。

```text
RL 专家轨迹 / MANUS command
            ↓
Guided Diffusion：生成未来两帧的 10 个手部关键点运动
            ↓
IDM：预测 21 维绝对 PD target
            ↓
Isaac Sim XHand / 真实 XHand
```

DexGen 的主要入口位于：

| 路径 | 内容 |
| --- | --- |
| `dexgen/commands/data/` | RL 轨迹采集和数据审计 |
| `dexgen/commands/train/` | Diffusion、IDM 及实验分支训练 |
| `dexgen/commands/eval/` | 离线评估、仿真闭环和 guided 推理 |
| `manus_dexgen_bridge/` | MANUS 输入、手势标定和遥操仿真 |
| `dexgen/deploy/` | XHand 真机 DexGen/MANUS 部署 |

当前正式推荐使用 active-recovery v3：

```text
Diffusion: dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM:       dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
```

交付仓库只保留上述正式 v3 模型，便于接手人安装后立即进行第一次仿真测试；历史
DexGen 对照模型、RL 轨迹数据和训练/评估日志不随仓库交付。需要继续训练时，使用
仓库内保留的 CW/CCW RL 专家重新采集数据：CW 为
`ckpts/overnight_pose5/student_9402.pth`，CCW 为
`ckpts/overnight_ccw/student_9503.pth`。具体采集和审计流程见
[`dexgen/data/README.md`](dexgen/data/README.md) 以及交接手册第 4 节。

首次接手请按以下顺序阅读：

1. **[DexGen 项目交接与快速上手手册](dexgen/HANDOVER.md)**：环境安装、云端训练、
   RL 数据采集、IDM/Diffusion 训练、仿真推理、MANUS/XHand 部署及安全检查；
2. **[DexGen 运行手册](dexgen/README.md)**：日常启动命令、参数和交互按键；
3. **[DexGen 开发与训练流程](dexgen/docs/DEVELOPMENT.md)**：数据契约、模型结构和实验验收；
4. **[MANUS Bridge 快速启动](manus_dexgen_bridge/README.md)**；
5. **[XHand 真机运行说明](dexgen/deploy/README.md)**。

最短仿真验证命令：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab

python dexgen/commands/eval/eval_dexgen.py \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --mode cw --num_envs 16 --steps 1200 --ddim_steps 25 \
  --sensor_profile raw --so3 --device cuda:0
```

项目中的 `IDM` 是 learned inverse dynamics model，内部交流有时称为“逆运动学”，
但它并非传统解析 IK。新实验不要覆盖上述正式 checkpoint；数据、Diffusion、IDM
以及触觉 profile 必须保持同一训练契约。
