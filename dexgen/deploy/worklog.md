# XHand + DexGen 真机部署工作日志

## 2026-07-28：SDK、强化学习策略与 MANUS/DexGen 真机链路

### 今日目标

- 安装并验证 `/home/user/rotate_dexgen/xhandSDK` 厂商 SDK。
- 使用 EtherCAT 网口 `enp129s0` 连接已独立供电的 XHand Green。
- 先跑通已有强化学习旋转策略，确认真机控制参数。
- 在不修改仿真 MANUS 脚本的前提下，新建真机 sim2real 部署程序。
- 真机程序需要支持预抓取、纯 MANUS 遥操、DexGen 接管、方向切换、暂停、
  缓慢松手以及退出卸力。

## 当前结论

采用“Docker 管硬件、本机管 CUDA 推理”的两进程方案：

```text
MANUS/RetargetX TCP 8020
          |
          v
本机 env_isaaclab (Python 3.11 + CUDA)
MANUS + RL command + Diffusion + IDM
          |
          | localhost TCP 8031
          v
Docker xhand-deploy:22.04 (Python 3.10)
厂商 SDK + EtherCAT enp129s0 + XHand
```

这样设计的原因：

- 厂商 SDK wheel 是 Python 3.10 ABI，并依赖 `libfmt.so.8`。
- 本机 `env_isaaclab` 是 Python 3.11，运行 PyTorch/CUDA/DexGen。
- SDK 留在已验证的 Docker 环境最稳定，也不会影响后续 MANUS/DexGen 推理。
- `8031` 只监听 `127.0.0.1`，不会把硬件接口暴露到局域网。

## 已完成工作

### 1. 厂商 SDK 和 Docker

- SDK 目录：`/home/user/rotate_dexgen/xhandSDK`
- Docker 镜像：`xhand-deploy:22.04`
- 已完成 SDK 安装、构建和 matplotlib 补全。
- SDK 测试结果：20 项通过，1 项跳过。
- EtherCAT 已实机进入 OP 状态，并成功识别：

```text
Found device id 0 at slave 1
Ethercat INIT SUCCESS
```

- SDK 高频 `get_joint_state` info 日志已默认抑制，连接和异常日志仍保留。

### 2. 已验证的原始真机强化学习策略

启动文档：

```text
dexgen/deploy/README_XHAND_HARDWARE.md
```

已成功运行 `student_8103_best.pth`，固定参数如下，后续不要随意修改：

```text
mode=4
kp=10
kd=0.3
speed=0
torque=0
action_scale_mul=1.0
min_fingers=2
disable_tactile_ids=1,4
```

原始策略按键已经整理为：

- `e`：缓慢预抓取。
- `Enter`：启动策略。
- `w`：冻结/继续。
- `q`：沿预抓取反向轨迹缓慢张开。
- `Ctrl-C`：发送 mode 0，再关闭 EtherCAT。

此前危险行为已经修复：`w` 和 `q` 不再让手瞬间完全张开，退出后舵机不会
继续处于上电僵硬状态。

### 3. 新增 MANUS + DexGen 真机部署

新增文件：

```text
dexgen/deploy/protocol.py
dexgen/deploy/hardware_client.py
dexgen/deploy/xhand_sdk_server.py
dexgen/deploy/dexgen_runtime.py
dexgen/deploy/manus_dexgen_real.py
dexgen/deploy/run_xhand_sdk_server.sh
dexgen/deploy/run_manus_dexgen_real.sh
dexgen/deploy/README.md
```

职责：

- `xhand_sdk_server.py` 在 Docker 内独占 EtherCAT，读取状态并发送绝对关节目标。
- `manus_dexgen_real.py` 在本机 CUDA 环境运行 MANUS、CW/CCW RL command、
  DexGen Diffusion 和 IDM。
- `protocol.py` / `hardware_client.py` 提供长度前缀 JSON 的本机 TCP 通信。
- Docker 服务 5 秒收不到控制器请求时自动发送 mode 0。

真机控制参数保持已验证配置：

```text
mode=4
kp=10
kd=0.3
speed=0
torque=0
20 Hz hardware control
10 Hz DexGen generation
每周期目标变化上限=1/24 rad
```

DexGen 默认参数保持仿真 README 第 2 节配置：

```text
Diffusion: diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM:       idm_modes_full_dr_active_recovery_v3_best.pt
CW expert: student_9402.pth
CCW expert: student_9503.pth
MANUS gain: 0.55
guidance strength: 50000
DDIM steps: 25
```

注意触觉契约：

- 预抓取接触判定沿用成功的 V1 真机配置，屏蔽食指/小指触觉 `1,4`。
- DexGen v3 和 9402/9503 推理状态保留全部五路原始触觉，不能套用 8103 的
  V1 mask，否则会造成训练/部署观测不一致。

### 4. 当前真机按键

| 按键 | 行为 |
| --- | --- |
| `e` | 缓慢回 home、空载触觉标定、预抓取；完成后 HOLD。 |
| `Enter` / `n` | RL command → DexGen → IDM 自主旋转。 |
| `t` | 纯 MANUS 21 维关节遥操作。 |
| `h`，再按 `a` | MANUS keypoint guidance → DexGen → IDM。 |
| `d` | 保持当前姿态。 |
| `w` | 暂停/恢复暂停前的输入源。 |
| `q` | 缓慢张开回 home，并保持。 |
| `c` / `v` | CW / CCW。 |
| `i` | 打印输入源、MANUS 状态、历史长度和温度。 |
| `x` / `Ctrl-C` | mode 0、保存日志并退出。 |

预抓取后默认仍为 HOLD，必须明确按 `Enter`、`n` 或 `t` 才会开始动作；按
`w` 不会意外启动策略。

MANUS 超过 200 ms 没有新数据、断连或数据无效时会自动进入 HOLD。

### 5. 离线与真机验证

本机环境：

```text
Python 3.11.15
torch 2.7.0+cu128
GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB
```

已验证：

- Diffusion 和 IDM checkpoint 格式、83 维 tactile state、关节顺序一致。
- 9402/9503 均可加载，并输出有限的 `(1, 21)` action。
- DexGen CUDA warmup 成功。
- DDIM 25：首次约 0.154 秒，预热后约 0.038–0.040 秒，满足 10 Hz 生成。
- Docker SDK → TCP 8031 → 本机 CUDA 控制器完整 HOLD 链路通过。
- HOLD 冒烟测试记录 594 帧，数据全部有限，控制状态始终为 `hold`。
- 全项目测试：65 项通过，1 项跳过。

测试时确认退出路径：

```text
[hardware] mode=0 and EtherCAT disconnected
Ethercat Close Success
```

## 今日遇到的问题

### 问题 1：Python/动态库环境不能直接合并

厂商 SDK 是 Python 3.10，并要求 `libfmt.so.8`；本机 CUDA 环境是 Python
3.11，系统侧是 libfmt9。因此没有强行把 SDK 安装到 `env_isaaclab`，而是采用
Docker SDK 服务和本机 CUDA 控制器的分层方案。

### 问题 2：终端被 SDK 高频日志覆盖

`get_joint_state` 会逐关节打印大量 SDK info。已在
`rl_isaaclab/deploy_xhand/hw_interface.py` 中加入默认静默处理；需要调试 SDK
时可给服务传 `--sdk-logs` 恢复日志。

### 问题 3：预抓取后没有明显继续抓紧

首次 MANUS/DexGen 真机预抓取输出：

```text
[pregrasp] complete and HOLD:
contacts=2 force=[0.48, 0.0, 0.48, 0.06, 0.0]
```

新脚本最初在检测到两指接触后立即 HOLD，没有体现原策略的
`final_pose = trajectory[j+2]` 意图。已修改为：

- 检测到接触后沿原轨迹再缓慢执行最多 2 帧。
- 终端打印：

```text
[pregrasp] contact detected; tightening slowly for 2 settling frame(s)...
```

- 随后进入 HOLD，仍需按 `Enter` 才启动 DexGen。

该修复先通过离线测试。XHand 自然冷却后曾完成真机复测，当时用户确认以下流程
能够运行：

- 启动时温度/错误检查正常。
- `e` 缓慢回 home、触觉标定和预抓取正常。
- 检测接触后继续慢合 2 帧能够执行。
- 抓取完成后正确进入 HOLD，没有自动启动策略。
- `w` 不会从预抓取 HOLD 意外启动策略。
- `q` 缓慢张开回 home 正常。
- 退出时 mode 0 和 EtherCAT 断开正常。

**2026-07-29 更新：该“两帧主动慢合”方案已撤销。** 后续更长测试发现它会
使拇指/无名指附近夹持过紧并加速升温。成功的 8103 baseline 并未在 HOLD 前
把 `j+2` 两帧继续压到硬件，因此不能把“能够执行”误认为安全 baseline。

### 问题 4：关节过温保护 `1501028`

一次预抓取后程序安全退出：

```text
RuntimeError: XHand joint error code(s): 18:1501028
```

SDK 定义：

```text
1501028 = JOINT_ERROR_TEMP_PROCTED = 关节板过温保护
```

失败日志 `dexgen_manus_real.npz`：

- 514 帧。
- 最高采样温度为 `71°C`。
- 多个关节达到 `69–71°C`。
- 报错向量的 IsaacLab index 18 是 `right_hand_ring_dip_joint`，
  对应 XHand SDK id 15。
- 最高采样温度所在 IsaacLab index 8 是 `right_hand_ring_mcp1_joint`。

程序检测到异常后已保存日志、发送 mode 0 并关闭 EtherCAT。Docker 终端显示
`Ethercat Close Success` 是正常安全退出，不是服务崩溃。

已做的安全修复：

- 预抓取和缓慢回 home 的每一帧都检查温度和错误码。
- 主控制循环持续检查所有 21 路错误码。
- 错误提示现在包含关节名称和中文错误含义。
- 软件温度停机线由 80°C 调整为 68°C，以便在观察到的固件过温区间前退出。
- Docker 服务连接后先读取诊断；存在锁存错误时不发送 mode 4 控制命令。
- 无论正常退出、Ctrl-C、异常还是日志保存失败，最终都尝试 mode 0。

硬件状态交接：

- 最后一次程序退出已确认 mode 0 和 EtherCAT 断开。
- 用户已被明确要求关闭 XHand 独立电源并自然冷却。
- 不要通过忽略 `1501028` 继续运行。
- 不要使用冰袋、冷凝水或强冷风直吹。

### 问题 5：pytest 被 ROS 系统插件污染

直接运行 pytest 时，ROS Jazzy 的 Python 3.12 `launch_testing` 插件会被
Python 3.11 环境自动加载，造成缺少 `lark` 的启动错误。测试应使用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
```

这不是项目代码故障。

另有一个旧测试仍断言 186 维观测，已经更新为当前 v2 的 141 维契约：

```text
单帧 47 = 21 joint_pos + 21 target + 5 tactile
三帧 141
历史 30 x 47
```

### 问题 6：DexGen 真机关节运动呈一帧一帧的台阶感

冷却后的预抓取、MANUS 按键和 DexGen 策略切换均正常，但用户观察到 DexGen
旋转时关节运动不够平滑。

对比实现后确认：

- 成功的 8103 RL 部署每个 20 Hz 周期直接产生 action。
- 新脚本最初把约 40 ms 的 Diffusion/IDM 推理和 20 Hz 硬件控制放在同一线程。
- DexGen 每 10 Hz 更新一次绝对目标，硬件循环再以 `1/24 rad` 限幅追赶，
  因而容易形成 100 ms 一次的新目标台阶。

日志 `dexgen_manus_real.npz` 中，DexGen 模式约 33% 的目标关节变化达到
`0.04167 rad` 单周期上限，实际关节变化 95 分位约为 `0.074 rad/帧`；成功的
8103 日志约为 `0.047 rad/帧`。这说明问题主要是推理/控制调度，而不是
mode、kp、kd 或 EtherCAT。

已修改：

- Diffusion/IDM 和 RL observation 更新移到独立后台线程。
- 硬件主循环保持独立、稳定的 20 Hz。
- 后台队列拥堵时丢弃最旧状态，禁止积累陈旧控制延迟。
- 保持 `1/24 rad` 最大目标步幅不变。
- 新增加速度限制目标平滑器，默认 `8 rad/s²`，并使用制动速度接近目标，
  避免目标切换时直接吸附。
- 日志新增 `desired_target`、`command_target`、`monotonic_s` 和
  `loop_dt_s`，可直接检查目标平滑和真实周期。

假硬件 + 完整 CUDA 模型测试结果：

```text
20 Hz loop mean:  50.000 ms
20 Hz loop p95:   50.140 ms
20 Hz loop p99:   50.370 ms
20 Hz loop max:   50.876 ms
target step max:  0.041667 rad
```

完整后台推理、CW DexGen、日志保存和退出均通过；该平滑版本尚未重新连接
真机验证。

## 2026-07-29：预抓取过紧与第二次热保护

### 现象

- 用户观察到预抓取时拇指和无名指捏合特别紧。
- 随后设备再次保护性停止。
- 最新 `dexgen_manus_real.npz` 共 964 帧：
  - 启动 HOLD 最高温度 `58°C`。
  - 预抓取/HOLD 后升至 `65°C`。
  - DexGen 运行 524 帧，期间最高约 `66°C`。
  - 暂停 HOLD 后继续升至 `68°C`，程序/固件保护退出。
  - 触觉峰值约为 `[2.21, 3.82, 2.69, 2.20, 0.00] N`。

### 根因

新脚本底层参数确实沿用了成功 baseline：

```text
mode=4, kp=10, kd=0.3, speed=0, torque=0
action step=1/24 rad
pregrasp contact threshold=0.1 N
min fingers=2
pregrasp tactile mask=1,4
```

但“检测接触后再主动慢合 2 帧”是后加逻辑，不是 baseline 的真实硬件行为。
baseline 在检测接触后由冻结状态把硬件目标设为实测位置，并不会在 HOLD 前
继续挤压两帧。此次还在手已经达到 `58°C` 时开始预抓取，热余量不足。

### 2026-07-29 修复

- 删除接触后的两帧主动慢合。
- 达到两指接触后立即调用硬件 HOLD，把目标设为实测当前位置，消除残余
  position error 带来的继续夹紧。
- 增加预抓取单指力量上限 `1.0 N`：即使不足两指，只要任一有效指达到上限
  也立即停止闭合。
- 起抓温度门最初设为 `55°C`；用户于 2026-07-29 明确要求改为
  `68°C`，现与总体软件停机线一致。
- 总体软件停机条件改为温度达到（不是超过）`68°C`。
- 67 项测试通过，1 项跳过。
- 该版本尚未重新上真机；设备应继续断电自然冷却。

### 后续 baseline 约定

- `dexgen/deploy/README_XHAND_HARDWARE.md` 中的 8103 成功部署作为硬件控制、
  预抓取、按键安全和退出行为 baseline。
- 新功能不得再改变 baseline 的 `mode/kp/kd/speed/torque/action_scale`。
- DexGen v3 推理仍使用其训练契约对应的 9402/9503 和五路触觉；8103 的
  `disable_tactile_ids=1,4` 只用于其自身策略和新脚本的预抓取接触判定，
  不能直接套到 DexGen state。

## 明日启动顺序

先确认手已经充分自然冷却，再重新上电。不要直接从策略旋转开始。

### 终端 1：MANUS 输入源

```bash
sudo docker exec ros2_dev supervisorctl restart manus_core_mini
sudo docker exec -it ros2_dev bash
```

容器内：

```bash
cd /root/xbot/algorithm/teleopx/retargetx-2.2.18
bash script/run_hand_manus_dexgen_source.sh
```

### 终端 2：Docker SDK 服务

```bash
cd /home/user/rotate_dexgen
bash dexgen/deploy/run_xhand_sdk_server.sh
```

必须看到：

```text
[hardware] listening on tcp://127.0.0.1:8031
```

如果启动时仍报告 `1501028`，立即保持断电继续冷却，不进行后续测试。

### 终端 3：本机 CUDA 控制器

```bash
cd /home/user/rotate_dexgen
bash dexgen/deploy/run_manus_dexgen_real.sh
```

## 下一步计划

按以下顺序逐项进行，每一步正常后再进入下一步：

1. 空手启动，观察是否存在锁存错误，按 `i` 检查 MANUS 状态和最高温度。
2. 不放物体，只测试 `t` 的小幅 MANUS 关节遥操；用 `w` 暂停、`d` HOLD。
3. 测试 `q` 是否仍沿安全轨迹缓慢张开。
4. 设备充分冷却后重新验证 2026-07-29 baseline 预抓取修复：接触后必须立即
   显示 HOLD，不再出现 `tightening slowly for 2 settling frame(s)`。
5. 观察温度；若达到 68°C，接受程序自动 mode 0，不继续测试。
6. 冷却充分后重新启动，按 `Enter` 测试 CW DexGen，先运行很短时间并用
   `w` 暂停。重点确认后台推理/加速度限幅版本的关节运动是否明显变平滑。
7. 再测试 `v` 的 CCW，以及 `h` 后 `a` 的 MANUS-guided DexGen。
8. 检查 `dexgen_manus_real.npz` 中的关节跟踪、触觉、温度和输入源状态；
   接触后必须保持 baseline 的立即 HOLD，不再尝试额外闭合。不要先调整
   kp/kd、动作幅度或策略参数。

## 明日必须记住

- “接触后再慢合 2 帧”已因过紧和升温风险撤销，禁止恢复。
- `1501028` 是真实过温保护，绝对不能屏蔽。
- XHand 当前应在断电冷却状态；重新上电前先确认手体已经冷却。
- 不同时运行两个 EtherCAT 主站。
- 真机下方始终放承接平台；mode 0 会释放物体。
- 8103 原始策略和新 DexGen 部署的触觉契约不同，不要混用 mask。
- 已验证的 `mode=4, kp=10, kd=0.3, speed=0, torque=0` 保持不变。

## 2026-07-29：MANUS 自动 CW/CCW 与张手 HOLD 真机入口

根据 `dexgen/README.md` 第 3 节新增独立真机脚本：

```text
dexgen/deploy/manus_auto_real.py
dexgen/deploy/run_manus_auto_real.sh
```

实现方式：

- 继承现有 `Controller`，不复制或修改 Docker SDK、EtherCAT、预抓取、
  平滑目标、温度保护和 mode 0 退出逻辑。
- 复用 `RealTimeAutoModeWorker` 和现有个人 profile：
  - `manus_auto_mode_profile.pt`
  - `manus_open_profile.pt`
- 自动线程以 profile 的 10 Hz 真实时间节拍读取 MANUS。
- CW/CCW trend 自动修改 DexGen mode；切换时清空旧方向的 RL/DexGen 历史。
- 完全张开持续 1 秒只会锁存 XHand 实测目标并进入 HOLD，不会把人的张手
  关节角发送给硬件。
- 恢复自然抓握约 0.2 秒后恢复进入 HOLD 前的方向和控制源。
- MANUS 断连、过期或手势推理错误时自动 HOLD。
- 手动 `w` 暂停优先于自动恢复。
- `t` 纯 MANUS 遥操时禁用自动手势接管；`h` 后 `a` 启动 guidance 时重新
  启用自动方向和张手 HOLD。

同时对基础真机控制器做了小型可扩展重构：

- 增加 `update_external_control()`、`close_extensions()` 和
  `ready_prompt()` hook。
- 抽出共用 CLI parser/参数校验/控制器运行函数。
- 原 `run_manus_dexgen_real.sh` 行为保持不变。

验证：

- 两个 profile 均通过加载和 schema 校验。
- 自动 HOLD 状态机单测通过：首次张手只调用一次硬件 HOLD，重复轮询不会
  重复锁存；恢复抓握会返回原控制源；方向切换会重置策略历史。
- 假硬件 + 完整 CUDA 模型冒烟通过：自动入口成功加载两份 profile；MANUS
  source 为 error 时按 `Enter` 自动锁存 HOLD；`w` 暂停/恢复不会绕过自动
  HOLD；Ctrl-C 正常保存日志并发送 mode 0。
- `dexgen/deploy/README.md` 已精简为三终端启动、两种终端 3 模式和统一按键表。
- 当前自动手势真机入口已完成代码和离线测试，尚未连接 XHand 实测。
