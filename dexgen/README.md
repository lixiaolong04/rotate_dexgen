# DexGen 运行手册

项目根目录：`/home/user/rotate_dexgen`。DexGen 有三种运行方式：RL 加噪声基线、MANUS 按键切换遥操作、MANUS 手势自动切换遥操作。

程序已按数据、训练、评估、流水线和工具分类整理；目录说明及本机/云端迁移约定见
[docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)。所有运行命令均使用分类后的新路径，
根目录不再保留重复入口。

## 1. 纯 RL 加噪声：按键切换 CW/CCW

在宿主机终端运行：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab

python dexgen/commands/eval/eval_guided.py \
  --controller noisy_rl \
  --mode cw \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --sensor_profile raw \
  --noise_alpha 0.1 \
  --steps 1200 \
  --so3 \
  --interactive_mode_switch \
  --device cuda:0
```

`--noise_alpha` 是归一化 RL action 的均匀噪声半幅；例如 `0` 为无噪声、`0.05` 为轻噪声、`0.1` 为当前标准对照。`--mode cw` 仅指定启动方向，运行中可通过按键切换。

### 历史对照模型

交付目录已清理 tactile-v1、早期 tactile-v2b、v5、DAgger v4 和 contact-risk 等历史
checkpoint，只保留上方的 active-recovery v3 成对模型。历史结果仍保留在
`dexgen/docs/` 和 `dexgen/worklog.md`；确需复现实验时从云端归档取回对应权重或按
记录重新训练。不要把不同版本的 Diffusion 和 IDM 混用。

在相同主动扰动评估中，v3 v2b 的恢复成功率为 73.68%，历史 tactile-v1 为
68.53%。

| 按键 | 功能 |
| --- | --- |
| `c` | 切换为 CW（顺时针）RL 专家。 |
| `v` | 切换为 CCW（逆时针）RL 专家。 |
| `m` | 在 CW 与 CCW 间切换。 |
| `i` | 打印当前旋转模式和运行状态。 |
| `q` | 安全退出仿真。 |

### 1.1 RL-guided 双手交互测试

这一入口不是直接执行 noisy-RL：它用 RL+噪声生成上层关键点 command，再经 Guided Diffusion → IDM 驱动执行手，同时显示无物体的 command 参考手。`s/a/r/o/1–5/x` 仅在此入口可用。`a` 是手动 PD HOLD 测试：锁存执行手当前 PD target，并绕过 Diffusion/IDM；`r` 会经 RL warm-up 后恢复旋转。`s` 保持原有的 zero-motion guidance 对照。

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab

python dexgen/commands/eval/eval_dual_hand.py \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --interactive_mode_switch \
  --noise_alpha 0.1 \
  --dangerous_command_strength 0.6 \
  --dangerous_command_duration 0.4 \
  --max_command_motion_m 0.03 \
  --guidance_strength 100000 \
  --command_time_mode separate \
  --command_horizon_steps 1 \
  --command_second_horizon_steps 2 \
  --ddim_steps 25 \
  --steps 3000 \
  --device cuda:0
```

调试参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--noise_alpha` | `0.1` | RL action 的均匀噪声半幅，范围为 `[-0.1,+0.1]`；每次 10 Hz command 更新重新采样，并在整个运行期间持续生效。 |
| `--dangerous_command_strength` | `1.0` | `o` / `1–5` 张掌或撤指 command 的幅度；`1.0` 表示把所选手指的屈曲关节完全插值到伸展下限。 |
| `--dangerous_command_duration` | `1.0` 秒 | 每次按下 `o` / `1–5` 后危险 command 的持续时间；按 `x` 可提前取消。 |
| `--max_command_motion_m` | `0.03` m | 每个关键点 command 的最大移动距离，即默认 30 mm；调成 `0.01` 即限制为 10 mm。 |

五根手指当前共用 `dangerous_command_strength` 和 `dangerous_command_duration`；`1–5` 只决定选择哪根手指。`noise_alpha` 是连续 RL action 噪声，不使用 `dangerous_command_duration`。

除 `c/v/m/i/q` 外，还支持：

| 按键 | 功能 |
| --- | --- |
| `s` | 将 external guidance motion 置零并冻结参考手；执行手仍由 Diffusion/IDM 闭环，不等同于暂停。 |
| `r` | 结束零位移测试，恢复 RL+噪声 command。 |
| `o` | 注入整手张开危险 command。 |
| `1` / `2` / `3` / `4` / `5` | 分别注入撤拇指 / 食指 / 中指 / 无名指 / 小指的危险 command。 |
| `x` | 取消当前危险 command。 |

## 2. MANUS 遥操作：按键切换方向和输入源

先在终端 1 启动 MANUS → RetargetX → TCP 8020 输入源：

```bash
sudo docker exec ros2_dev supervisorctl restart manus_core_mini
sudo docker exec -it ros2_dev bash
```

进入 Docker 后运行，并保持该终端：

```bash
cd /root/xbot/algorithm/teleopx/retargetx-2.2.18
bash script/run_hand_manus_dexgen_source.sh
```

终端 2 先确认手套输入正常：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/command_probe.py --seconds 30
```

随后启动仿真：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/eval_manus_guided.py \
  --manus_command_gain 0.55 \
  --guidance_strength 50000 \
  --device cuda:0
```

`--manus_command_gain` 控制手套相对动作转换成关键点 command 的幅度；`--guidance_strength` 控制 Diffusion 跟随该 command 的程度。调试时可以分别修改，前者越大手套动作幅度越大，后者越大执行手越服从手套 command。

运行时有三条互斥控制路径：

```text
h: MANUS → 关键点 command → Diffusion → IDM → XHand PD target
n: noisy RL → Diffusion → IDM → XHand PD target
t: MANUS 21 维 q_target → 关节限位/变化限幅 → XHand PD target
```

`t` 是纯 MANUS 关节遥操：它直接采用 RetargetX 经 TCP 8020 输出的绝对关节目标，绕过 Diffusion、IDM 和 tactile-conditioned 策略。仿真端仍使用原有绝对 PD 控制，并会进行关节名称重排、Isaac 关节限位和单周期变化限幅。

按键功能：

| 按键 | 功能 |
| --- | --- |
| `a` | 以当前手套与仿真手姿态重新标定，并 arm MANUS guidance。 |
| `d` | 停止 MANUS guidance 或纯关节遥操；不再采用新手套目标，保持当前安全 PD target。 |
| `h` | 选择 MANUS → DexGen guidance；随后按 `a` 手动标定。 |
| `n` | 选择 RL 专家加噪声 → DexGen guidance，用于与手套输入即时对照。 |
| `t` | 选择纯 MANUS 关节遥操；要求 8020 当前为有效 `recording`，不需要按 `a` 标定。 |
| `c` | 选择 CW（顺时针）模型与专家。 |
| `v` | 选择 CCW（逆时针）模型与专家。 |
| `i` | 打印当前控制路径、8020 数据年龄、标定/arm 状态和旋转模式。 |
| `q` | 安全退出仿真。 |

纯关节遥操默认将每次 10 Hz 更新的单关节目标变化限制为 `0.15 rad`，可通过 `--manus_direct_max_joint_delta_rad` 调整。`c/v` 在纯遥操期间只更新下次 DexGen guidance 使用的方向，不影响当前直接关节输出；自动 CW/CCW 和张手 HOLD 也不会接管纯遥操。按 `h` 返回 MANUS guidance 后必须按 `a` 重新标定。

默认 MANUS task-space gain 为 `0.55`。reset 后纯遥操会等待一个新的有效 MANUS 帧再恢复；8020 数据过期、断连或关节契约失败时则自动 disarm，保持当前安全 PD target，链路恢复后需再次按 `t`。不要把断连当成静止手势。手套链路运行期间，不要同时启动 GeoRT、MANUS ROS bridge 或 RetargetX visualizer。

### 2.1 MANUS 平台抓取，然后切换 DexGen 旋转

该入口独立于上面的 `eval_manus_guided.py`。场景中圆柱竖直放在空中平台上，
XHand 掌心朝下悬停在圆柱上方；MANUS 只控制手指关节，抓稳后由脚本缓慢抬升
手根，随后切换为 DexGen 自主旋转。

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab

python manus_dexgen_bridge/eval_manus_tabletop_pickup.py \
  --manus_command_gain 0.55 \
  --guidance_strength 50000 \
  --device cuda:0
```

建议操作顺序：

1. 启动前保持手套张开，等待 8020 显示 `recording`。
2. 按 `t` 进入纯 MANUS 关节遥操，然后逐渐闭合手指抓住圆柱。
3. 用方向键微调手根的水平位置：`↑/↓` 对应世界坐标 `±X`，`←/→`
   对应世界坐标 `±Y`；`u` / `j` 分别升高/降低手根。
4. 按 `i` 检查接触手指数和总接触力；默认至少三指接触且总力达到
   `0.8 N` 才允许抬升。
5. 按 `g`，脚本用 2 秒将手根抬高 0.12 m。抬升期间禁止切换输入源。
6. 抬升完成后按 `n`，切换为 noisy-RL command → DexGen → IDM 自主旋转。
7. 用 `c` / `v` 切换 CW / CCW；按 `t` 可重新返回纯 MANUS 遥操。

抓取失败、圆柱掉落或希望重新开始时，可随时按 `r` 复位：圆柱回到平台，
XHand 回到掌心朝下、手指张开的初始位置，同时清除抓取、抬升和输入源切换状态。

平台和抬升参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--tabletop_surface_z` | `0.48` m | 平台上表面的世界坐标高度。 |
| `--tabletop_size_xy` | `0.50` m | 方形平台边长。 |
| `--tabletop_thickness` | `0.06` m | 平台厚度。 |
| `--tabletop_hand_height_m` | `0.16` m | 初始手根相对圆柱中心的高度。 |
| `--tabletop_lift_height` | `0.12` m | 抓取后的手根抬升距离。 |
| `--tabletop_lift_seconds` | `2.0` s | 平滑抬升时间。 |
| `--tabletop_min_contact_fingers` | `3` | 允许按 `g` 抬升所需的最少接触手指数。 |
| `--tabletop_min_total_contact_force_n` | `0.8` N | 允许抬升所需的五指总接触力。 |
| `--tabletop_grasp_assist_rad` | `0.05` rad | 抬升时附加到屈曲关节的小幅夹持量。 |
| `--tabletop_nudge_step_m` | `0.005` m | 每次方向键或 `u/j` 移动手根的距离。 |

此模式不会因抓取失败自动 reset：圆柱掉回平台后可以重新张手并再次抓取。按
`q` 退出并重新启动可恢复初始平台布局。为避免 DexGen 旋转期间改变任务坐标系，
按下 `g` 开始抬升后将不再接受手根微调按键。

平台模式会固定写入弹性指尖摩擦 `1.6`、圆柱摩擦 `1.0`。按 `g` 后会锁存
当前抓姿，并给各屈曲关节增加默认 `0.05 rad` 的小幅夹持量，直到按 `n`
切换 DexGen；圆柱仍由重力、接触和摩擦物理抬起，不会被直接吸附或传送。

### 2.2 XHand 真机 sim2real

真机入口位于 `dexgen/deploy`。厂商 Python 3.10 SDK 和 EtherCAT 在
`xhand-deploy:22.04` Docker 中运行；MANUS、RL command、Diffusion 和 IDM
在本机 `env_isaaclab` 中使用 CUDA 运行。这样既保持厂商 SDK 环境稳定，也可
直接复用本节的 MANUS/DexGen 输入源切换逻辑。

完整的三终端启动顺序、按键和首次分阶段验证见：

```text
dexgen/deploy/README.md
```

真机控制参数固定为已验证的 `mode=4, kp=10, kd=0.3, speed=0`。退出、
控制器断连或 5 秒无心跳时，Docker 硬件服务会发送 mode 0。

## 3. MANUS 遥操作：按手势自动切换 CW/CCW，并以张开手势 HOLD

终端 1：启动 MANUS → RetargetX → TCP 8020 输入源。

```bash
sudo docker exec ros2_dev supervisorctl restart manus_core_mini
sudo docker exec -it ros2_dev bash
```

进入 Docker 后运行，并保持该终端：

```bash
cd /root/xbot/algorithm/teleopx/retargetx-2.2.18
bash script/run_hand_manus_dexgen_source.sh
```

终端 2：检查 30 秒手套输入。

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/command_probe.py --seconds 30
```

终端 2：启动自动手势模式。

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/eval_manus_auto_mode.py \
  --manus_command_gain 0.55 \
  --guidance_strength 5000 \
  --max_command_motion_m 0.03 \
  --object_scale 0.5 \
  --device cuda:0
```

`--max_command_motion_m 0.03` 表示 MANUS 每个关键点 command 最大 30 mm。`--object_scale 0.6` 是圆柱资产的统一缩放系数；基础圆柱为直径 80 mm、高 64 mm，因此 0.6 对应直径约 48 mm、高约 38.4 mm。

程序会在独立真实时间 10 Hz 线程中读取 MANUS 轨迹。它会比较空手/持圆柱两种条件下的 CW、CCW 手势原型，再自动选择现有的 CW 或 CCW 模型；终端中的 `[auto-mode-rt]` 显示趋势积分，`[mode-switch] source=manus-trend-rt` 表示自动切换已生效。

完全张开手掌连续 **1 秒**会进入 `HOLD`：执行手锁存进入该状态时的 PD target，并绕过 Diffusion/IDM，因此不会跟随人的张开姿态而松手。恢复自然抓握约 0.2 秒后，策略按进入 HOLD 前的 CW 或 CCW 方向恢复，并经既有 RL warm-up 回到 guided 控制。终端将分别显示：

```text
[manus-hold] ... started; PD target latched, Diffusion/IDM bypassed
[manus-hold] ... ended; mode=CW/CCW; RL warm-up then guided rotation resumed
```

运行中 `c` / `v` 可人工强制为 CW / CCW，`m` 可在两个方向间切换，`i` 打印状态，`q` 安全退出。其余 MANUS 按键与第二节相同：`a` 重标定并 arm、`d` disarm、`h` 选 MANUS 输入、`n` 临时切为 RL+噪声输入。

首次使用或需要重新录制个人 profile 时，先标定张开 HOLD profile：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/calibrate_open_mode.py
python manus_dexgen_bridge/probe_open_mode.py
```

在 `calibrate_open_mode.py` 中，`o` 开始/结束一段完全张开、平坦的手掌姿态，`g` 开始/结束一段自然稳定抓握姿态；每类录 2–3 段、每段静止保持 3–5 秒，最后按 `q` 保存。`probe_open_mode.py` 中，短暂张开、CW/CCW 和短暂伸指不应触发 HOLD；完全张开持续 1 秒应出现 `ENTER HOLD`，恢复抓握应出现 `LEAVE OPEN`。

随后标定方向 profile：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/calibrate_auto_mode.py
python manus_dexgen_bridge/probe_auto_mode.py
```

在 `calibrate_auto_mode.py` 中，按 `1` 录空手 CW、`2` 录空手 CCW、`3` 录持圆柱 CW、`4` 录持圆柱 CCW；每类录 2 段、每段连续 5–6 秒。`x` 丢弃当前段，`s` 查看状态，`q` 训练并保存 profile。录制时保持同一类手型与连续 finger gait，不要在一段内停顿、反向、完全张开或切换手型。

随后在 `probe_auto_mode.py` 中分别以空手与持圆柱做 CW/CCW，观察稳定的 `SWITCH -> CW/CCW`；`r` 清空积分，`q` 退出。完整三态输入可用 `python manus_dexgen_bridge/probe_control_mode.py` 验证：CW/CCW 趋势会切换方向，张开持续 1 秒进入 HOLD，恢复抓握后恢复此前方向。

详细程序说明、测试记录和交接日志位于 [docs/](docs/)。

### 3.1 XHand 真机 sim2real

真机自动手势入口是 `dexgen/deploy/manus_auto_real.py`。它复用已验证的
Docker SDK、8103 baseline 预抓取、`w` 暂停、`q` 缓慢松手、mode 0 退出
和温度保护。CW/CCW 手势自动切换方向，持续张开手掌进入 HOLD；张手姿态
不会作为关节目标发送给 XHand。

完整运行指令和按键统一见：

```text
dexgen/deploy/README.md
```

## 4. 当前训练基线与后续计划（2026-07-27）

### 4.1 当前唯一推荐部署版本

目前 v5/v6/v7 均未超过 active-recovery v3 v2b，因此所有仿真测试和后续真机
部署仍应默认使用：

```text
Diffusion: dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM:       dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
```

v3 的正式主动恢复基准为：总恢复成功率 `73.68%`、episode 掉落率约 `55%`、
CW/CCW 转速约 `0.8 rad/s`。正式闭环评测入口是 `eval_dexgen.py`，不是用于
MANUS/guidance 交互的 `eval_guided.py`。

| 版本 | 核心改动 | 恢复率 | 掉落率 | 结论 |
| --- | --- | ---: | ---: | --- |
| v3 | 当前 active-recovery 基线 | 73.68% | 约 55.00% | 继续部署 |
| v5 | 16 帧、stride 2 长触觉历史 | 70.34% | 63.44% | 未超过 v3 |
| v6 | RMA encoder 初始化后联合微调 | 71.95% | 56.72% | 未超过 v3 |
| v7 | 冻结 RMA 触觉骨干 | 67.64% | 71.06% | 明显退化 |
