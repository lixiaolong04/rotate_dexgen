# MANUS → RetargetX → DexGen：快速启动

当前推荐链路：

```text
MANUS 手套 → RetargetX（Docker）→ 关节映射 → TCP 8020 → DexGen（Isaac Sim）
```

RetargetX 输出的是同事提供的高质量 XHand Green 重定向结果。桥接程序只将它的 21 个硬件风格 `bend/rota` 关节名转换为 DexGen 使用的 `mcp/pip/dip` 关节名和顺序；DexGen 再自行转换为 10 个关键点（每指 PIP + 指尖）并运行 Diffusion/IDM。

## 每次运行：先启动 Docker 数据源

在**主机终端**执行，进入 Docker：

```bash
sudo docker exec -it ros2_dev bash
```

在 Docker 内执行，保持此终端不要关闭：

```bash
cd /root/xbot/algorithm/teleopx/retargetx-2.2.18
bash script/run_hand_manus_dexgen_source.sh
```

正常会显示：

```text
Starting input-only path: MANUS -> RetargetX -> DexGen tcp://127.0.0.1:8020
```

该启动器不启动 Viser、SDK 或 `/hand_controller/commands` 控制输出；只提供 DexGen 输入。

## 验证手套输入（推荐）

另开一个**主机终端**：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/command_probe.py --seconds 30
```

开始后先静止约 2 秒，再分别活动五根手指。输出应包含约 80 Hz 的数据率和十个关键点的毫米级位移。

## 启动 DexGen 物体旋转仿真

仍在主机终端执行：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/eval_manus_guided.py --device cuda:0
```

启动后等待日志出现 `AUTO-REARMED`，然后以小幅手势开始控制。手套仅作为相对手型参考；最终写入仿真手的 IDM 目标仍受仿真关节范围保护。

当前已验证的 MANUS 默认任务空间增益为 **0.55**。若需要临时试验其他映射幅度，可在命令末尾加 `--manus_command_gain 0.45` 或 `--manus_command_gain 0.60`。

常用按键：

| 按键 | 作用 |
| --- | --- |
| `a` | 在当前位置重新标定并开始 MANUS 引导 |
| `d` | 停止 MANUS 引导，保持当前安全目标 |
| `c` / `v` | 切换顺时针 / 逆时针旋转模式 |
| `h` | 重新选择 MANUS 输入，随后按 `a` 标定 |
| `n` | 切到原来的 RL 专家加噪声输入，用于对照 |
| `i` | 打印输入延迟、标定状态和模式 |
| `q` | 退出仿真 |

退出仿真时会自动打印并保存 MANUS 与 RL 加噪声的 10 点位移对比：

```text
manus_dexgen_bridge/logs/latest_command_comparison.pt
```

采集方法：先用 `h`、`a` 操作 MANUS 约 20–30 秒；再按 `n` 运行 RL 加噪声约 20–30 秒；最后按 `q`。输出包含每个 PIP/指尖点的平均值、P95 和最大移动距离（mm）。

## 轻量输入采集（不启动 Isaac Sim）

只比较手套输入与已保存的 RL+噪声命令时，使用下面脚本；Docker 的 RetargetX→DexGen 数据源仍需保持运行：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/record_input_comparison.py
```

按键：`a` 标定手套零点；`r` 开始/暂停 MANUS 采集；`p` 开始/暂停 RL+噪声离线回放采集；`s` 查看状态；`q` 保存并退出。结果写入：

```text
manus_dexgen_bridge/logs/latest_lightweight_input_compare.pt
```

此脚本不导入 IsaacLab、不创建仿真环境、不加载策略，也不会控制任何手。RL 侧回放的是上一次保存的 `latest_command_comparison.pt` 中的真实 RL+噪声 10 点命令；若该文件不存在，先完成一次 Isaac Sim 的对照采集。

## 根据 MANUS 趋势自动切换 CW / CCW

先做一次个人手势标定（不启动 Isaac Sim）：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/calibrate_auto_mode.py
```

方向标定支持空手与持物两种条件：`1` 开始/结束空手 CW，`2` 空手 CCW，`3` 持圆柱 CW，`4` 持圆柱 CCW；`x` 停止当前段，`s` 查看状态，`q` 训练并保存。四类各录 2 段、每段连续 5–6 秒。运行时会比较四个原型但只输出 CW/CCW，因此可兼容空手和持物。配置保存为：

```text
manus_dexgen_bridge/logs/manus_auto_mode_profile.pt
```

先用轻量探测确认方向判定，不启动 Isaac Sim：

```bash
python manus_dexgen_bridge/probe_auto_mode.py
```

连续做 CW/CCW 手势，观察是否稳定出现 `SWITCH -> CW/CCW`；按 `r` 清空当前积分，按 `q` 退出。

之后用自动模式脚本启动 DexGen：

```bash
python manus_dexgen_bridge/eval_manus_auto_mode.py --device cuda:0
```

程序用独立的真实时间 10 Hz 线程读取约 1 秒的 MANUS 10 点轨迹，不受 Isaac Sim 帧率影响；再通过积分、确认时间和切换冷却自动选择 CW/CCW 模型。终端中的 `[auto-mode-rt]` 会显示当前积分，出现 `[mode-switch] source=manus-trend-rt` 表示已切换。`c`、`v` 仍保留为调试时的人工覆盖键。

## 标定 MANUS 张开手势（HOLD 的前置步骤）

此步骤不启动 Isaac、不会控制执行手。它只记录 RetargetX 的 21 维手套姿态，建立个人化的 `open_score`；完全张开后续将表示 HOLD，而不是让 XHand 张开。

先保持 Docker 输入源运行，再在主机执行：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/calibrate_open_mode.py
```

按 `o` 开始/结束一段完全张开、平坦的手掌记录；按 `g` 开始/结束一段自然稳定抓握记录。每类录制 2–3 段，每段保持静止 3–5 秒；按 `q` 训练并保存：

```text
manus_dexgen_bridge/logs/manus_open_profile.pt
```

随后在不启动 Isaac 的情况下验证：

```bash
python manus_dexgen_bridge/probe_open_mode.py
```

探针默认要求完全张开连续保持 **1 秒**才输出 `ENTER HOLD`；恢复自然抓握连续约 0.2 秒后输出 `LEAVE OPEN`。按 `q` 退出。正常 CW/CCW 手势、短暂伸指和静止抓握都不应误触发 HOLD。

## 三态输入探针（CW / CCW / HOLD）

已同时完成方向和张开标定后，可在不启动 Isaac 的情况下验证完整输入状态机：

```bash
python manus_dexgen_bridge/probe_control_mode.py
```

预期行为：CW/CCW 趋势切换到相应方向；完全张开超过 1 秒输出 `HOLD`；恢复抓握约 0.2 秒后，立即恢复进入 HOLD 前的 CW 或 CCW。方向识别器会重新开始采样，后续检测到反向趋势时才切换方向。按 `r` 清空状态，按 `q` 退出。

`eval_manus_auto_mode.py` 会自动加载上述方向与张开 profile。在 Isaac 中完全张开超过 1 秒时，日志会显示 `[manus-hold] started`：执行手锁存当前 PD target，且不调用 Diffusion/IDM；恢复抓握后会显示 `[manus-hold] ended` 并按此前方向恢复。

## 不能同时运行的程序

MANUS 的 8000 端口只能被一个接收端消费。运行本链路时，不要同时启动：

```text
geort_publisher.py
manus_ros_bridge.py
run_hand_manus_visualizer.sh（RetargetX 手部网页可视化）
```

若之前启动过 RetargetX 可视化，在 Docker 内先停止它：

```bash
pkill -INT -f 'python hand_retarget.py'
```

然后再启动 `run_hand_manus_dexgen_source.sh`。

## 仅查看 RetargetX 手部网页（不运行 DexGen）

在 Docker 内执行：

```bash
cd /root/xbot/algorithm/teleopx/retargetx-2.2.18
bash script/run_hand_manus_visualizer.sh
```

主机浏览器打开：`http://127.0.0.1:8021`。
