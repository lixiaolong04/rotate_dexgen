# XHand + DexGen 真机运行

## 运行前

- XHand 独立供电，EtherCAT 网线连接 `enp129s0`。
- 手和物体下方放置承接平台。
- 不要同时启动两个 XHand SDK 服务或两个终端 3 控制器。
- 固定硬件参数为 `mode=4, kp=10, kd=0.3, speed=0, torque=0`。

## 终端 1：MANUS 输入

宿主机执行：

```bash
sudo docker exec ros2_dev supervisorctl restart manus_core_mini
sudo docker exec -it ros2_dev bash
```

进入 `ros2_dev` 后执行：

```bash
cd /root/xbot/algorithm/teleopx/retargetx-2.2.18
bash script/run_hand_manus_dexgen_source.sh
```

## 终端 2：XHand SDK

```bash
cd /home/user/rotate_dexgen
bash dexgen/deploy/run_xhand_sdk_server.sh
```

看到以下内容后再启动终端 3：

```text
[hardware] listening on tcp://127.0.0.1:8031
```

## 终端 3：选择一种控制模式

### 模式 A：按键切换遥操与 DexGen

```bash
cd /home/user/rotate_dexgen
bash dexgen/deploy/run_manus_dexgen_real.sh
```

### 模式 B：MANUS 手势自动切换 CW/CCW

```bash
cd /home/user/rotate_dexgen
bash dexgen/deploy/run_manus_auto_real.sh \
  --guidance-strength 1000
```

自动模式使用现有个人 profile：

```text
manus_dexgen_bridge/logs/manus_auto_mode_profile.pt
manus_dexgen_bridge/logs/manus_open_profile.pt
```

调试时直接修改命令中的 `5000`。数值越大，Diffusion 越强地服从引导
command；建议每次只调整该参数并记录日志，不要同时修改硬件增益或动作限幅。

按 `e` 完成预抓取，再按 `Enter` 启用自动 DexGen：

- CW/CCW 手势自动选择旋转方向。
- 手掌完全张开并保持 1 秒：自动 HOLD，只锁存 XHand 当前姿态，不会让
  XHand 跟随人的张手动作松开。
- 恢复自然抓握约 0.2 秒：恢复进入 HOLD 前的方向和控制源。
- MANUS 断连或数据过期：自动 HOLD。
- 手动 `w` 暂停的优先级高于手势自动恢复。

## 按键

| 按键 | 功能 |
| --- | --- |
| `e` | 按 8103 baseline 缓慢张开、触觉标定、接触即停预抓取。 |
| `Enter` / `n` | 启动 DexGen；自动模式下同时启用手势识别。 |
| `t` | 纯 MANUS 21 关节遥操；自动手势暂不接管。 |
| `h` 后按 `a` | 标定并启动 MANUS guidance → DexGen；自动模式下方向仍由手势选择。 |
| `w` | 手动暂停并 HOLD；再次按下恢复。 |
| `d` | 停止自动/遥操输入并 HOLD。 |
| `q` | 沿安全轨迹缓慢张开回 home。 |
| `c` / `v` | 手动选择 CW / CCW。 |
| `m` | 自动模式下手动切换 CW/CCW。 |
| `i` | 显示控制源、方向、MANUS 状态和最高温度。 |
| `x` / `Ctrl-C` | 发送 mode 0、保存日志并退出。 |

## 建议操作顺序

1. 按 `i` 检查 MANUS 为 `recording`。
2. 放好圆柱，按 `e` 预抓取。
3. 按 `Enter` 启动 DexGen。
4. 先运行 3–5 秒，按 `w` 测试暂停。
5. 按 `q` 缓慢松手。
6. 按 `Ctrl-C` 退出，确认终端 2 显示：

```text
[hardware] mode=0 and EtherCAT disconnected
```

任一关节达到 `68°C` 或出现厂商错误码时会自动 mode 0。预抓取两指接触后
立即 HOLD 实测姿态，不再额外闭合；任一有效指达到 `1.0 N` 也会停止闭合。
