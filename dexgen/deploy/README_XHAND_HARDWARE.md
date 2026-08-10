# XHand 真机启动手册

项目目录：`/home/user/rotate_dexgen`

本文档记录已经在 XHand1 Pro 真机上验证成功的启动指令。后续测试优先复用这里的固定参数，不自行调整策略幅度、控制增益、控制模式或触觉屏蔽配置。

## 1. 已验证：XHand 强化学习旋转策略

### 硬件准备

1. XHand 已独立供电。
2. EtherCAT 网线已连接到电脑有线网口 `enp129s0`。
3. 手和物体下方有承接平台，急停或退出时物体不会直接掉落。
4. 确认没有其他 EtherCAT 主站程序占用 `enp129s0`。

### 启动指令

在主机终端执行：

```bash
cd /home/user/rotate_dexgen

docker run -it --rm \
  --network host \
  --privileged \
  -v /home/user/rotate_dexgen:/workspace \
  -w /workspace \
  xhand-deploy:22.04 \
  bash scripts/deploy_unilab_v1_student.sh 8103 \
    --log v1_8103_real.npz
```

该脚本使用已经成功部署过的固定配置：

```text
checkpoint=student_8103_best.pth
mode=4
kp=10
kd=0.3
speed=0
action_scale_mul=1.0
min_fingers=2
disable_tactile_ids=1,4
```

### 操作顺序

1. 程序启动后，等待 XHand 缓慢张开并稳定。
2. 将圆柱体放到预抓取位置。
3. 按 `e`，执行缓慢预抓取。
4. 终端显示“抓取完成并保持”后，按 `Enter` 启动旋转策略。
5. 按 `w` 冻结并保持当前姿态；再次按 `w` 继续策略。
6. 按 `q` 沿安全轨迹缓慢张开回家。
7. 按 `Ctrl-C` 退出：程序先发送 `mode=0` 释放舵机，再断开 EtherCAT。

从另一终端停止程序：

```bash
touch /home/user/rotate_dexgen/STOP
```

`Ctrl-C` 或 `STOP` 都会让手失去夹持力，因此物体下方必须有承接平台。硬件异常时直接断电。

### 日志分析

本次运行日志保存在：

```text
/home/user/rotate_dexgen/v1_8103_real.npz
```

分析指令：

```bash
cd /home/user/rotate_dexgen

docker run --rm \
  --network none \
  -v /home/user/rotate_dexgen:/workspace \
  -w /workspace \
  xhand-deploy:22.04 \
  python3 scripts/analyze_deploy_log.py \
    v1_8103_real.npz \
    --ckpt ckpts/unilab_v1_students_final_20260630/student_8103_best.pth
```
