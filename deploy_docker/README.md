# Xhand 真机部署 —— Docker 用法

为什么用 Docker(Ubuntu 22.04 容器): Xhand 的 `hand_sdk .deb` 和 `hand_controller` wheel 都是给
**Ubuntu 22.04 (jammy)** 编的(依赖 `libfmt.so.8` 等),在本机 **Ubuntu 24.04** 上装不上(24.04 是
libfmt 9)。容器里用原生 22.04 => `.deb` 直接装得上、wheel 开箱即用,且不污染主系统、将来可原样搬到
机器人板载电脑。推理用 CPU 版 torch(策略很小,不需要 GPU)。

镜像里烤进: 系统 SDK + Python 绑定 + torch 等推理依赖。
**项目代码 + 当前 141-D 部署权重 + cache 在运行时挂载**到 `/workspace`(改代码不用重建镜像)。

---

## 0. 一次性: 装 Docker(主机 Ubuntu 24.04, 需 sudo)

```bash
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker "$USER"     # 加入 docker 组, 之后免 sudo
# 注销重登 或 newgrp docker 使组生效
newgrp docker
docker run --rm hello-world         # 验证
```

## 1. 构建镜像(需联网下载 torch/apt; 约几分钟)

当前没有真机或厂商 SDK 时，先构建离线验证镜像：

```bash
cd /home/user/rotate_dexgen/deploy_docker
./build.sh --offline
```

该镜像标记为 `xhand-deploy:offline`，不包含 `hand_controller`，只允许使用 FakeHW
执行 `--dry_run` 和离线测试。`run.sh` 在正式镜像不存在时会自动选择它。

拿到与当前部署代码匹配的旧版 SDK 后，将以下资产放入项目根目录的 `xhandSDK/`：

```text
xhandSDK/hand_sdk_*.deb
xhandSDK/.../hand_controller/       # 解包后的 Python 包目录
```

再构建真机镜像：

```bash
cd /home/user/rotate_dexgen/deploy_docker
./build.sh
```

产出镜像 `xhand-deploy:22.04`。

`build.sh` 会自动把 `xhandSDK/` 里带空格/`(1)` 的 `.deb` 和 `hand_controller/` 包目录规整进干净的
构建上下文,无需手动改名。

## 2. 真机未到 —— 先验证镜像本身没问题

```bash
# 进容器, 确认 SDK 绑定能 import(不需要硬件):
./run.sh
# 容器内:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 整条推理链离线 dry-run(用 FakeHW, 不需要硬件):
python -m rl_isaaclab.deploy_xhand.deploy_xhand --dry_run --max_steps 20
python -m pytest tests/deploy_xhand/ -q
```

只有正式镜像才包含厂商绑定，可额外检查：

```bash
python -c "import hand_controller; print([s for s in dir(hand_controller) if not s.startswith('_')])"
python scripts/verify_xhand_api.py
```

`run.sh` 在 `/dev/ttyUSB0` 不存在时会自动跳过设备透传,所以真机没到也能进容器。

### 两种进容器方式
- **`./run.sh [设备]`**: 快速、单容器、自动跳过不存在的串口。EtherCAT 真机应使用下面的 host 网络 + privileged 方式。
- **docker-compose**: host 网络 + privileged + 挂仓库，适合真机使用。
  ```bash
  docker compose -f deploy_docker/docker-compose.yml up -d
  docker exec -it xhand_deploy bash
  ```
  当前键盘监听直接读取终端 stdin，**不依赖 X11**；必须使用 `-it` 进入交互终端。

## 3. 真机到了 —— 进容器开始 Phase B

```bash
# 主机上先确认串口(插上手后):
ls /dev/ttyUSB*            # 或 /dev/ttyACM*
# 进容器并透传该串口:
./run.sh /dev/ttyUSB0
# 容器内即可跑 plan 的 B 阶段命令, 例如 B1:
python /usr/bin/xhand1pro_test.py -d /dev/ttyUSB0 -f get_version --hand-id 0
python /usr/bin/xhand1pro_test.py -d /dev/ttyUSB0 -f get_pos     --hand-id 0
python scripts/inspect_xhand_sdk.py        # 发现 SDK Python API, 回填 hw_interface 的 [B1]
```
`xhand1pro_test.py` 等 CLI 工具由 `.deb` 装在容器的 `/usr/bin/`。

最终上策略(对应 plan B8/B9):
```bash
# 默认开键盘: e=缓慢预抓取 Enter=启动策略 w=冻结/继续
# q=沿预抓取反向轨迹缓慢张开 t=触觉标定。
# Ctrl-C/STOP=先发送 mode0 松开舵机，再断连退出（会释放物体）。
python -m rl_isaaclab.deploy_xhand.deploy_xhand --action_scale_mul 0.3   # S2 低增益+限幅起步
python -m rl_isaaclab.deploy_xhand.deploy_xhand --action_scale_mul 1.0   # S3+ 正常
# 或无键盘自动跑(warm_up 先轻柔抓取 → 策略, Ctrl-C 急停):
python -m rl_isaaclab.deploy_xhand.deploy_xhand --no_keyboard --action_scale_mul 0.3
```
键盘状态机使用 stdin/cbreak，不需要 pynput 或 X11。没有交互 TTY 时才使用
`--no_keyboard`，但它会在预抓取后自动运行策略，真机首测不推荐。

---

## 注意事项

- **串口权限**: 容器默认以 root 运行,`--device` 透传后可直接读写 `/dev/ttyUSB0`,免去主机 dialout 组问题。
- **EtherCAT(二期才用)**: EtherCAT 是裸网卡实时,容器要加 `--network host --privileged`(或 `--cap-add=NET_ADMIN --cap-add=NET_RAW`)。一期串口不需要,故 `run.sh` 未加;改用 EtherCAT 时再说。
- **改了 deploy 代码**: 因仓库是挂载的,容器内立即生效,不用重建镜像。只有改了 SDK / torch 依赖才需重新 `./build.sh`。
- **权重/缓存**: 当前默认使用与 141-D/47-D 部署状态契约匹配的
  `ckpts/overnight_pose5/student_9402.pth`；旧 t603 是 186-D/62-D 合约，不能与当前
  `DeployCfg` 混用。关节表/暖启动轨迹(`cache/`)随仓库挂载进容器。
- 设计/计划详见 `docs/superpowers/specs/2026-06-04-...` 和 `docs/superpowers/plans/2026-06-04-...`。
