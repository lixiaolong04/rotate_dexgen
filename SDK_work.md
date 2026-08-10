# RobotEra 本地 SDK、仿真与 Manus → Green 灵巧手遥操作工作交接

**交接日期：** 2026-07-20  
**目标：** 在 Ubuntu 本机通过 Docker 部署 RobotEra/XBot SDK，先运行 Green 灵巧手仿真，后续接入 Manus 手套完成遥操作。  
**当前阶段：** Docker、GPU、公司镜像、SDK 和基础容器已部署完成；Manus/TeleopX/RetargetX 组件及仿真启动链路尚待核查。

---

## 1. 最终目标

期望链路如下：

```text
Manus 手套
    ↓
manus_core_mini（手套数据接入）
    ↓
retargetx（人手/人体动作重定向）
    ↓
retargetx_combind（组合控制）
    ↓
xbot_sdk / robot_control
    ↓
l3.4 + Green 灵巧手仿真
```

目前没有真机，只做仿真。任何真机抱闸、使能、关节控制脚本都不要运行。

---

## 2. 宿主机环境

已确认：

- 原生 Ubuntu。
- 主机：Lenovo Legion Y9000P IAX10。
- NVIDIA GPU：RTX 5060 系列，约 8 GB 显存。
- NVIDIA 驱动：`580.159.03`。
- 宿主机 `nvidia-smi` 正常。
- Docker 已安装并可由当前用户直接运行。
- NVIDIA Container Toolkit 已安装并注册到 Docker。
- 当前 Conda 环境可能显示为 `(env_isaaclab)`，但 Conda 与 Docker 部署互不影响。

Docker Runtime 已确认：

```text
Runtimes: io.containerd.runc.v2 nvidia runc
Default Runtime: runc
```

`Default Runtime` 保持 `runc` 是正常的，关键是 `Runtimes` 中存在 `nvidia`。

容器 GPU 已确认：

```bash
docker exec -it ros2_dev nvidia-smi
```

能够正常显示 GPU 信息。

---

## 3. 公司 Docker Registry

Docker 已配置以下 insecure registry：

```text
10.128.114.9:5000
reg.robotera.com:5000
```

配置文件：

```text
/etc/docker/daemon.json
```

### 已发现的重要网络现象

直接访问 IP：

```bash
curl http://10.128.114.9:5000/v2/
```

会得到：

```text
Empty reply from server
```

但使用公司域名：

```bash
curl http://reg.robotera.com:5000/v2/
```

返回：

```text
HTTP/1.1 200 OK
{}
```

给 IP 请求手动添加 Host 头也能成功：

```bash
curl -H 'Host: reg.robotera.com:5000' \
  http://10.128.114.9:5000/v2/
```

因此后续拉取镜像应优先使用：

```text
reg.robotera.com:5000
```

不要优先使用裸 IP。

---

## 4. 公司 Docker 镜像

已成功拉取：

```text
reg.robotera.com:5000/ros-humble-nuc:latest
```

镜像摘要：

```text
sha256:8fdb846e6def0243677f38e6e578c4fb27aa15110ecab02730cd0ec6eecdfe67
```

本地标签：

```text
reg.robotera.com:5000/ros-humble-nuc:latest
10.128.114.9:5000/ros-humble-nuc:latest
ros-humble-nuc:latest
```

三者当前指向同一个 IMAGE ID：

```text
8fdb846e6def
```

---

## 5. SDK 下载与解压

下载的是 x86_64 版本：

```text
xbot_robot_sdk_dev_2.0.136.tar
```

没有下载 `aarch64` 版本。

压缩包大小约：

```text
339 MB
```

宿主机文件位置：

```text
~/xbot_robot_sdk_dev_2.0.136.tar
```

SDK 已解压到：

```text
/opt/xbot
```

实际目录结构：

```text
/opt/xbot/
└── hardware/
    ├── config.json
    ├── install/
    │   ├── setup.bash
    │   └── scripts/
    │       ├── setup_env.sh
    │       ├── start_robot.sh
    │       ├── sdk.sh
    │       └── ...
    ├── robot_config/
    └── supervisor_config/
        └── supervisord.conf
```

日志目录已创建：

```text
/system_log/log/xbot_control
```

---

## 6. Docker 容器

正式容器名称：

```text
ros2_dev
```

当前容器 ID（可能因重建而变化）：

```text
a8f6b881185d
```

当前状态已确认是：

```text
Up
```

创建容器时使用的核心配置：

```bash
docker run -d \
  --name ros2_dev \
  --restart=always \
  --privileged \
  --network=host \
  --security-opt seccomp=unconfined \
  --gpus all \
  --cap-add=sys_nice \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /opt/xbot:/root/xbot \
  -v /opt/xbot/hardware/supervisor_config:/etc/supervisor/conf.d \
  -v /system_log:/system_log \
  -v /dev:/dev \
  ros-humble-nuc:latest
```

宿主机图形环境：

```text
DISPLAY=:1
```

已执行：

```bash
xhost +local:root
```

进入容器建议使用：

```bash
docker exec -it ros2_dev /bin/bash
```

也可以创建或继续使用：

```text
~/go_to_docker.sh
```

---

## 7. ROS_DOMAIN_ID

当前使用：

```text
ROS_DOMAIN_ID=42
```

已确认写入容器：

```text
/root/.bashrc
```

对应内容：

```bash
export ROS_DOMAIN_ID=42
```

已确认 `/root/xbot/hardware/supervisor_config/supervisord.conf` 中所有相关程序的：

```text
ROS_DOMAIN_ID="42"
```

均已替换成功。

禁止使用：

```text
211
```

---

## 8. setup_env.sh

已在容器内执行：

```bash
cd /root/xbot/hardware/install/scripts
./setup_env.sh
```

输出：

```text
[INFO] 复制脚本文件到 /root...
[INFO] 设置脚本文件补全...
[INFO] 复制配置文件...
[INFO] 目标配置已存在，合并缺失键...
```

返回码：

```text
0
```

说明基础环境部署脚本执行成功。

脚本生成或维护了：

```text
/root/xbot/config.json
```

同时 SDK 模板仍位于：

```text
/root/xbot/hardware/config.json
```

后续运行优先使用：

```text
/root/xbot/config.json
```

---

## 9. 已发现的主启动入口

Supervisor 中的核心程序：

```ini
[program:process_manager]
command=/bin/bash -c "source /root/.bashrc && source /opt/ros/humble/setup.bash && source /root/xbot/hardware/install/setup.bash && ros2 run process_manager process_manager_node"

[program:xbot_sdk]
command=/bin/bash -c "source /root/xbot/hardware/install/setup.bash && ros2 launch robot_control xbot.launch.py"
```

因此 XBot SDK 主启动命令是：

```bash
source /opt/ros/humble/setup.bash
source /root/xbot/hardware/install/setup.bash
ros2 launch robot_control xbot.launch.py
```

建议首次测试以前台方式运行，便于完整看到报错；不要一开始就隐藏到 Supervisor 后台。

---

## 10. Supervisor 中发现的遥操作相关程序

已在 `supervisord.conf` 中发现：

```text
process_manager
xbot_sdk
sandbox_launch
xhand_logistics_monitor
license_verify
rb_ros
image_bridge
webxr
signaling
retargetx
retargetx_combind
mpc
battery
xbot_record
xbot_record_http_server
foxglove_bridge
xviz_web
manus_core_mini
xbot_diagnostic_node
system_monitor
```

与 Manus 遥操作最相关的启动命令：

```ini
[program:manus_core_mini]
command=/bin/bash -lc 'source /root/xbot/hardware/install/setup.bash && ros2 run manus_core_mini manus_core_mini_node'
```

```ini
[program:retargetx]
command=/bin/bash -lc "source /root/.bashrc && source /opt/ros/humble/setup.bash && source /root/xbot/hardware/install/setup.bash && export PYTHONPATH=/opt/conda/envs/teleop/lib/python3.10/site-packages:$PYTHONPATH && /root/xbot/hardware/install/process_manager/lib/process_manager/supervisor_args_wrapper.sh --name retargetx -- /opt/conda/envs/teleop/bin/python /root/xbot/algorithm/teleopx/retargetx/tele_node.py"
```

```ini
[program:retargetx_combind]
command=/bin/bash -lc "source /root/.bashrc && source /opt/ros/humble/setup.bash && source /root/xbot/hardware/install/setup.bash && export PYTHONPATH=/opt/conda/envs/teleop/lib/python3.10/site-packages:$PYTHONPATH && /root/xbot/hardware/install/process_manager/lib/process_manager/supervisor_args_wrapper.sh --name retargetx_combind -- /opt/conda/envs/teleop/bin/python /root/xbot/algorithm/teleopx/retargetx/combine_tele_controller.py"
```

这说明公司系统已经定义了 Manus、RetargetX 和组合遥操作链路，但目前尚未确认这些程序文件是否实际存在。

---

## 11. 当前配置状态：必须重新核对

最后一次实际打印出的 `/root/xbot/config.json` 仍是：

```json
{
  "hand_type": "xhand",
  "hardware_type": "no_hand",
  "robot_type": "l3.4",
  "auto_start": false,
  "launch_mode": "pos",
  "mode": "position",
  "l3_mode": "pd",
  "use_mpc_controller": false,
  "use_rl_controller": false,
  "enable_developer": false,
  "lift_joint_mode": "position",
  "head_display_mode": "false"
}
```

之后提出了切换 Green 手和仿真的需求，但没有留下“修改成功后的终端输出”。

因此 Codex **不要假设配置已完成**，必须首先检查：

```bash
cat /root/xbot/config.json
```

目标配置应至少是：

```json
{
  "auto_start": false,
  "enable_developer": true,
  "hand_type": "green",
  "hardware_type": "hand",
  "robot_type": "l3.4",
  "launch_mode": "sim",
  "mode": "sim",
  "l3_mode": "pd",
  "lift_joint_mode": "position",
  "use_mpc_controller": false,
  "use_rl_controller": false,
  "head_display_mode": "false"
}
```

`green` 与 `l3.4` 是文档列出的兼容组合。

建议使用 Python 安全修改，而不是手改 JSON：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/root/xbot/config.json")
backup = Path("/root/xbot/config.json.before-green-sim")

if not backup.exists():
    backup.write_bytes(path.read_bytes())

cfg = json.loads(path.read_text(encoding="utf-8"))
cfg.update({
    "auto_start": False,
    "enable_developer": True,
    "hand_type": "green",
    "hardware_type": "hand",
    "robot_type": "l3.4",
    "launch_mode": "sim",
    "mode": "sim",
    "l3_mode": "pd",
    "lift_joint_mode": "position",
    "use_mpc_controller": False,
    "use_rl_controller": False,
})

path.write_text(
    json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

print(path.read_text(encoding="utf-8"))
PY
```

---

## 12. 当前最关键的潜在问题：挂载遮蔽镜像内容

容器创建时使用：

```bash
-v /opt/xbot:/root/xbot
```

宿主机 `/opt/xbot` 当前已确认主要只有：

```text
hardware/
```

这会将镜像原本的 `/root/xbot` 整个遮住。

如果公司镜像内原本包含：

```text
/root/xbot/algorithm
/root/xbot/sandbox
/root/xbot/xviz-web
```

这些目录在当前容器中也可能因为挂载而不可见。

这很可能是后续 Manus/TeleopX 无法启动的核心原因之一。

### Codex 必须先对比“镜像原始内容”和“当前容器内容”

当前容器：

```bash
docker exec ros2_dev bash -lc '
  echo "===== mounted container ====="
  find /root/xbot -maxdepth 3 -type d 2>/dev/null | sort | head -200
'
```

不挂载 SDK 的临时镜像探针：

```bash
docker run --rm \
  --entrypoint bash \
  ros-humble-nuc:latest \
  -lc '
    echo "===== original image ====="
    find /root/xbot -maxdepth 3 -type d 2>/dev/null | sort | head -200
  '
```

重点比较是否存在：

```text
/root/xbot/algorithm/teleopx
/root/xbot/algorithm/teleopx/retargetx
/root/xbot/algorithm/teleopx/rboperation
/root/xbot/sandbox
/root/xbot/xviz-web
```

在没有确认前，不要随意重建或删除容器，也不要直接覆盖 `/opt/xbot`。

---

## 13. Codex 下一步任务清单

### 阶段 A：只读诊断，不改环境

1. 检查当前容器、镜像、挂载和 GPU。
2. 检查 `/root/xbot/config.json` 当前真实内容。
3. 对比镜像原始 `/root/xbot` 与挂载后的 `/root/xbot`。
4. 检查以下组件是否存在：

```bash
ros2 pkg prefix manus_core_mini
ls -ld /opt/conda/envs/teleop
ls -ld /root/xbot/algorithm/teleopx
ls -ld /root/xbot/algorithm/teleopx/retargetx
ls -ld /root/xbot/algorithm/teleopx/rboperation
ls -ld /root/xbot/sandbox
ls -ld /root/xbot/xviz-web
```

5. 查找 Manus 和 Green 手配置：

```bash
grep -RniE \
  'manus|green|retarget|glove|skeleton|finger|hand_type' \
  /root/xbot/hardware/install/share \
  /root/xbot/algorithm \
  /etc/supervisor \
  2>/dev/null | head -300
```

6. 读取以下启动文件及参数：

```bash
ros2 pkg prefix manus_core_mini
ros2 pkg executables manus_core_mini
ros2 pkg prefix robot_control
```

```bash
sed -n '1,260p' \
  "$(ros2 pkg prefix robot_control)/share/robot_control/launch/xbot.launch.py"
```

7. 输出诊断结论，不要先启动控制链。

### 阶段 B：修正 Green 仿真配置

确认备份后，将：

```text
hand_type=green
hardware_type=hand
robot_type=l3.4
mode=sim
launch_mode=sim
enable_developer=true
auto_start=false
```

写入 `/root/xbot/config.json`。

然后验证：

```bash
python3 -m json.tool /root/xbot/config.json
```

### 阶段 C：前台验证基础仿真

先不要使用 Supervisor，前台执行：

```bash
source /opt/ros/humble/setup.bash
source /root/xbot/hardware/install/setup.bash
ros2 launch robot_control xbot.launch.py
```

记录完整日志，重点判断：

- 是否启动仿真器；
- 是否缺少 RL 算法；
- 是否缺少机器人模型；
- 是否缺少 license；
- 是否只启动控制节点，没有图形界面；
- 是否出现 Green 手相关节点；
- 是否存在 `/joint_states`、`/clock`、Green 手命令话题或服务。

公司文档指出 SDK `>= 2.0.110` 时，双足机器人仿真需要先部署 RL 算法。当前版本是 `2.0.136`，因此完整双足仿真可能还缺 RL 包。需要区分“完整双足仿真”和“仅 Green 手仿真”是否都受该限制。

### 阶段 D：验证 Manus 数据输入

组件存在后，再启动或前台运行：

```bash
source /opt/ros/humble/setup.bash
source /root/xbot/hardware/install/setup.bash
ros2 run manus_core_mini manus_core_mini_node
```

检查：

```bash
ros2 node list | grep -Ei 'manus|hand|retarget'
ros2 topic list -t | grep -Ei 'manus|glove|skeleton|finger|hand|retarget'
```

查看节点接口：

```bash
for node in $(ros2 node list | grep -Ei 'manus|retarget'); do
  echo "===== $node ====="
  ros2 node info "$node"
done
```

确定：

- Manus 服务在 Linux 本机还是 Windows 主机；
- 使用 USB、局域网还是本机进程；
- Manus SDK/服务版本；
- IP 和端口；
- 左右手手套 ID；
- 校准流程；
- 输出话题、消息类型和数据频率。

### 阶段 E：验证 RetargetX 与 Green 映射

检查：

```text
/root/xbot/algorithm/teleopx/retargetx/tele_node.py
/root/xbot/algorithm/teleopx/retargetx/combine_tele_controller.py
```

查找参数文件和 Green 手映射：

```bash
grep -RniE \
  'green|xhand|manus|joint_names|joint_limit|mapping|retarget' \
  /root/xbot/algorithm/teleopx/retargetx \
  2>/dev/null | head -300
```

需要确认：

- Manus 骨骼关节顺序；
- Green 手关节名称和顺序；
- 自由度数量；
- 左右手坐标系；
- 关节范围；
- 静态零位；
- 关节缩放；
- 低通滤波；
- 异常值和丢帧处理；
- 遥操作启停条件。

在这些内容未确认前，不要直接向控制话题发送关节数组。

### 阶段 F：组件齐全后按顺序启动

建议顺序：

```bash
supervisorctl start process_manager
supervisorctl start xbot_sdk
supervisorctl start manus_core_mini
supervisorctl start retargetx
supervisorctl start retargetx_combind
```

每一步都要检查：

```bash
supervisorctl status <name>
supervisorctl tail -100 <name>
```

不要一次全部启动，否则难以定位故障。

---

## 14. 禁止和注意事项

当前无真机，严禁执行或自动执行：

```text
open_brake_all.sh
close_brake_all.sh
open_brake_*.sh
close_brake_*.sh
activate.sh
ready.sh
set_joint_position.sh
zero.sh
start_robot.sh
```

除非已经确认脚本在 `mode=sim` 下完全不会触发真实硬件。

不要做以下操作：

- 不要重新安装 NVIDIA 驱动。
- 不要重新安装 Docker。
- 不要删除 `/opt/xbot`。
- 不要删除 `ros-humble-nuc:latest`。
- 不要删除 `ros2_dev`，除非先备份并解释原因。
- 不要改变 `ROS_DOMAIN_ID=42`，除非确认与其他设备冲突。
- 不要使用 `ROS_DOMAIN_ID=211`。
- 不要假设 Manus/TeleopX 已经安装。
- 不要在未知消息类型和关节顺序时发布控制指令。
- 不要默认启动 MPC、battery 或真机诊断服务。

---

## 15. 常用命令

宿主机：

```bash
docker ps -a --filter name=ros2_dev
docker start ros2_dev
docker stop ros2_dev
docker restart ros2_dev
docker logs --tail=200 ros2_dev
docker exec -it ros2_dev /bin/bash
docker exec -it ros2_dev nvidia-smi
```

容器内：

```bash
source /opt/ros/humble/setup.bash
source /root/xbot/hardware/install/setup.bash
echo "$ROS_DOMAIN_ID"
python3 -m json.tool /root/xbot/config.json
supervisorctl status
ros2 node list
ros2 topic list -t
ros2 service list
```

---

## 16. 给 Codex 的首轮提示词

将下面整段复制给 Codex：

```text
你正在接手一个 RobotEra/XBot 本地 SDK、Green 灵巧手仿真和 Manus 手套遥操作项目。

当前已完成：
1. Ubuntu 宿主机上的 Docker 已安装。
2. NVIDIA Container Toolkit 已安装并注册，Docker Runtime 中存在 nvidia。
3. ros-humble-nuc:latest 已从 reg.robotera.com:5000 拉取，IMAGE ID 为 8fdb846e6def。
4. xbot_robot_sdk_dev_2.0.136.tar（x86_64）已解压到宿主机 /opt/xbot。
5. ros2_dev 容器正在运行，使用 host 网络、GPU、X11、/dev，并将 /opt/xbot 挂载到 /root/xbot。
6. 容器内 nvidia-smi 正常。
7. ROS_DOMAIN_ID=42，已写入 /root/.bashrc 和 supervisord.conf。
8. /root/xbot/hardware/install/scripts/setup_env.sh 已执行成功，返回码 0。
9. 主 SDK 启动入口为 ros2 launch robot_control xbot.launch.py。
10. supervisord.conf 中存在 manus_core_mini、retargetx、retargetx_combind、webxr、signaling 等程序定义。

重要未知项：
1. 最后一次实际打印的 /root/xbot/config.json 仍是 xhand、no_hand、position 模式；需要先核实是否已经切换为 l3.4 + green + sim。
2. 宿主机 /opt/xbot 主要只有 hardware 目录，而容器用了 /opt/xbot:/root/xbot，可能遮蔽镜像原本的 algorithm/teleopx、sandbox、xviz-web 等目录。
3. 尚未确认 manus_core_mini、/opt/conda/envs/teleop、TeleopX、RetargetX 和 Green 映射配置是否真实存在。
4. 尚未前台启动 xbot.launch.py 验证仿真。
5. SDK 2.0.136 可能要求额外 RL 算法才能运行完整双足仿真。
6. 尚未确认 Manus 手套的连接方式、SDK 版本、网络端口、话题和校准方法。

你的首轮任务必须只做只读诊断：
- 不要重装 Docker/NVIDIA。
- 不要删除容器、镜像或 /opt/xbot。
- 不要运行抱闸、使能、真机关节控制脚本。
- 不要发布未知的 ROS 控制话题。
- 先输出当前环境、配置、挂载、原始镜像内容、TeleopX/Manus 组件存在性和启动文件分析。
- 对比不挂载时镜像内的 /root/xbot 与当前容器内的 /root/xbot，确认是否发生挂载遮蔽。
- 给出下一步最小风险操作，不要一次启动全部 Supervisor 服务。
- 所有修改前先备份，所有命令给出预期输出和回滚方法。

目标配置：
auto_start=false
enable_developer=true
hand_type=green
hardware_type=hand
robot_type=l3.4
launch_mode=sim
mode=sim
use_mpc_controller=false
use_rl_controller=false

最终目标：
Manus → manus_core_mini → retargetx → retargetx_combind → xbot_sdk → Green 灵巧手仿真。
```

---

## 17. 建议 Codex 首先运行的诊断命令

在宿主机执行：

```bash
echo "===== Docker ====="
docker ps -a --filter name=ros2_dev
docker inspect ros2_dev \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'

echo "===== Current mounted container ====="
docker exec ros2_dev bash -lc '
  echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
  echo "===== config ====="
  cat /root/xbot/config.json 2>/dev/null || true
  echo "===== xbot dirs ====="
  find /root/xbot -maxdepth 3 -type d 2>/dev/null | sort | head -250
  echo "===== components ====="
  source /opt/ros/humble/setup.bash
  source /root/xbot/hardware/install/setup.bash
  ros2 pkg prefix manus_core_mini 2>&1 || true
  ls -ld /opt/conda/envs/teleop 2>&1 || true
  ls -ld /root/xbot/algorithm/teleopx 2>&1 || true
  ls -ld /root/xbot/algorithm/teleopx/retargetx 2>&1 || true
  ls -ld /root/xbot/algorithm/teleopx/rboperation 2>&1 || true
  ls -ld /root/xbot/sandbox 2>&1 || true
  ls -ld /root/xbot/xviz-web 2>&1 || true
'

echo "===== Original image without host mount ====="
docker run --rm \
  --entrypoint bash \
  ros-humble-nuc:latest \
  -lc '
    find /root/xbot -maxdepth 3 -type d 2>/dev/null | sort | head -250
    echo "===== /opt/conda/envs ====="
    ls -ld /opt/conda/envs/teleop 2>&1 || true
  '
```

完成诊断后，再决定是：

1. 当前镜像已经包含 TeleopX，只是被挂载遮蔽；
2. 需要从镜像安全复制算法目录到 `/opt/xbot`；
3. 需要从 Jenkins 或公司同事获取额外算法包；
4. 需要额外部署 RL 仿真包；
5. Manus 接入依赖独立 Windows 主机或 Manus 官方运行时。

---

## 18. 当前进度结论

**已完成：**

- Docker 基础环境；
- NVIDIA GPU 容器支持；
- 公司 Registry 配置；
- 公司 ROS 2 镜像下载；
- SDK 2.0.136 下载和解压；
- 正式容器创建；
- SDK 挂载；
- ROS_DOMAIN_ID；
- `setup_env.sh`；
- 主启动入口和遥操作 Supervisor 程序识别。

**未完成：**

- Green 仿真配置的最终确认；
- 仿真前台启动验证；
- RL 仿真包确认；
- Manus 手套连接与校准；
- `manus_core_mini` 数据验证；
- TeleopX/RetargetX 文件存在性确认；
- Manus → Green 关节映射；
- 遥操作链路启动；
- 仿真中的 Green 手动作验证。

**下一步最高优先级：**

1. 检查挂载遮蔽；
2. 核实 Green + sim 配置；
3. 核实 Manus/TeleopX/RetargetX 组件；
4. 前台启动基础仿真并保存日志；
5. 再逐级接入 Manus 和 RetargetX。
