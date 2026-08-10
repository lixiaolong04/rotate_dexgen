# DexGen 项目交接与快速上手手册

> 适用对象：第一次接手 DexGen、需要复现实验或继续做 MANUS/XHand 部署的同事。  
> 最后核对：2026-08-10。项目根目录为 `/home/user/rotate_dexgen`（本机）或
> `/root/rotate_dexgen`（云端）。所有命令默认从项目根目录执行。

## 1. 接手后先记住的结论

1. DexGen 不是 RL 策略本身。它使用已经训练好的 CW/CCW RL 专家采集轨迹，再训练：
   - Diffusion：根据状态、方向和可选 guidance，预测未来两帧 10 个手部关键点运动；
   - IDM：根据状态和关键点运动，预测 21 维绝对 PD target。
2. 代码中的 `train_idm.py` 是 learned inverse dynamics model。项目交流中有时称它为
   “逆运动学”，但它不是传统解析 IK；交接时建议统一称为 **IDM/逆动力学模型**。
3. 当前唯一推荐的正式部署权重是 active-recovery v3：

   ```text
   dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
   dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
   ```

4. 正式闭环评估入口是 `dexgen/commands/eval/eval_dexgen.py`；
   `eval_guided.py`、`eval_dual_hand.py` 和 MANUS 脚本用于 guidance/交互测试。
5. RL 数据的方向和专家权重不能弄反：
   - CW：`ckpts/overnight_pose5/student_9402.pth`
   - CCW：`ckpts/overnight_ccw/student_9503.pth`
6. 数据、模型、触觉预处理和方向必须保持同一契约。78/83 维状态、
   `direct_v1/v2b/film_v3` 架构、`raw/frontpad` 传感器 profile 不能随意混搭。
7. 不要覆盖或批量同步两端的 `dexgen/data/`、`dexgen/checkpoints/`、
   `dexgen/logs/`、`dexgen/deploy/`。先确认目标文件，再增量传输。
8. 本次 GitHub 交付只保留当前 active-recovery v3 的 Diffusion/IDM、数据采集所需
   的 CW/CCW RL 专家和真机 8103 baseline；历史 DexGen 对照模型、数据集和训练/
   评估日志均已清理。接手人可直接用 v3 checkpoint 做第一次测试；需要继续训练时，
   按第 4 节重新采集数据。`dexgen/data/README.md` 和 `dexgen/logs/README.md` 是
   空目录的交付说明。

## 2. 系统架构

```text
RL 专家 rollout / MANUS 手套 command
                   ↓
10 个腕坐标系关键点（5 × PIP + 5 × 指尖胶垫）
                   ↓
Guided Diffusion：预测未来 0.1 s / 0.2 s 关键点 motion
                   ↓
IDM：输出 21 维绝对 PD target
                   ↓
Isaac Sim XHand 或真实 XHand
```

部署状态为 83 维：78 维本体状态加五指触觉标量。历史长度 4 帧，规划频率
10 Hz，底层任务频率 20 Hz。Diffusion 输出形状为两个未来时刻的 `[10, 3]`
motion；IDM 输出下一规划时刻的 21 维绝对关节目标。

纯 MANUS 关节遥操是另一条独立路径：

```text
MANUS 21 维 q_target → 关节重排/限位/变化限幅 → XHand PD target
```

它绕过 Diffusion、IDM 和触觉条件策略。

主要目录：

| 路径 | 用途 |
| --- | --- |
| `dexgen/commands/data/` | RL 轨迹采集、恢复数据审计 |
| `dexgen/commands/train/` | IDM、Diffusion 及实验分支训练 |
| `dexgen/commands/eval/` | 离线、仿真闭环、guided 评估 |
| `dexgen/pipelines/` | 可续跑的数据、训练、验证流水线 |
| `dexgen/deploy/` | XHand 真机硬件服务和控制器 |
| `manus_dexgen_bridge/` | MANUS TCP 协议、标定、探针和仿真入口 |
| `dexgen/data/` | 大型训练数据，不应无差别覆盖 |
| `dexgen/checkpoints/` | DexGen 权重，不应无差别覆盖 |
| `ckpts/` | 上游 RL teacher/student 专家权重 |
| `rl_isaaclab/` | IsaacLab 任务、RL 算法和 wrapper |

新入口放入对应的分类目录，不要在 `dexgen/` 根目录复制脚本或创建兼容软链接。

## 3. 环境安装

### 3.1 已验证的环境

云端训练机在 2026-08-10 的只读核对结果：

| 项目 | 云端已验证值 |
| --- | --- |
| SSH | `ssh -p 1178 root@180.76.108.184` |
| 项目 | `/root/rotate_dexgen` |
| Conda | `/home/ray/anaconda3/envs/env_isaaclab` |
| Python | 3.10.20 |
| Isaac Sim | 4.5.0.0 |
| IsaacLab | v2.2.0，commit `46dff135f`，源码 `/home/ray/IsaacLab` |
| PyTorch/CUDA wheel | 2.7.0+cu128 / CUDA 12.8 |
| GPU | 当时可见 2 × NVIDIA L20 46 GB |
| Driver | 535.230.02 |

云端 `dexgen/data/` 约 11 GB、`dexgen/checkpoints/` 约 12 GB。机器资源和可见
GPU 数量以后可能变化，开训前必须重新执行 `nvidia-smi` 和 `df -h`。

本机已有 `env_isaaclab`，项目总 README 记录的是 Python 3.11、Isaac Sim 5.1、
IsaacLab 2.3.2。它可用于本机调试，但**复现历史云端采集和训练时优先使用上表的
4.5/v2.2/Python 3.10 组合**，不要把两个环境的结果直接当作同一基线。

### 3.2 从零安装云端兼容环境

建议 Ubuntu 22.04、NVIDIA GPU、至少 32 GB RAM 和充足 SSD。先确认：

```bash
nvidia-smi
ldd --version
df -h
```

Isaac Sim 4.5 pip 包要求 Python 3.10 和 Linux GLIBC 2.34+。以下步骤与云端环境
对齐；如公司已有镜像或可克隆现有 conda 环境，应优先复用已验证镜像。

```bash
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
python -m pip install 'setuptools<81'

# Isaac Sim 4.5（首次下载和首次启动耗时较长）
python -m pip install 'isaacsim[all,extscache]==4.5.0' \
  --extra-index-url https://pypi.nvidia.com

# IsaacLab 固定到 v2.2.0
git clone --branch v2.2.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
sudo apt-get update
sudo apt-get install -y cmake build-essential
cd ~/IsaacLab
./isaaclab.sh --install

# 若 isaaclab 核心包未被安装，手动补 editable install
python -m pip install --no-build-isolation -e ~/IsaacLab/source/isaaclab
python -m pip install --no-build-isolation -e ~/IsaacLab/source/isaaclab_tasks
```

官方参考：
[Isaac Sim 4.5 pip 安装](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_python.html)、
[IsaacLab v2.2.0 文档](https://isaac-sim.github.io/IsaacLab/v2.2.0/)。

安装本仓库和额外运行依赖：

```bash
cd /path/to/rotate_dexgen
conda activate env_isaaclab
python -m pip install -e .
python -m pip install numpy scipy tensorboard tensorboardX termcolor pytest pynput
```

`pyproject.toml` 当前没有完整列出 DexGen 依赖，所以不能只依赖 `pip install -e .`
重建全部环境。完成后验证：

```bash
python - <<'PY'
import torch
import isaaclab
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('isaaclab:', getattr(isaaclab, '__version__', 'unknown'))
print('gpu available:', torch.cuda.is_available())
PY

python -m pytest tests/test_dexgen_pipeline.py -q
python -m pytest tests/test_manus_dexgen_bridge.py tests/test_dexgen_deploy.py -q
```

首次 Isaac Sim 启动可能需要联网下载扩展。服务器运行 headless 时可设置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

### 3.3 使用现有云端训练机

```bash
ssh -p 1178 root@180.76.108.184
cd /root/rotate_dexgen
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
nvidia-smi
```

建议用 `tmux` 保持长任务：

```bash
tmux new -s dexgen
# Ctrl-b d：退出但保持任务
tmux attach -t dexgen
```

开始前确认目标 GPU 没有冲突：

```bash
nvidia-smi
ps -ef | rg 'collect_rl_data|train_idm|train_diffusion|eval_dexgen'
df -h /root/rotate_dexgen
```

本机和云端增量传输示例，先 `--dry-run`：

```bash
# 上传单个源码文件
rsync -av --dry-run -e 'ssh -p 1178' \
  dexgen/commands/train/train_diffusion.py \
  root@180.76.108.184:/root/rotate_dexgen/dexgen/commands/train/

# 下载指定 checkpoint
rsync -av --dry-run -e 'ssh -p 1178' \
  root@180.76.108.184:/root/rotate_dexgen/dexgen/checkpoints/NAME_best.pt \
  dexgen/checkpoints/
```

确认列表无误后去掉 `--dry-run`。不要用会删除远端文件的 `--delete`，也不要直接
同步整个 `data/` 或 `checkpoints/`。SSH 地址可以写入本文档；密码、私钥和访问
token 不得提交到仓库。

## 4. RL 轨迹数据采集

> 仓库交付状态：历史数据集已清理，`dexgen/data/` 默认只有说明文件。以下目录和
> `.pt` 分片需要接手人按本节重新生成。当前模型 checkpoint 已保留，所以第一次
> 上手验证不要求先采集或训练。

### 4.1 数据流和正式数据组成

采集器 `dexgen/commands/data/collect_rl_data.py` 加载冻结的 RL 专家，在 IsaacLab
中保存 episode。主要字段包括关节状态、绝对 PD target、10 个关键点、五指触觉、
物体状态、reward、终止类型，以及恢复扰动标签。

active-recovery v3 的训练数据由两部分组成：

- `dexgen/data/large_modes_v5/`：稳定连续旋转主分布，每方向 10 个分片；
- `dexgen/data/recovery_modes_v6/`：带标注的主动扰动恢复数据，每方向 10 个分片。

恢复数据只能作为增强，不能完全替代稳定数据。采集器在扰动阶段不监督受扰动作，
随后记录干净 RL 专家的恢复动作。恢复成功要求物体未掉落、至少三指接触并继续沿
目标方向旋转。

### 4.2 推荐：可续跑的正式恢复数据采集

本机 8 GB 显存默认 50 env × 每 env 20 episode，每个分片共 1000 episode：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
./dexgen/pipelines/collect_recovery_dataset.sh all 0 9
```

只补部分分片：

```bash
./dexgen/pipelines/collect_recovery_dataset.sh cw 3 5
./dexgen/pipelines/collect_recovery_dataset.sh ccw 7 9
```

已有 `.pt` 分片会被审计后跳过，因此脚本可以续跑。显存更充足时可提高并行环境，
但保持每分片总 episode 数一致：

```bash
DEXGEN_NUM_ENVS=100 \
DEXGEN_EPISODES_PER_ENV=10 \
DEXGEN_DEVICE=cuda:0 \
./dexgen/pipelines/collect_recovery_dataset.sh all 0 9
```

云端脚本默认 Python 路径写的是本机 `/home/user/...`。在云端必须覆盖：

```bash
DEXGEN_PYTHON_BIN=/home/ray/anaconda3/envs/env_isaaclab/bin/python \
DEXGEN_NUM_ENVS=100 \
DEXGEN_EPISODES_PER_ENV=10 \
DEXGEN_DEVICE=cuda:0 \
./dexgen/pipelines/collect_recovery_dataset.sh all 0 9
```

另开终端查看进度：

```bash
watch -n 5 python dexgen/tools/monitor_recovery_progress.py
```

### 4.3 单次采集命令

需要新建普通 full-DR 数据分片或调试采集器时可直接运行：

```bash
python dexgen/commands/data/collect_rl_data.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 100 \
  --episodes_per_env 10 \
  --algorithm ProprioAdapt \
  --load_path ckpts/overnight_pose5/student_9402.pth \
  --output dexgen/data/EXPERIMENT/cw_full_00.pt \
  --mode cw \
  --seed 10000 \
  --sensor_profile raw \
  --domain_randomization full \
  --so3 --headless --device cuda:0
```

采集 CCW 时至少修改 `--mode ccw`、seed、输出名和专家权重：

```text
ckpts/overnight_ccw/student_9503.pth
```

`--so3` 时 `--num_envs` 必须能被 10 整除。正式恢复采集还应使用
`--recovery_perturbations`；优先使用上一节的 pipeline，避免漏参数。

### 4.4 数据验收

每个分片采集完成后立即审计；所有分片完成后做总审计：

```bash
python dexgen/commands/data/audit_recovery_data.py --data \
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt
```

重点检查：

- CW/CCW 的有向转速符号是否正确；
- episode 数量和各环境贡献是否均衡；
- 掉落与成功恢复比例是否异常；
- action bias、finger release、object force 类型是否都有样本；
- 有效训练窗口是否为零或显著偏少；
- `.progress.json` 是否停止更新、磁盘是否写满。

采集器带启动方向检查；方向不符会在保存前终止。不要为了得到文件而关闭检查后继续
训练，否则会把 CW/CCW 标签污染到同一个分布。

## 5. IDM（逆动力学）训练

推荐先训练 IDM，因为可以用验证集中的 oracle future motion 单独确认“关键点运动
→ PD target”是否可学。正式 active-recovery v3 命令：

```bash
conda activate env_isaaclab
cd /home/user/rotate_dexgen   # 云端改为 /root/rotate_dexgen

python dexgen/commands/train/train_idm.py --data \
  dexgen/data/large_modes_v5/cw_full_*.pt \
  dexgen/data/large_modes_v5/ccw_full_*.pt \
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt \
  --include_tactile --tactile_arch v2b \
  --explicit_recovery_only --recovery_fraction 0.25 \
  --tactile_aux_weight 0.005 \
  --batch_size 512 --epochs 15 \
  --output dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --latest_output dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_latest.pt \
  --device cuda:0
```

注意：

- `best.pt` 是验证集最优权重；`latest.pt` 包含完整训练状态，可用于续训；
- 新实验必须改输出文件名，禁止覆盖当前正式 v3；
- `--recovery_fraction 0.25` 表示每个方向采样中 25% 为明确成功恢复窗口；
- `--explicit_recovery_only` 排除旧版触觉启发式恢复标签；
- 续训使用 `--resume <..._latest.pt>`，同时保持数据和模型参数契约一致；
- 显存不足先减小 `--batch_size`，不要先改变状态/历史/触觉架构。

## 6. Diffusion 训练

IDM 通过 oracle-motion 验收后再训练 Diffusion。正式配置使用 epsilon prediction、
cosine noise schedule、100 diffusion steps：

```bash
python dexgen/commands/train/train_diffusion.py --data \
  dexgen/data/large_modes_v5/cw_full_*.pt \
  dexgen/data/large_modes_v5/ccw_full_*.pt \
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt \
  --include_tactile --tactile_arch v2b \
  --explicit_recovery_only --recovery_fraction 0.25 \
  --tactile_aux_weight 0.005 \
  --prediction_type epsilon --noise_schedule cosine \
  --batch_size 512 --epochs 15 \
  --output dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --latest_output dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_latest.pt \
  --device cuda:0
```

同样应为新实验换一个清晰文件名，例如包含数据版本、架构和 seed。Diffusion 与
IDM 必须有相同的 state dim、history、prediction steps、keypoint representation
和 tactile contract，否则评估器会拒绝加载。

云端建议在 `tmux` 中保存日志。历史训练日志不随项目交付，新实验开始前先创建
独立目录：

```bash
mkdir -p dexgen/logs/EXPERIMENT
python dexgen/commands/train/train_diffusion.py ... \
  2>&1 | tee dexgen/logs/EXPERIMENT/train_diffusion.log
```

日志文件默认不提交 GitHub；正式实验完成后把关键命令、配置和结论整理到
`dexgen/docs/` 或 `dexgen/worklog.md`。不要只凭训练 loss 选模型；必须继续做
离线串联和 Isaac 闭环验证。

## 7. 离线评估

离线评估不启动 Isaac Sim，先隔离 IDM，再检查 Diffusion → IDM 串联误差：

```bash
python dexgen/commands/eval/eval_offline.py --data \
  dexgen/data/large_modes_v5/cw_full_*.pt \
  dexgen/data/large_modes_v5/ccw_full_*.pt \
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --split val --ddim_steps 25 --batch_size 512 --device cuda:0
```

重点看：

- oracle motion → IDM 的 target MAE/RMSE；
- Diffusion motion MAE/RMSE、幅值比和相关性；
- Diffusion → IDM 的 target MAE；
- CW、CCW 两个 mode 是否一边明显退化；
- DDIM=8 与线上默认 DDIM=25 的速度/质量权衡。

## 8. 仿真推理与闭环评估

### 8.1 正式无 guidance 闭环

CW 示例：

```bash
python dexgen/commands/eval/eval_dexgen.py \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --mode cw --num_envs 16 --steps 1200 --ddim_steps 25 \
  --sensor_profile raw --so3 --device cuda:0
```

CCW 使用相同命令并改为 `--mode ccw`。正式结论不能只跑一个 seed；至少固定配置后
跑多个 environment seed 和 diffusion seed，并记录总 exposure。主要指标是：

- drop/env-minute 或 episode 掉落率；
- signed rotation speed 和 wrong-way；
- generated → executed keypoint RMSE；
- IDM target tracking；
- 主动扰动后的 recovery success。

主动恢复评估可加 `--recovery_perturbations`。正式对比必须保持 mode、seed、
Diffusion seed、DDIM steps、传感器 profile、steps 和总 exposure 一致。

### 8.2 noisy-RL/guided 交互对照

```bash
python dexgen/commands/eval/eval_guided.py \
  --controller noisy_rl --mode cw \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --sensor_profile raw --noise_alpha 0.1 \
  --steps 1200 --so3 --interactive_mode_switch --device cuda:0
```

按键：`c` CW、`v` CCW、`m` 切换方向、`i` 状态、`q` 安全退出。

双手参考/执行手交互入口：

```bash
python dexgen/commands/eval/eval_dual_hand.py \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --interactive_mode_switch --noise_alpha 0.1 \
  --max_command_motion_m 0.03 --guidance_strength 100000 \
  --command_time_mode separate --command_horizon_steps 1 \
  --command_second_horizon_steps 2 --ddim_steps 25 \
  --steps 3000 --device cuda:0
```

## 9. MANUS 遥操仿真推理

### 9.1 启动输入源

终端 1，在宿主机进入已有 `ros2_dev` 容器：

```bash
sudo docker exec ros2_dev supervisorctl restart manus_core_mini
sudo docker exec -it ros2_dev bash
```

容器内：

```bash
cd /root/xbot/algorithm/teleopx/retargetx-2.2.18
bash script/run_hand_manus_dexgen_source.sh
```

保持终端运行。链路为 MANUS → RetargetX → TCP 8020。不要同时启动 GeoRT、
MANUS ROS bridge 或 RetargetX visualizer；MANUS 8000 端口只能被一个接收端消费。

终端 2 先检查输入：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python manus_dexgen_bridge/command_probe.py --seconds 30
```

正常应看到约 80 Hz 输入和十个关键点的毫米级位移。

### 9.2 手动切换遥操

```bash
python manus_dexgen_bridge/eval_manus_guided.py \
  --manus_command_gain 0.55 \
  --guidance_strength 50000 \
  --device cuda:0
```

三条互斥控制路径：

- `h`：MANUS command → Diffusion → IDM；随后按 `a` 标定并 arm；
- `n`：noisy RL → Diffusion → IDM；
- `t`：MANUS 21 关节目标直接遥操，绕过 Diffusion/IDM。

`d` disarm 并保持安全目标，`c/v` 切方向，`i` 看状态，`q` 退出。8020 数据过期、
断连、reset 或关节契约失败时必须 disarm；不能把断连解释成静止手势。

### 9.3 自动 CW/CCW/HOLD

新操作者应重新录制个人 profile：

```bash
python manus_dexgen_bridge/calibrate_open_mode.py
python manus_dexgen_bridge/probe_open_mode.py
python manus_dexgen_bridge/calibrate_auto_mode.py
python manus_dexgen_bridge/probe_auto_mode.py
python manus_dexgen_bridge/probe_control_mode.py
```

profile 保存到：

```text
manus_dexgen_bridge/logs/manus_open_profile.pt
manus_dexgen_bridge/logs/manus_auto_mode_profile.pt
```

启动自动手势仿真：

```bash
python manus_dexgen_bridge/eval_manus_auto_mode.py \
  --manus_command_gain 0.55 \
  --guidance_strength 5000 \
  --max_command_motion_m 0.03 \
  --object_scale 0.5 \
  --device cuda:0
```

持续完全张掌 1 秒进入 HOLD：锁存执行手当前 PD target，并绕过 Diffusion/IDM，
不会让机器人跟随人的张手而松开。恢复自然抓握约 0.2 秒后恢复进入 HOLD 前的方向。

## 10. XHand 真机部署推理

### 10.1 安全条件

- XHand 独立供电，EtherCAT 网线连接本机 `enp129s0`；
- 手和物体下方必须放承接平台；
- 不同时启动两个 SDK 服务或两个控制器；
- 固定硬件参数：`mode=4, kp=10, kd=0.3, speed=0, torque=0`；
- 任一关节达到 68°C 或出现厂商错误码会自动 mode 0；
- 首次操作必须分阶段验证，不要直接长时间旋转。

### 10.2 三终端启动

终端 1：按第 9.1 节启动 MANUS → RetargetX → TCP 8020。

终端 2：启动 Docker 内的 XHand SDK 服务：

```bash
cd /home/user/rotate_dexgen
bash dexgen/deploy/run_xhand_sdk_server.sh
```

看到以下日志后再启动控制器：

```text
[hardware] listening on tcp://127.0.0.1:8031
```

终端 3，二选一：

```bash
# 手动切换 MANUS/DexGen
bash dexgen/deploy/run_manus_dexgen_real.sh

# 手势自动 CW/CCW/HOLD
bash dexgen/deploy/run_manus_auto_real.sh --guidance-strength 1000
```

推荐操作顺序：

1. 按 `i`，确认 MANUS 为 `recording`、硬件连接和温度正常；
2. 放好圆柱，按 `e` 执行 8103 baseline 预抓取；
3. 按 `Enter` 启动 DexGen；
4. 先运行 3–5 秒，按 `w` 测试暂停/HOLD，再次按 `w` 恢复；
5. 按 `q` 沿安全轨迹缓慢张开并释放物体；
6. `Ctrl-C` 退出，确认 SDK 终端显示 mode 0 和 EtherCAT disconnected。

关键按键：

| 按键 | 功能 |
| --- | --- |
| `e` | 缓慢张开、触觉标定、接触即停预抓取 |
| `Enter` / `n` | 启动 DexGen |
| `t` | 纯 MANUS 21 关节遥操 |
| `h` 后 `a` | 标定并启动 MANUS guidance → DexGen |
| `w` | 手动暂停并 HOLD，再按恢复 |
| `d` | 停止输入并 HOLD |
| `q` | 沿安全轨迹缓慢张开回 home |
| `c` / `v` / `m` | 选择或切换旋转方向 |
| `i` | 显示输入源、方向、MANUS 状态和最高温度 |
| `x` / `Ctrl-C` | mode 0、保存日志并退出 |

详细硬件说明见 `dexgen/deploy/README.md` 和
`dexgen/deploy/README_XHAND_HARDWARE.md`。真机控制不要在不了解后果时调整 kp/kd、
关节限幅、温度保护或退出轨迹。

## 11. 标准实验验收顺序

新数据或模型建议严格按以下顺序推进：

1. 对每个新数据分片运行审计；
2. 运行 `tests/test_dexgen_pipeline.py`；
3. 训练 IDM，检查验证集 oracle-motion target MAE；
4. 训练 Diffusion，做离线 Diffusion → IDM 串联评估；
5. Isaac headless 短 smoke test；
6. CW/CCW、多 seed、固定 exposure 的正式闭环评估；
7. 与 clean RL、noisy RL、当前 v3 做公平对照；
8. MANUS 30 秒输入探针和仿真 guidance 测试；
9. 真机分阶段测试：连接 → 预抓取 → 3–5 秒旋转 → HOLD → 安全松手；
10. 保存命令、代码版本、数据清单、checkpoint、seed 和日志摘要。

当前 v3 的参考结果：主动恢复成功率 73.68%，episode 掉落率约 55%，CW/CCW
转速约 0.8 rad/s。v5/v6/v7 均未超过 v3，因此实验结果没有明确胜出前不要替换默认
部署模型。

## 12. 常见问题

### Isaac Sim 启动失败或 import 顺序报错

Isaac 应用必须在大部分 Isaac/Omniverse 模块之前创建；项目入口已经按该顺序组织，
不要随意把 Isaac import 移到 `AppLauncher` 之前。先确认 Python、Isaac Sim 和
IsaacLab 版本属于同一套环境。

### CUDA OOM

- 采集：减少 `DEXGEN_NUM_ENVS`，并相应增加 `DEXGEN_EPISODES_PER_ENV` 保持总量；
- 训练：先减小 batch size；
- 评估：减少 `--num_envs` 或 risk candidates；
- 检查是否有其他 Isaac/训练进程占用 GPU。

### Diffusion 和 IDM 无法一起加载

通常是 state dim、触觉架构、history/prediction steps、关键点语义或数据 normalizer
不一致。不要绕过 checkpoint 合约检查，应重新选择同一实验生成的成对权重。

### MANUS 没有数据或数据 stale

确认 `ros2_dev`、`manus_core_mini`、RetargetX source 脚本和 TCP 8020；停止其他会
消费 MANUS 8000 端口的程序，然后运行 `command_probe.py --seconds 30`。链路恢复后
需要重新 arm/标定，不能自动把旧帧继续用于控制。

### 真机不动、猛动或异常发热

立即停止并 mode 0。检查 EtherCAT、`mode=4`、`kp=10`、`kd=0.3`、`speed=0`、
关节顺序、目标限幅和温度。不要在物体下方无承接平台时退出或断电。

## 13. 离职交接清单

建议交接人和接手人一起完成以下确认：

- [ ] 接手人能进入本地仓库和云端训练机；
- [ ] SSH 权限通过公司安全流程移交，没有密码/私钥写入 Git；
- [ ] 当前模型 checkpoint 清单和用途已确认；历史数据及日志不随仓库交付；
- [ ] 接手人知道数据需要重新生成，并理解 CW/CCW 专家不能互换；
- [ ] 接手人能激活 `env_isaaclab` 并通过核心 pytest；
- [ ] 接手人能运行一次 RL 数据采集 smoke test 和数据审计；
- [ ] 接手人能启动一次 IDM/Diffusion 小规模训练并正确续训；
- [ ] 接手人能完成离线评估和 Isaac 闭环推理；
- [ ] 接手人能完成 MANUS 30 秒 probe、个人手势 profile 标定；
- [ ] 接手人理解真机三终端启动、预抓取、HOLD、安全松手和 mode 0；
- [ ] 当前推荐 v3 权重及其参考结果已确认；
- [ ] 未完成实验、失败原因和下一步建议补充到 `dexgen/worklog.md`。

## 14. 继续阅读

- `dexgen/README.md`：日常启动命令和按键；
- `dexgen/docs/DEVELOPMENT.md`：训练设计与数据契约；
- `dexgen/docs/PROJECT_LAYOUT.md`：目录和本机/云端迁移约定；
- `dexgen/docs/ACTIVE_RECOVERY_V3_RESULTS.md`：当前正式基线结果；
- `dexgen/worklog.md`：实验历史、失败分支和后续计划；
- `manus_dexgen_bridge/README.md`：MANUS 标定、探针与自动模式；
- `dexgen/deploy/README.md`：XHand 真机启动和安全操作；
- 项目根目录 `README.md`：上游 RL teacher/student 与原始 sim2real 流程。
