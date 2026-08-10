# Rotate DexGen 工作日志

## 2026-07-15：XHand Isaac Sim 仿真预览与云端训练前准备

### 今日目标

- 检查项目的 Isaac Sim / IsaacLab 启动入口。
- 使用 `ckpts` 中的 student 权重在仿真中运行 XHand 旋转物体。
- 支持 SO(3) 随机化手和物体的整体初始方向。
- 整理后续上传云服务器和继续训练时的注意事项。

### 项目与权重确认

- XHand 仿真任务：`Isaac-Inhand-Rotate-Xhand-Wave-v0`
- 仿真预览入口：`rl_isaaclab/scripts/play.py`
- 当前仿真配置：`rl_isaaclab/tasks/inhand_rotate/xhand_wave_env_cfg.py`
- 当前仿真推荐权重：`ckpts/frontpad_students_20260701/student_8500.pth`
- 8500 是 ProprioAdapt student，使用 141 维观测：
  - 单帧 47 维 = 21 关节角 + 21 目标角 + 5 指触觉。
  - 3 帧观测为 141 维。
  - student 本体历史输入为 `(30, 47)`。
- 8500 对应 option C front-pad 触觉几何过滤：
  - `enable_contact_force_vec = False`
  - `disable_tactile_ids = []`
  - `frontpad_filter = True`
- 对应抓取缓存：`cache/xhand_grasp_linspace_0.4-0.7-10.npy`

### 问题 1：加载权重时 186 维与 141 维不匹配

错误摘要：

```text
[startup-check] OK: num_joints=21, action_space=21, observation_space=186
RunningMeanStd: (186,)
RunningMeanStd: (30, 62)
size mismatch: checkpoint running_mean shape [141], current model shape [186]
```

根因：

- 在 `/home/user/rotate_dexgen` 中启动了 `play.py`，但 conda 环境里原有的 editable 安装仍指向旧仓库 `/home/user/era-dexgen`。
- 运行时因此混用了两个仓库的源码。旧仓库配置为 186 维观测，8500 权重是 141 维，导致 `RunningMeanStd.load_state_dict()` 尺寸不匹配。
- traceback 中出现 `/home/user/era-dexgen/rl_isaaclab/...` 是确认混用源码的关键证据。

处理：

```bash
conda activate env_isaaclab
cd /home/user/rotate_dexgen
python -m pip install -e .
```

验证导入路径：

```bash
python -c "import rl_isaaclab; print(rl_isaaclab.__file__)"
```

本机正确结果应指向：

```text
/home/user/rotate_dexgen/rl_isaaclab/__init__.py
```

正确启动时应看到：

```text
observation_space=141
RunningMeanStd: (141,)
RunningMeanStd: (30, 47)
```

### 问题 2：使用 `--so3` 时找不到 8 桶抓取缓存

错误摘要：

```text
FileNotFoundError: cache/xhand_grasp_linspace_0.4-0.6-8.npy
```

根因：

- `play.py` 的旧 `--so3` 分支是早期 UniLab 复刻配置，写死为 `scale_range = [0.4, 0.6, 8]`。
- 后续 8500 训练配置已演进为 `scale_range = [0.4, 0.7, 10]`，但 `play.py --so3` 没有同步更新。
- 仓库中有 XHand 10 桶缓存，但没有 XHand 8 桶缓存。
- `cache/sharpa_grasp_linspace_0.4-0.6-8.npy` 属于 SharpaWave，其状态是 29 维；XHand 是 28 维，不能代替使用。

最终决定与修改：

- 用户确认将 `--so3` 对齐到当前 8500 训练配置和现有缓存。
- 已修改 `rl_isaaclab/scripts/play.py`：

```python
env_cfg.reset_random_quat = True
env_cfg.scale_range = [0.4, 0.7, 10]
```

- `--so3` 帮助文本也已更新，明确 `num_envs` 需为 10 的倍数。
- 中间曾为 8 桶缓存生成向 `gen_grasp.py` 加入临时参数，最终已完全撤销；`gen_grasp.py` 无最终改动。

### SO(3) 随机化的实际含义

- `reset_random_quat = True` 会在 episode reset 时随机化手和物体的整体空间方向。
- 手、物体和 reward 使用的旋转轴会一起变换，保持抓取时的相对关系。
- 随机化发生在重置时，不是运行中持续旋转整只手。
- SO(3) 随机化本身与尺寸桶数没有必然关系；当前 `--so3` 只是将方向随机化和当前的 10 尺寸分布组合在一起。
- 环境实现会把并行环境平均分给每个尺寸桶，所以 10 桶配置下 `num_envs % 10` 必须为 0。

### 当前可用的仿真命令

单尺寸、掌心朝上的快速预览：

```bash
conda activate env_isaaclab
cd /home/user/rotate_dexgen

python rl_isaaclab/scripts/play.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 1 \
  --algorithm ProprioAdapt \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth
```

SO(3) 随机方向 + 10 尺寸分布：

```bash
python rl_isaaclab/scripts/play.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 10 \
  --algorithm ProprioAdapt \
  --so3 \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth
```

需要更多并行环境时，`--num_envs` 可使用 `20` / `30` / `40` 等 10 的倍数。省略 `--headless` 才会打开 Isaac Sim GUI。

### 云服务器部署检查清单

1. 确认服务器的 NVIDIA 驱动、Isaac Sim、IsaacLab 和 Python 版本互相兼容。本地参考环境是 Isaac Sim 5.1.0、IsaacLab 2.3.2、Python 3.11。
2. 上传或 clone 本仓库后，必须在云端目标 conda 环境里重新执行 editable 安装：

```bash
conda activate env_isaaclab
cd /path/to/rotate_dexgen
python -m pip install -e .
```

3. 不要复制本机 `site-packages` 或 editable `.pth` 文件到云端；这些文件包含本机绝对路径，容易再次导入错仓库。
4. 每次切换 clone / branch / 仓库后验证导入路径：

```bash
python -c "import rl_isaaclab; print(rl_isaaclab.__file__)"
```

5. 仿真启动时检查 `observation_space=141`，以及 `RunningMeanStd: (141,)` 和 `(30, 47)`。如果变成 186 / 62，优先排查导入路径和观测配置。
6. 确认云端包含训练所需的缓存和权重，特别是：

```text
cache/xhand_grasp_linspace_0.4-0.7-10.npy
ckpts/frontpad_students_20260701/student_8500.pth
```

7. 训练 teacher 的基本命令：

```bash
export CUDA_VISIBLE_DEVICES=0
python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --headless \
  --num_envs 10
```

8. 蒸馏 student：

```bash
python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --headless \
  --algorithm ProprioAdapt \
  --load_path /path/to/teacher/stage1_nn/last.pth
```

9. 云端扩大 `num_envs` 时需根据 GPU 显存逐步上调。当前 10 尺寸桶配置建议使用 10 的倍数，例如 4090 机器可从 10240 或更保守数量起测；L20 历史训练使用 20480。
10. 大规模 headless 训练前，先用小规模仿真或 smoke test 确认资产、缓存、观测维度和 checkpoint 均能正常加载。

### 今日最终文件变更

- `rl_isaaclab/scripts/play.py`
  - 将 `--so3` 尺寸分布从历史的 `[0.4, 0.6, 8]` 更新为 `[0.4, 0.7, 10]`。
  - 更新 CLI 帮助文本，说明 `num_envs` 必须是 10 的倍数。
- `work_log.md`
  - 新增本日问题、根因、处理和云端部署记录。

已对 `play.py` 和 `gen_grasp.py` 执行 Python 语法编译检查，检查通过。
