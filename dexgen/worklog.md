# DexGen 交接文档（2026-07-23）

本文用于在重启对话后快速恢复工作。项目根目录为
`/home/user/rotate_dexgen`；本地 Isaac/训练环境为 `env_isaaclab`。

## 当前结论与推荐部署版本

当前应部署主动恢复数据训练得到的 **v2b** 组合：

```text
Diffusion: dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM:       dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
```

两个文件已从云端同步至本机并用 `torch.load` 验证：

| 模型 | epoch | 最佳验证值 |
| --- | ---: | ---: |
| IDM v3 | 15 | normalized target MSE = 0.0052309 |
| Diffusion v3 v2b | 15 | validation loss = 0.0967256 |

不要部署 `film_v3`：它是保留的架构消融，主动扰动闭环结果略差于 v2b。

## 已完成的工作

### 文档与运行入口

- `README.md` 已简化为三类启动方式：RL+噪声、MANUS 手动方向切换、MANUS 手势自动方向切换。
- `DEVELOPMENT.md`、历史 `test.md`、历史 worklog 已移至 `dexgen/docs/`。
- `README.md` 中的 RL、双手可视化和 MANUS 默认检查点已更新到 active-recovery v3 v2b。
- 已新增 `docs/ACTIVE_RECOVERY_V3_RESULTS.md`，保存正式对比结果。

### 主动恢复数据与训练管线

已实现并验证以下修改：

- `collect_rl_data.py` 支持三类标注扰动：action bias、单指释放、物体外力。
- 扰动阶段不作为 action supervision；只保留干净 RL 专家恢复轨迹。
- 每个事件记录 `supervision_valid`、扰动类型、手指编号、恢复成功、旋转轴等字段。
- `data.py` 会剔除跨越无效监督帧的窗口；可通过 `--explicit_recovery_only --recovery_fraction 0.25` 精确让成功恢复窗口占每个方向训练样本的 25%。
- `train_idm.py`、`train_diffusion.py` 已支持上述恢复采样参数。
- 新增 `audit_recovery_data.py`、`monitor_recovery_progress.py`、`collect_recovery_dataset.sh`。
- `eval_dexgen.py` 已增加与采集器一致的主动扰动评估：三类扰动、5–8 秒间隔、扰动 0.2 秒、恢复观察 2 秒；成功标准为未掉落、至少 3 指接触、有向转速不低于 0.15 rad/s。
- 测试：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests/test_dexgen_pipeline.py`，最近结果为 **17 passed**。

### 正式主动恢复数据

云端正式数据目录：

```text
/root/rotate_dexgen/dexgen/data/recovery_modes_v6/
```

包含 CW/CCW 各 10 个分片，共 20 个分片。审计结果：

| 指标 | 数值 |
| --- | ---: |
| episodes | 20,000 |
| transitions | 6,339,776 |
| 扰动事件 | 31,473 |
| action bias 成功率 | 75.66% |
| 单指释放成功率 | 67.54% |
| 物体外力成功率 | 76.07% |
| 明确恢复训练窗口 | 455,799 |
| 平均有向恢复转速 | +0.7715 rad/s |

正式训练混合旧稳定旋转数据 `large_modes_v5` 与上述恢复数据。训练用 75% 普通旋转 + 25% 明确成功恢复窗口，避免模型只夹紧物体而停止旋转。

## 已完成的模型比较

### IDM 验证

新 IDM 的 oracle-motion 评估（100,000 验证样本）：

```text
Oracle motion -> IDM target MAE = 0.00280 rad
恢复子集                         = 0.00282 rad
```

说明 IDM 对恢复窗口没有明显退化；完整系统误差主要来自 Diffusion motion。

### Diffusion 架构消融

`film_v3` 将触觉作为独立 FiLM 校正注入每个 U-Net block，并含 10% tactile condition dropout；代码、检查点兼容和测试均已实现。离线指标接近，但主动扰动闭环不如 v2b：

| 指标（CW+CCW） | v2b | film_v3 |
| --- | ---: | ---: |
| 离线 motion MAE | 0.9097 mm | 0.9183 mm |
| 离线 Diffusion→IDM target MAE | 0.01636 rad | 0.01641 rad |
| 主动恢复成功率 | 392/532 = 73.68% | 381/529 = 72.02% |
| 扰动事件掉落 | 61 | 77 |

结论：保留 `film_v3` 作为消融，不继续作为当前部署候选。

### 与用户常用 tactile-v1 的严格对比

旧对照：

```text
diffusion_modes_full_dr_tactile_epsilon_cosine_v1_best.pt
idm_modes_full_dr_tactile_v1_best.pt
```

评估条件相同：SO(3) 随机化、CW/CCW、每方向 20 环境、2400 步、DDIM 8、固定 seed、三类主动扰动。

| 指标（CW+CCW） | tactile-v1 | active-recovery v3 v2b | 变化 |
| --- | ---: | ---: | ---: |
| 恢复成功率 | 368/537 = 68.53% | 392/532 = 73.68% | +5.15 pp |
| episode 掉落率 | 284/462 = 61.47% | 231/420 = 55.00% | -6.47 pp |
| 扰动事件掉落 | 81 | 61 | -24.69% |
| action bias 恢复率 | 70.95% | 75.28% | +4.33 pp |
| 单指释放恢复率 | 60.34% | 66.10% | +5.77 pp |
| 物体外力恢复率 | 74.30% | 79.66% | +5.36 pp |

新版不是靠降低旋转换稳定性：CW 转速 `0.7987 -> 0.8332 rad/s`，CCW 转速 `0.7573 -> 0.7931 rad/s`；wrong-way 分别下降 2.32 和 1.74 个百分点。

更早的旧 v2b 最佳组合（`r2a02 Diffusion + r2a005 IDM`）也已做主动扰动对照：恢复率 66.98%，仍低于新 v2b 的 73.68%。

## 当前运行命令

### RL+噪声双手可视化

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab

python dexgen/commands/eval/eval_dual_hand.py \
  --diffusion dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --idm dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --interactive_mode_switch \
  --noise_alpha 0.1 \
  --dangerous_command_strength 1.0 \
  --dangerous_command_duration 1.0 \
  --max_command_motion_m 0.03 \
  --guidance_strength 100000 \
  --command_time_mode separate \
  --command_horizon_steps 1 \
  --command_second_horizon_steps 2 \
  --ddim_steps 25 \
  --steps 3000 \
  --device cuda:0
```

`noise_alpha` 是持续运行、每 10 Hz 重采样的 RL action 噪声，没有单独持续时间。`o` 与 `1–5` 的危险 command 共用强度和持续时间；若要匹配训练时的单指扰动，先用：

```bash
--dangerous_command_strength 0.30 --dangerous_command_duration 0.20
```

### MANUS 自动模式（当前调试建议）

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab

python manus_dexgen_bridge/eval_manus_auto_mode.py \
  --manus_command_gain 0.55 \
  --guidance_strength 10000 \
  --max_command_motion_m 0.03 \
  --object_scale 0.6 \
  --device cuda:0
```

基础圆柱直径 80 mm、高 64 mm；`object_scale=0.6` 对应直径约 48 mm、高约 38.4 mm。`max_command_motion_m=0.03` 会限制 MANUS 每个关键点的目标位移到 30 mm（此前曾误认为它不作用于 MANUS，现已由代码确认：它传入 `RelativeKeypointCommand`）。

## MANUS：已定位但尚未解决的问题

现象：手套做旋转时，DexGen 执行手旋转不明显；同一模型在 RL+噪声输入下旋转良好。

已确认：最近的 MANUS command 日志平均关键点位移约 7.2 mm，95% 分位约 16.5 mm，最大值达到 30 mm 限幅。因此 `manus_command_gain=0.55` 并非简单“太小”。

根因假设（强，已由代码路径支持）：

1. RL 专家每 0.1 秒持续给出与旋转一致的手指 gait command；MANUS 当前提供的是相对标定抓姿的静态残差目标。执行手追近该目标后，残差趋小，不能持续表达旋转。
2. MANUS 当前被强制使用 `--command_time_mode broadcast`，即两个 Diffusion 预测时刻使用同一个 command；RL 双手入口使用 `separate` 的两帧时序 command，能表达运动方向和速度。
3. 过高的 `guidance_strength` 会迫使 Diffusion 跟随非旋转或接近静止的 MANUS command，从而压制模式条件中的 CW/CCW 旋转先验。`50000` 很可能过高；当前建议先用 `10000`，并测试 `5000/10000/20000`，固定 gain 后再调 gain。
4. 如果用户主要转动的是手腕/整只手而不是手指关节，当前 21-D glove joint → wrist-frame keypoint 映射本身不会保留全局手腕旋转，这也是信息缺失来源。

## 下一步计划（按优先级）

1. **MANUS 参数扫频，无需改代码。** 固定 `object_scale=0.6`、`max_command_motion_m=0.03`、gain=0.55，依次测试 guidance `5000/10000/20000`；每次记录转速、掉落和 command 比较日志。只在确认 guidance 合适后测试 gain `0.45/0.55/0.70`。不要同时改多个参数。
2. **若仍不能持续旋转，改 MANUS command 表达。** 为 MANUS 增加基于最近 10 Hz 手套 keypoint 速度的两帧时序 command（当前帧 + 短时外推帧），并解除仅 `broadcast` 的限制；这比继续提高 gain 更符合 RL command 的时序语义。改完后必须进行 MANUS/RL command 毫米幅度、闭环转速和掉落对比。
3. **本地 Isaac Sim 5.1 实测新 v2b。** 云端正式评估基于较旧 Isaac；本地可运行推荐 checkpoint，先确认 30 秒以上持续旋转、掉落时间和 MANUS 行为。云端本地版本差异不应混合到训练数据；本地 Isaac 5 的单个 CW 恢复分片曾单独保留，未并入正式云端数据。
4. **若要继续训练，先以闭环主动扰动基准决定。** 当前不建议再训练 film_v3；优先补充真实 MANUS 失败姿态/轨迹（若获得授权采集），并保持 75% 稳态旋转 + 25% 成功恢复采样和有向转速约束。

## v3 后续性能与稳定性提升建议（夜间工作计划）

### 优先级 1：DexGen 闭环 DAgger 恢复数据

当前恢复数据由 RL 专家在随机扰动后生成；实际部署时由 DexGen 自己动作，误差会逐步累积。下一轮最有价值的数据不是再扩大普通 RL 数据，而是记录 v3 自己的失败前状态，再让 RL 专家从同类状态给出恢复目标。

建议采集：

- 掉落前 1–2 秒；
- 单指接触下降、接触数下降、旋转变慢或反向的片段；
- MANUS 遥操下的失败姿态与命令；
- 连续两次扰动、不同手指组合松脱、逐渐减小接触力等更接近真实滑落的情况。

重点补强单指释放：当前它是最弱项，新 v2b 成功率仅 66.10%。下一轮采样建议从当前 75% 普通旋转 / 25% 恢复改为更关注失败模式的比例，例如：

| 类别 | 建议采样占比 |
| --- | ---: |
| 稳态旋转 | 65–70% |
| 单指释放恢复 | 20–25% |
| 物体外力恢复 | 10–15% |
| action bias 恢复 | 10–15% |

总和可按实际采样事件重叠微调；原则是不要降低稳态旋转到会使模型变成只夹紧而不旋转。

### 优先级 2：MANUS 改为两帧时序 command

这是最重要的非训练改动。当前 MANUS 使用相对标定姿态残差，并被限制为 `broadcast`，即两个 Diffusion 预测时刻使用同一个静态 command。RL 输入则是持续的、与旋转一致的手指 gait，因此 RL+噪声旋转明显好于当前 MANUS。

目标设计：

```text
最近 10 Hz glove keypoint 历史
        -> 当前 keypoint 速度
        -> 第一帧：短时位移
        -> 第二帧：速度外推位移
        -> DexGen 两帧 separate command
```

实现时应：

1. 保留当前 `broadcast` 路径作为开关和对照；
2. 为 MANUS 添加速度平滑、异常/断连回退和每关键点限幅；
3. 允许 MANUS 使用 `separate`，不再在 evaluator 中强制 `broadcast`；
4. 记录 MANUS 与 noisy-RL command 的毫米幅度、时间差分、闭环转速与掉落；
5. 先用固定 `object_scale=0.6` 和同一 CW/CCW 方向做 A/B，不要和模型训练改动混在一起。

注意：如果用户主要转的是手腕/整只手，而不是手指关节，当前 21-D glove joint 到 wrist-frame keypoint 的映射会丢失全局手腕旋转信息；此时还需要确认 RetargetX 是否能提供可用的腕部姿态或专门的旋转意图信号。

### 优先级 3：运行时滑落保护层

可先不训练，直接在运行时检测：接触指数量下降 + 触觉下降速度变大 + 有向旋转速度下降或反向。触发后：

1. 暂时降低/冻结 MANUS guidance；
2. 保留当前 CW/CCW mode，让 Diffusion 的旋转先验主导；
3. 限制张掌或撤指 command；
4. 接触恢复且转速正常后，再平滑恢复 MANUS 权重。

这套保护层应先作为可开关的安全机制，不应在没有闭环基准前直接取代学习策略。

### 优先级 4：触觉与状态特征扩展

v2b 已编码五指触觉历史、差分和 contact mask。若 DAgger 后仍不能处理滑落，可增加：

- 每指触觉下降速度与接触数变化；
- 物体有向角速度相对目标方向的误差；
- 物体线速度/高度异常；
- 若仿真接口可取，接触点的切向速度或指尖相对物体滑动速度。

不要优先继续扩大 `film_v3`；现有消融显示它没有带来主动恢复收益。

### 每个候选版本的验收门槛

所有后续模型和 MANUS command 改动都必须运行相同的主动扰动闭环 benchmark。接受新版本至少需要：

| 指标 | 当前 v3 v2b | 下一版最低要求 |
| --- | ---: | ---: |
| 恢复成功率 | 73.68% | 高于 73.68% |
| 单指释放恢复 | 66.10% | 高于 66.10% |
| episode 掉落率 | 55.00% | 低于 55.00% |
| CW/CCW 转速 | 约 0.8 rad/s | 不低于当前 |

夜间建议执行顺序：先做 MANUS guidance/gain 扫频并保存 command 日志；然后实现并 smoke-test 两帧时序 MANUS command；最后再安排 v3 失败状态 DAgger 采集。不要在同一轮同时改变模型、数据和 MANUS 映射，否则无法定位提升来源。

## 关键文件与日志

- 正式结果：`dexgen/docs/ACTIVE_RECOVERY_V3_RESULTS.md`
- 云端下载日志副本：`dexgen/logs/cloud_active_recovery_v3/`
- 最新 MANUS/RL command 比较：`manus_dexgen_bridge/logs/latest_command_comparison.pt`
- 主动恢复 evaluator：`dexgen/commands/eval/eval_dexgen.py`
- MANUS relative command：`manus_dexgen_bridge/command.py`
- MANUS evaluator wrapper：`manus_dexgen_bridge/eval_manus_guided.py`

## 云端信息

- 云端项目：`/root/rotate_dexgen`
- 云端 Python：`/home/ray/anaconda3/envs/env_isaaclab/bin/python`
- 正式数据与训练已完成；没有需要继续等待的 IDM、Diffusion 或主动恢复 benchmark 任务。
- 云端代码对 active-recovery v3、评估器和结果文档做过备份，备份目录以 `_backup_...` 开头。

---

# 2026-07-27 交接更新：触觉 RMA、长历史消融、训练提速与闭环验收

本节是 2026-07-27 当天工作的最新状态。下次重启对话时，应优先阅读本节；
上面的 2026-07-23 内容仍作为 v3、MANUS 和主动恢复数据的历史背景。

## 一句话结论

**当前唯一推荐部署版本仍然是 active-recovery v3 v2b：**

```text
Diffusion: dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM:       dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
```

今天完成的长历史 v5、RMA 初始化 v6、冻结 RMA 骨干 v7 均未超过 v3，不能替换
部署权重。云端当前没有训练或评测进程，两张 L20 均为空闲。

## 今天已解决的问题

### 1. MANUS 纯关节遥操模式

`dexgen/commands/eval/eval_guided.py` 已支持绕过 Diffusion/IDM 的纯 MANUS 关节输出：

- `t`：切换为 direct MANUS joints；
- `h`：返回 MANUS → DexGen guidance，随后必须按 `a` 重新标定；
- 直接关节目标默认每个 10 Hz tick 限制 `0.15 rad`，参数为
  `--manus_direct_max_joint_delta_rad`；
- reset 后等待新的有效 MANUS 帧；
- 8020 数据过期、断连或关节契约失败时自动 disarm，并保持安全 PD target；
- 纯遥操期间 `c/v` 只更新下次 DexGen guidance 的方向，不接管当前直接输出。

README 对应说明已更新，键盘提示会显示：

```text
h=MANUS → DexGen, n=noisy-RL → DexGen, t=direct MANUS joints
```

### 2. 云端 MANUS bridge 缺失

云端最初运行 `eval_guided.py` 时缺少 `manus_dexgen_bridge`。已将本地 bridge
同步到 `/root/rotate_dexgen/manus_dexgen_bridge/`，并验证
`RelativeKeypointCommand` 可导入，MANUS/guided 冒烟测试可以启动。

### 3. 正确区分两类闭环评测入口

曾使用 `eval_guided.py` 做 v5/v3 对比，结果与历史 v3 不一致。已确认：

- `eval_guided.py` 是 MANUS/guidance 路径，不是历史 active-recovery 接受指标；
- 历史 v3 正式基准使用 `eval_dexgen.py`；
- 正式协议为 SO(3)、20 env、2400 steps、DDIM 8、CW/CCW 分开、三类扰动。

后续 v5/v6/v7 均使用 `eval_dexgen.py` 同口径多 seed 评测。

### 4. 为 RMA 数据增加 `priv_info`

`dexgen/commands/data/collect_rl_data.py` 已在每个 transition 保存 9 维 `priv_info`：

```text
[object displacement xyz, friction, mass, COM xyz, scale]
```

`dexgen/data.py` 已能把 `priv_info` 作为可选监督目标返回；旧数据没有该字段时仍
保持兼容。新增正式采集脚本：

```text
dexgen/pipelines/collect_rma_privileged_dataset.sh
```

### 5. 正式 RMA 特权数据采集完成

云端目录：

```text
/root/rotate_dexgen/dexgen/data/rma_privileged_v1/
```

包含 CW/CCW 各 5 个正式分片，共 5,000 episode：

| 指标 | 数值 |
| --- | ---: |
| transitions / priv_info frames | 1,593,896 |
| episode 掉落率 | 30.84% |
| 扰动事件 | 7,945 |
| action bias 恢复率 | 74.88% |
| finger release 恢复率 | 66.69% |
| object force 恢复率 | 75.98% |
| 可训练窗口 | 719,272 |
| 明确成功恢复窗口 | 114,048 |
| 平均恢复转速 | +0.7724 rad/s |

9 维标签覆盖检查通过，主要随机化范围：

```text
friction: 0.50–2.00
mass:     0.0107–0.2500 kg
COM:      approximately ±0.01 m
scale:    0.4–0.7
```

### 6. RMA 适配器与迁移接口

新增：

- `dexgen/commands/train/train_rma.py`
- `dexgen.models.RMAAdapter`
- `initialize_tactile_encoder_from_rma`
- `freeze_tactile_feature_backbone`
- `train_diffusion.py/train_idm.py` 的 `--rma_init`
- `--freeze_rma_tactile_backbone`

首轮 9 维直接回归明显过拟合。逐维验证发现：

- scale 最容易辨识，normalized MSE `0.0139`；
- friction/mass 有一定可辨识性；
- COM 基本无法从当前短触觉历史可靠辨识。

因此后续 RMA 监督收缩为 `priv_info[3,4,8]`，即 friction/mass/scale：

| RMA | 历史 | 最佳 normalized val MSE | RMSE |
| --- | ---: | ---: | ---: |
| v2 | 16 帧 | 0.431638 | 0.6570 |
| h4-v3 | 4 帧 | 0.506946 | 0.7120 |

为保持 v3 的时间契约，v6/v7 使用 4 帧 h4-v3 RMA checkpoint：

```text
dexgen/checkpoints/rma_privileged_friction_mass_scale_h4_v3_best.pt
```

### 7. 自动训练—测试接力流水线

新增并泛化：

```text
dexgen/pipelines/run_rma_v6_pipeline.sh
```

它会等待 Diffusion/IDM 完成，然后自动执行：

1. `tests/test_dexgen_pipeline.py`；
2. 100,000 样本离线评估；
3. CW/CCW × seeds 123/456/789 主动恢复闭环；
4. 写入 `TRAINING/OFFLINE/CW/CCW/PIPELINE_COMPLETE` 标记。

曾有两个流水线问题，均已修复：

- Diffusion 完成日志实际为 `best val=`，不是旧的 `best normalized`；
- rsync 后脚本执行位被清除，需要 `chmod +x`。

当前测试结果为 **18 passed**。

### 8. 训练效率提升

v7 epoch 1 使用 batch 512，之后从完整 checkpoint 续训并调整为：

```text
batch_size=4096
num_workers=16
prefetch_factor=4
```

同时加入训练字段裁剪，worker 只返回模型实际使用的张量，避免传输 episode ID、
扰动掩码、专家动作等无用字段。最终吞吐：

```text
Diffusion: approximately 96.9k samples/s
IDM:       approximately 94.0k samples/s
```

原吞吐约 32–39k samples/s，提速超过 2 倍。L20 显存仍只占约 1–2 GB，因为
模型本身很小；GPU 呈短时 60–90% 计算、随后等待 Python 随机滑窗/IPC 的突发
模式。真正的剩余瓶颈是 Dataset 逐样本构造，而不是显存容量或磁盘 I/O。

## 今天完成但未通过的实验

### A. v5 长历史触觉

权重：

```text
diffusion_modes_full_dr_active_recovery_v5_tactile_longhistory_best.pt
idm_modes_full_dr_active_recovery_v5_tactile_longhistory_best.pt
```

配置：tactile-v2b、16 history steps、stride 2，约 1.5 秒历史。三 seed 正式评估：

| 方向 | 掉落率 | 转速 | wrong-way | 恢复率 |
| --- | ---: | ---: | ---: | ---: |
| CW | 60.04% | +0.7279 | 17.50% | 70.57% |
| CCW | 66.84% | +0.6649 | 18.16% | 70.10% |
| 合计 | 63.44% | +0.6964 | 17.83% | 70.34% |

结论：仅增加触觉历史长度无效，且明显低于 v3。详细文档：

```text
dexgen/docs/V5_LONGHISTORY_ACTIVE_RECOVERY_RESULTS.md
```

### B. v6：RMA tactile encoder 初始化后全量联合微调

权重：

```text
diffusion_modes_full_dr_active_recovery_rma_init_v6_best.pt
idm_modes_full_dr_active_recovery_rma_init_v6_best.pt
```

离线：

```text
motion MAE              = 0.9130 mm
Oracle motion -> IDM    = 0.00278 rad
Diffusion -> IDM        = 0.01644 rad
```

三 seed 闭环汇总：

| 方向 | 平均掉落率 | 平均转速 | wrong-way | 恢复率 |
| --- | ---: | ---: | ---: | ---: |
| CW | 51.60% | +0.8256 | 13.31% | 574/805 = 71.30% |
| CCW | 61.83% | +0.7876 | 12.87% | 583/803 = 72.60% |
| 合计 | 56.72% | +0.8066 | 13.09% | 1157/1608 = 71.95% |

action bias 为 79.10%，但 object force 只有 71.40%。结论：初始化后的表征在
15 epochs 联合微调中被冲淡，整体未超过 v3。详细文档：

```text
dexgen/docs/RMA_INIT_V6_RESULTS.md
```

### C. v7：冻结 RMA 触觉骨干

权重：

```text
diffusion_modes_full_dr_active_recovery_rma_frozen_v7_best.pt
idm_modes_full_dr_active_recovery_rma_frozen_v7_best.pt
```

离线：

```text
motion MAE              = 1.2444 mm
Oracle motion -> IDM    = 0.00338 rad
Diffusion -> IDM        = 0.02488 rad
```

三 seed 闭环：

| 方向 | 平均掉落率 | 平均转速 | wrong-way | 恢复率 |
| --- | ---: | ---: | ---: | ---: |
| CW | 67.47% | +0.7677 | 16.64% | 529/803 = 65.88% |
| CCW | 74.65% | +0.7119 | 16.49% | 558/804 = 69.40% |
| 合计 | 71.06% | +0.7398 | 16.56% | 1087/1607 = 67.64% |

结论：明显退化，不能部署。冻结骨干限制了恢复任务对触觉特征的重塑。同时
batch 由 512 放大到 4096 后，每 epoch optimizer update 数减少 8 倍，是该实验
的混杂因素；v7 能否定当前配置，但不能单独证明 RMA 思路无效。详细文档：

```text
dexgen/docs/RMA_FROZEN_V7_RESULTS.md
```

## 为什么这还不是完整的 ProprioAdapt/RMA

原旋转 RL 的 `ProprioAdapt` 代码位于：

```text
rl_isaaclab/algo/padapt/padapt.py
rl_isaaclab/algo/models/models.py
```

实际机制：

1. Teacher 用 `env_mlp(priv_info)` 生成环境 latent；
2. Student 的 `adapt_tconv` 读取 30 帧 `proprio_hist`，其中包含本体与五指触觉；
3. Student 拟合的是 teacher latent，而不是直接回归动作；
4. 策略始终把该 latent 拼入 actor 输入；
5. stage 2 冻结 actor，只训练 `adapt_tconv`。

今天的 v6/v7 只做了“预测物理参数的触觉表征预训练 → 初始化/冻结 DexGen
触觉 encoder”，**没有将 teacher/student latent 持续作为 Diffusion/IDM 的显式
条件**。因此它们只是 RMA 启发的消融，不是完整复现。

## 仍未解决的问题

1. **尚无新模型超过 v3。** v5/v6/v7 均失败，当前部署版本不要改变。
2. **完整 teacher/student latent DexGen 尚未实现。** 这是触觉借鉴 RL 的核心缺口。
3. **v7 存在大 batch 混杂。** 提速后的 batch 4096 未同步调整学习率/更新预算；
   后续公平实验必须固定 optimizer update 数或采用明确的 LR scaling。
4. **DataLoader 仍是吞吐瓶颈。** 字段裁剪后仍存在逐样本 Python 滑窗构造和 IPC；
   若继续优化，应做矢量化 batch fetch、预计算窗口索引/连续张量或 GPU 缓存。
5. **本地 CUDA Error 804 未解决。** 本机 Isaac 启动曾报
   `forward compatibility was attempted on non supported HW`。云端 L20 正常；
   本机需要检查 NVIDIA 驱动、CUDA compatibility library/容器映射并重启验证，
   不能把云端成功等同于本机已修复。
6. **真机尚未验证。** `deploy_docker`/XHand 真机链路与触觉传感器需要拿到硬件后
   验证，当前只能做安装、代码和仿真检查。
7. **MANUS 持续旋转表达仍未完成验证。** direct joints 已实现，但 MANUS 两帧
   velocity/separate command、腕部旋转意图和真机闭环仍待测试。

## 下次工作计划（严格按顺序）

### 优先级 1：实现真正的 DexGen teacher/student latent（建议命名 v8）

不要继续做 encoder 初始化消融。建议两阶段：

```text
Stage 1:
priv_info -> teacher env encoder -> latent
state/history + latent -> Diffusion / IDM

Stage 2:
state+tactile history -> student temporal adapter -> predicted latent
freeze teacher-conditioned policy
minimize student latent vs detached teacher latent

Deployment:
remove priv_info; only student latent conditions Diffusion / IDM
```

具体要求：

- 先使用现有 5,000 episode `rma_privileged_v1` 数据实现 teacher/student；
- 对照 RL 的 `prop_hist_len=30`，但同时保留 4 帧部署低延迟版本做消融；
- latent 不应直接等同 raw 9D 参数；由 teacher `env_mlp` 学习紧凑 latent；
- Diffusion 和 IDM 都必须显式接收同一 latent；
- stage 2 冻结策略主体，只训练 student adapter，严格对齐 ProprioAdapt；
- checkpoint 必须保存 teacher/student 配置、latent dim、history contract 和
  `priv_info` 索引；
- 在线 evaluator 只能访问 student history，严禁把 `priv_info` 泄漏到部署路径。

### 优先级 2：建立公平训练效率契约

大 batch 训练必须满足至少一个条件：

- 使用 linear/sqrt LR scaling 并先做短 benchmark；
- 或按 optimizer update 数而不是 epoch 数对齐；
- 或保留 batch 512 作为严格质量对照。

建议先测试 `batch=2048` 和 `4096` 的 1–2 epoch：

```text
记录 samples/s、平均 SM、val loss、optimizer updates、wall time
```

不要以“显存占满”为目标；模型很小，46 GB 不可能在不破坏更新语义的情况下自然
占满。当前安全吞吐点约为 batch 4096、16 workers、prefetch 4。

### 优先级 3：v8 验收顺序

1. teacher latent 对 privileged domain 参数的可分性；
2. student→teacher latent MSE；
3. Oracle/teacher latent 与 student latent 的离线 motion/target 差值；
4. `tests/test_dexgen_pipeline.py`，当前基线为 18 passed；
5. 100,000 validation windows 离线评估；
6. `eval_dexgen.py`：CW/CCW × seeds 123/456/789；
7. 只有以下指标全部满足才允许替换 v3：

| 指标 | v3 门槛 |
| --- | ---: |
| 总恢复成功率 | > 73.68% |
| finger release | > 66.10% |
| object force | > 79.66% |
| episode 掉落率 | < 55.00% |
| CW/CCW 转速 | 不低于约 0.8 rad/s |
| wrong-way | 不高于约 13% |

### 优先级 4：模型工作结束后再回到 MANUS/真机

- MANUS guidance 做 `5000/10000/20000` 扫频；
- 实现/验证两帧 velocity + separate command；
- direct MANUS joints 做断连、reset、限幅和重新 arm 真机验证；
- 有真机后安装并验证 `deploy_docker`、XHand SDK、触觉标定和控制频率；
- 采集真实 MANUS 失败片段后，再考虑部署态 DAgger。

## 下次启动时先执行的检查

```bash
# 云端
ssh -p 1178 root@180.76.108.184
cd /root/rotate_dexgen
source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# 确认没有遗留任务
nvidia-smi
pgrep -af 'train_diffusion.py|train_idm.py|eval_dexgen.py|eval_offline.py'

# 回归测试
python -m pytest tests/test_dexgen_pipeline.py -q

# 当前部署权重必须仍是 v3
ls -lh \
  dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
```

## 今天新增/修改的关键文件

```text
dexgen/commands/data/collect_rl_data.py
dexgen/pipelines/collect_rma_privileged_dataset.sh
dexgen/data.py
dexgen/models.py
dexgen/commands/train/train_rma.py
dexgen/commands/train/train_diffusion.py
dexgen/commands/train/train_idm.py
dexgen/training_utils.py
dexgen/pipelines/run_rma_v6_pipeline.sh
dexgen/commands/eval/eval_guided.py
dexgen/README.md
dexgen/docs/V5_LONGHISTORY_ACTIVE_RECOVERY_RESULTS.md
dexgen/docs/RMA_INIT_V6_RESULTS.md
dexgen/docs/RMA_FROZEN_V7_RESULTS.md
```

关键云端日志：

```text
dexgen/logs/v5_longhistory_active_recovery/
dexgen/logs/rma_privileged_v1/
dexgen/logs/rma_v6_validation/
dexgen/logs/rma_frozen_v7_validation/
```

## 结束状态（2026-07-27）

- 云端 v5/v6/v7 训练与全部自动评测均已完成；
- 云端 GPU 空闲，无需继续等待任何进程；
- v3 checkpoint 未被覆盖；
- v6/v7 checkpoint 与日志保留用于消融，不得作为默认部署；
- 下一次对话应从“v8 显式 teacher/student latent 架构设计与实现”开始。

---

# 2026-07-28 更新：v8 teacher/student latent 训练已启动

已实现首个 4 帧低延迟 v8 版本：

- `PrivilegedLatentTeacher`：把 normalized `priv_info[3,4,8]`
  （friction/mass/scale）编码为 8-D learned latent，并通过重建目标避免 latent
  塌缩；
- `LatentStudentAdapter`：仅从部署可用的 4 帧 83-D proprio+tactile history
  预测相同 latent；
- Diffusion 与 IDM 都新增显式 `env_latent` 条件投影；
- 策略训练使用 teacher latent，checkpoint 同时保存 teacher/student、物理参数
  normalization、history contract 和 `deployment_latent_source=student`；
- v3 的默认 checkpoint 未修改。

新增文件：

```text
dexgen/commands/train/train_v8_adapter.py
dexgen/commands/train/train_v8_policy.py
dexgen/pipelines/run_teacher_student_v8_pipeline.sh
```

本地和云端原有回归测试均为：

```text
18 passed
```

云端正式数据 smoke test 已分别贯通 adapter、Diffusion 和 IDM。smoke checkpoint
只用于接口验证，不是部署候选：

```text
dexgen/checkpoints/v8_adapter_smoke_best.pt
dexgen/checkpoints/v8_diffusion_smoke_best.pt
dexgen/checkpoints/v8_idm_smoke_best.pt
```

正式自动接力流水线已启动。配置固定为 batch 512，避免引入 v7 的大 batch
optimizer-update 混杂：

```text
pipeline shell PID: 3290471
data:               dexgen/data/rma_privileged_v1/{cw,ccw}_rma_*.pt
adapter epochs:     20, cuda:0
policy epochs:      15
Diffusion:          cuda:0
IDM:                cuda:1
recovery sampling:  25% explicit successful recovery
```

接力顺序：

```text
shared adapter
    -> Diffusion teacher-latent training (GPU 0)
    -> IDM teacher-latent training       (GPU 1, 与 Diffusion 并行)
```

关键输出：

```text
dexgen/checkpoints/teacher_student_v8_h4_best.pt
dexgen/checkpoints/diffusion_modes_teacher_student_v8_h4_best.pt
dexgen/checkpoints/idm_modes_teacher_student_v8_h4_best.pt
dexgen/logs/teacher_student_v8_h4/
dexgen/logs/teacher_student_v8_pipeline.nohup.log
```

完成标记：

```text
dexgen/logs/teacher_student_v8_h4/ADAPTER_COMPLETE
dexgen/logs/teacher_student_v8_h4/DIFFUSION_COMPLETE
dexgen/logs/teacher_student_v8_h4/IDM_COMPLETE
dexgen/logs/teacher_student_v8_h4/TRAINING_COMPLETE
```

启动后已确认正式 adapter 正常训练并保存 checkpoint；epoch 7 验证值：

```text
teacher reconstruction MSE = 0.000045
student -> teacher MSE      = 0.051867
teacher latent min std      = 0.222440
throughput                   approximately 99k samples/s
```

训练完成后不能直接部署。下一步必须补齐 evaluator 的 student-only 在线路径，
并按既定顺序进行 teacher/student 离线差值、100,000 windows 离线评估和
CW/CCW × seeds 123/456/789 主动恢复闭环验收。

## 2026-07-28 v8-h4 训练后续与 v8b 修正

v8-h4 adapter、Diffusion、IDM 均已完成训练。策略最终验证值：

```text
Diffusion validation loss       = 0.116907
IDM normalized target val MSE   = 0.011229
```

已在 `eval_offline.py` 和 `eval_dexgen.py` 中接入真正的 student-only runtime
路径；在线 evaluator 不读取 `priv_info`。本地与云端回归测试仍为 18 passed。

RMA 数据完整验证集共 71,269 windows，teacher/student 离线结果：

| latent | motion MAE | Oracle→IDM MAE | Diffusion→IDM MAE |
| --- | ---: | ---: | ---: |
| student | 1.8634 mm | 0.00447 rad | 0.03769 rad |
| teacher | 1.8654 mm | 0.00447 rad | 0.03771 rad |

student→teacher latent MSE 为 `0.044154`。teacher/student 策略结果几乎相同，
说明首轮失败主要不是 student adapter，而是策略只使用 5,000 episode
`rma_privileged_v1` 训练，丢失了 v3 的大规模稳态与主动恢复数据覆盖。该版本
离线明显低于 v3，因此没有浪费算力运行 Isaac 多 seed 闭环。

已实现混合数据兼容：

- 有 `priv_info` 的 RMA 窗口使用 teacher latent；
- 旧稳态/恢复窗口没有 `priv_info`，使用冻结 student latent；
- `priv_info_valid` 明确区分两条路径，不用全零伪标签训练 teacher；
- 仍保持 25% 显式成功恢复采样和 batch 512。

v8b mixed 正式训练已启动：

```text
pipeline PID:        3293340
Diffusion PID:       3293347, cuda:0
IDM PID:             3293349, cuda:1
train windows:       5,609,909
validation windows:    621,082
recovery windows:      512,998
updates/epoch:          10,956
epochs:                     15
```

数据组合：

```text
large_modes_v5        20 shards
recovery_modes_v6     20 shards
rma_privileged_v1     10 formal shards
```

输出：

```text
dexgen/checkpoints/diffusion_modes_teacher_student_v8b_mixed_best.pt
dexgen/checkpoints/idm_modes_teacher_student_v8b_mixed_best.pt
dexgen/logs/teacher_student_v8b_mixed/
```

自动验收 watcher 已启动（PID `3293981`）：

1. 等待 v8b 训练结束；
2. 自动运行 pytest；
3. 运行标准 large+recovery 100k student-only 离线评估；
4. 在 RMA 验证集比较 teacher/student；
5. 只有 motion MAE `<= 1.00 mm` 且 Diffusion→IDM MAE `<= 0.018 rad`
   才自动启动 CW/CCW × seeds 123/456/789 Isaac 主动恢复闭环；
6. 离线不达标则写入 `OFFLINE_REJECTED`，保留 v3 部署版本。

验收日志：

```text
dexgen/logs/teacher_student_v8b_validation/
dexgen/logs/teacher_student_v8b_validation.nohup.log
```

---

# 2026-07-29 更新：v9 候选动作接触风险 Teacher/Student

纯 RL 实物成功 pipeline 重新核对后，触觉路线调整为：

```text
冻结 active-recovery v3 Diffusion / IDM（10 Hz）
                         │
20 Hz、30 帧 proprio+tactile history
                         │
ProprioAdapt 风格 TConv student -> 16-D contact latent
                         │
四帧 v3 state + 候选 motion + mode + latent
                         │
candidate-conditioned contact risk
```

与 v8 的主要区别：

- 不重新训练、覆盖或改变 v3 行为先验；
- Teacher latent 通过候选动作风险目标学习，避免只重建静态
  friction/mass/scale；
- Teacher 额外可见 `priv_info + object pose/velocity` 共 22 维特权状态；
- Student 使用 30 帧、20 Hz（约 1.5 秒）历史，而不是 4 帧、10 Hz；
- 失败恢复尾部只进入风险训练，不重新混入动作先验；
- 同时训练全五指和 proven 8103 mask `[1,4]` 两个传感器 profile。

新增文件：

```text
dexgen/risk_data.py
dexgen/risk_models.py
dexgen/commands/train/train_contact_risk.py
dexgen/pipelines/run_contact_risk_v9.sh
```

云端回归与 smoke：

```text
tests/test_dexgen_pipeline.py: 18 passed
单 shard 30 帧 dataset/model/teacher/student GPU smoke: passed
```

正式数据：

```text
dexgen/data/rma_privileged_v1/{cw,ccw}_rma_*.pt
train windows: 1,217,315
val windows:     133,937
safe / unsafe: 923,536 / 293,779
recovery safe / unsafe: 207,334 / 64,359
```

正式云端任务：

```text
full5 pipeline PID: 3301682, cuda:0
mask14 pipeline PID: 3301936, cuda:1

dexgen/logs/contact_risk_v9_full5/
dexgen/logs/contact_risk_v9_mask14/

dexgen/checkpoints/contact_risk_v9_full5_{teacher,student}_best.pt
dexgen/checkpoints/contact_risk_v9_mask14_{teacher,student}_best.pt
```

full5 Teacher 第一轮：

```text
teacher AUC:     0.636416
zero-latent AUC: 0.627716
validation BCE:  0.656178
throughput:       about 65k windows/s
```

流水线每个 profile 先训练 15 epochs Teacher；AUC 至少达到 0.60 才自动继续
20 epochs Student。下一步按以下顺序验收：

1. Teacher AUC 与 zero-latent uplift；
2. Student→Teacher latent MSE、风险概率 MAE 和 Student AUC；
3. full5 与 mask14 对比；
4. 仅对胜者接入冻结 v3 的 Diffusion candidate guidance；
5. CW/CCW × seeds 123/456/789 闭环评估，未超过 v3 前不修改默认部署权重。

## 2026-07-29 v9 后续：吞吐优化与 DAgger/local-risk 修正

RMA-only v9 正式对照结果显示 event-level recovery 标签很难泛化：

```text
full5 best student: overall AUC 0.6385, recovery AUC 0.5745
mask14 best student: overall AUC 0.6443, recovery AUC 0.5726
```

两组都在第一轮后过拟合，因此保留 best 并提前停止 mask14 的无效 epoch；没有接入
v3 guidance。

训练吞吐测试：

```text
batch 1024: 约 65k–75k windows/s
batch 8192: 首轮约 31k，稳态约 56k windows/s
batch 4096: 稳态约 88k，正式混合数据 full5 约 99k–114k windows/s
```

正式后续固定为 `batch=4096, workers=24, prefetch=4`，并按 recovery AUC
early-stop。L20 上该小模型显存约 570 MiB，SM 利用率呈 20–32% 的短峰；衡量效率
应使用 windows/s，不能通过无意义扩大模型来追求显存占满。

v9b 首次加入原 v4 DAgger 的 v3 执行/自然失稳数据，但仍用 whole-event success
标签，恢复 AUC 仍只有：

```text
full5 teacher recovery AUC: 0.5758
mask14 teacher recovery AUC: 0.5746
```

两组均在 epoch4 自动早停，并被 0.60 gate 拒绝，未启动 Student。

原因判断：一个恢复事件内所有帧共享最终成功/失败标签，而每个候选 motion 只覆盖
接下来 0.2 秒，监督与候选动作因果错位。已实现 v9c，把恢复帧也统一改为候选后的
0.5 秒局部标签：

```text
终点至少 3 指接触
0.5 秒平均 signed rotation >= 0.15 rad/s
drop horizon 内掉落 = unsafe
```

v9c 两组已于云端启动：

```text
full5 PID: 3305432, cuda:0
mask14 PID: 3305433, cuda:1

dexgen/logs/contact_risk_v9c_local_full5/
dexgen/logs/contact_risk_v9c_local_mask14/
```

Teacher recovery AUC 必须达到 0.65 才会自动蒸馏 30 帧 Student。

## 2026-07-29 v9c 完成结果

训练期间发现 v9c 初始脚本遗漏把位置参数 `cuda:0/cuda:1` 传给
`train_contact_risk.py`，导致两组 Teacher 都落在默认 GPU0。Teacher checkpoint
有效；Student 启动后立即停止，脚本补上 `--device "${device}"`，随后从已保存
Teacher 直接在两张卡重启。最终确认：

```text
full5 Student:  cuda:0
mask14 Student: cuda:1
两卡显存各约 0.9 GiB，观测到 SM 峰值最高约 69–79%
```

Teacher 最佳：

| profile | overall AUC | recovery AUC | regular AUC |
| --- | ---: | ---: | ---: |
| full5 | 0.711764 | **0.735925** | 0.703049 |
| mask14 | 0.710526 | 0.733778 | 0.702163 |

Student 最佳（两组均 epoch1，后续由 patience=3 在 epoch4 早停）：

| profile | overall AUC | recovery AUC | regular AUC | risk probability MAE | latent MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| full5 | **0.710090** | **0.734201** | **0.701386** | 0.025832 | 0.108868 |
| mask14 | 0.708822 | 0.731986 | 0.700489 | **0.021950** | 0.112122 |

输出：

```text
dexgen/checkpoints/contact_risk_v9c_local_full5_{teacher,student}_best.pt
dexgen/checkpoints/contact_risk_v9c_local_mask14_{teacher,student}_best.pt
dexgen/logs/contact_risk_v9c_local_full5/
dexgen/logs/contact_risk_v9c_local_mask14/
```

新增因果触觉消融命令：

```text
dexgen/commands/eval/eval_contact_risk.py
```

full5 Student 消融：

| 输入 | recovery AUC |
| --- | ---: |
| normal | **0.734202** |
| 只清零 30 帧 adapter touch | 0.732775 |
| 只清零 4 帧 risk-head touch | 0.685151 |
| 两者都清零 | 0.679337 |
| 只打乱 adapter touch | 0.733207 |
| 两者都打乱 | 0.694113 |

mask14 Student 消融：

| 输入 | recovery AUC |
| --- | ---: |
| normal | **0.731988** |
| 只清零 30 帧 adapter touch | 0.730554 |
| 只清零 4 帧 risk-head touch | 0.706896 |
| 两者都清零 | 0.705894 |

结论：

- 触觉确实提供因果有效信息；full5 全部清零后 recovery AUC 下降约 0.055；
- 主要贡献来自最近四帧的即时触觉，单独清零/打乱 30 帧 Adapter 触觉只下降约
  0.001–0.0014；
- full5 略优于 mask14，当前仿真候选为 full5；
- 长历史 Student 已几乎无损复现 Teacher，但尚未证明长历史本身优于短历史；
- 下一步先把 full5 risk 作为冻结 v3 的多候选 reranker 做闭环仿真，不修改默认
  v3 checkpoint；只有 CW/CCW 多 seed 超过 v3 门槛后才考虑真机。

## 2026-07-29 v9c 多候选闭环验收

已完成冻结 v3 上的 contact-risk 在线重排实现：

```text
dexgen/contact_risk_runtime.py
dexgen/commands/eval/eval_dexgen.py
dexgen/pipelines/eval_contact_risk_v9c_rerank.sh
dexgen/commands/eval/summarize_recovery_logs.py
```

关键安全契约：

- v3 Diffusion/IDM checkpoint 不修改；
- candidate 0 独立使用原 `diffusion_seed` RNG stream，额外候选使用独立
  `risk_seed`；
- `risk_candidates=1` 时单元测试证明 candidate 0 与原采样逐元素相同；
- risk checkpoint 必须是 20 Hz、30 帧、candidate-local、83-D tactile
  Student，并且 normalizer 与 v3 逐元素一致；
- 30 帧历史未就绪时强制使用 candidate 0；
- 支持 `--risk_min_logit_gain`，分数提升不足时保持 candidate 0。

回归：

```text
tests/test_dexgen_pipeline.py: 20 passed
```

新增 motion 因果消融：

| 输入 | recovery AUC |
| --- | ---: |
| normal | 0.734202 |
| motion zero | 0.490246 |
| motion shuffle | 0.509276 |

因此风险头确实依赖候选 motion，并非只读取当前状态/触觉。

正式评测配置：

```text
CW/CCW × seeds 123/456/789
20 envs，2400 steps，DDIM 8，SO(3)，标准三类主动扰动
rerank: 4 candidates，min logit gain=0.25
```

结果：

| 指标 | 同批 v3 baseline | v9c risk rerank | 变化 |
| --- | ---: | ---: | ---: |
| 总恢复 | 1153/1603 = 71.93% | 1134/1611 = 70.39% | -1.54 pp |
| action bias | 74.81% | 76.30% | +1.49 pp |
| 单指释放 | 67.54% | 62.27% | -5.27 pp |
| 物体外力 | 73.45% | 72.61% | -0.84 pp |
| episode 掉落率 | 56.46% | 55.18% | -1.28 pp |
| 平均有向转速 | 0.8076 | 0.8256 rad/s | +0.0180 |
| wrong-way | 13.15% | 12.70% | -0.45 pp |

风险重排共评估 `129852` 个 ready decision，`52.70%` 选择非 baseline，
平均 logit/probability 增益为 `+0.37999/+0.07711`。虽然长期 episode 存活、
转速和 wrong-way 略有改善，但总恢复率与关键的单指释放明显下降，**未通过
部署门槛，不替换 v3，不接入真机**。

日志：

```text
dexgen/logs/contact_risk_v9c_rerank_formal_c4_m0.25/
```

下一步判断：

1. 高离线 AUC 不等于同状态下的反事实候选排序能力，argmax 还可能放大模型误差；
2. 风险头评分的是 Diffusion motion，但真正执行的是 IDM joint target，需要检查
   被选候选的 IDM 可实现性误差；
3. 下一版优先加入 IDM target/可实现 motion 或一致性惩罚，并为同状态候选构造
   pairwise/contrastive 监督；
4. 在离线 candidate-ranking 与小规模闭环门槛通过前，不再盲目扩大训练 epoch
   或直接跑真机。

### v9c IDM 可实现性诊断

新增：

```text
dexgen/commands/eval/eval_contact_risk_candidates.py
--risk_idm_consistency_weight
```

对验证集 50,000 个状态各采样 4 条候选，并将每条 motion 经过冻结 IDM 与 XHand
FK 后比较：

```text
risk logit vs -FK first-step error correlation: -0.2244
baseline first error: 6.5900 mm
selected first error: 6.6460 mm
baseline second error: 8.1099 mm
selected second error: 8.2179 mm
```

风险 argmax 的确轻微偏向 IDM 较难实现的候选。离线 penalty sweep 显示
`0.10 logit/mm` 能在基本保持选择率的同时，把平均 first-error delta 从
`+0.0559 mm` 翻转为 `-0.0528 mm`。

随后完成 CW/CCW、seed123、800 步配对 smoke。在线 FK error 也按预期下降：

```text
CW:   2.1078 -> 2.0509 mm
CCW:  2.2471 -> 2.1524 mm
```

但闭环恢复仍下降：

| 指标 | smoke baseline | risk + IDM consistency |
| --- | ---: | ---: |
| 总恢复 | 125/161 = 77.64% | 118/164 = 71.95% |
| 单指释放 | 75.93% | 62.96% |
| episode 掉落率 | 62.12% | 57.03% |
| 平均有向转速 | 0.8095 | 0.8308 rad/s |

结论：IDM 可实现性偏差存在，但不是主动恢复下降的主因；一致性惩罚同样不进入
正式评测/真机。v9c 学到的 factual motion 分类仍不足以支持同状态
counterfactual argmax。下一版应先采集或构造同一状态下的候选对并训练
pairwise ranking，特别提高 single-finger-release 样本权重；否则继续调候选数、
margin 或 consistency weight 只会趋近 baseline，不能产生可靠提升。
