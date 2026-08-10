# DexGen 程序与训练流程

本文说明开发和实验维护；日常启动命令见 [README.md](../README.md)。

## 系统流程

```text
RL 专家 rollout / MANUS 手套 command
             ↓
10 个腕坐标系关键点运动（5×PIP + 5×指尖胶垫）
             ↓
Guided Diffusion 生成未来 0.1 / 0.2 s motion
             ↓
IDM 生成 21 维绝对 PD 目标
             ↓
Isaac Sim XHand 闭环执行
```

正式部署状态为 83 维：78 维本体状态加五指触觉标量。历史长度为 4 帧，规划频率为 10 Hz；Diffusion 预测两个未来时刻的 `[10, 3]` motion，IDM 输出下一规划时刻的 21 维绝对 PD target。

## 主要程序

| 程序 | 作用 |
| --- | --- |
| `collect_rl_data.py` | 用 CW/CCW RL 专家采集 XHand rollout，保存关节、PD target、关键点、触觉与终止信息。 |
| `collect_recovery_dataset.sh` | 可续跑地采集 CW/CCW 主动扰动恢复分片，并逐分片审计。 |
| `audit_recovery_data.py` | 统计扰动类型、恢复成功率、有效训练窗口和恢复期有向转速。 |
| `data.py` | 校验 rollout、裁掉失败尾部、按 episode 切分 train/val、构造历史窗口与归一化统计。 |
| `xhand_keypoints.py` | 21-DOF XHand 的精确 FK；定义 PIP 与真实指尖胶垫的 10 点表示。 |
| `train_idm.py` | 训练 motion → 21 维绝对 PD target 的逆动力学模型。 |
| `train_diffusion.py` | 训练条件 Diffusion，生成未来两帧关键点 motion。 |
| `models.py` | Diffusion、IDM、guidance sampler、触觉编码器及 checkpoint 合约。 |
| `eval_offline.py` | 在验证集评估 oracle-motion→IDM 与 Diffusion→IDM，定位误差来源。 |
| `eval_dexgen.py` | 无上层 command 的 Diffusion→IDM 闭环。 |
| `eval_guided.py` | RL、RL+noise、guided DexGen 的 Isaac 闭环评估；MANUS provider hook 也位于此处。 |
| `eval_dual_hand.py` | 在 Isaac 中显示 command 参考手与实际执行手。 |
| `xhand_motion_command.py` | Jacobian/FK command 的数值诊断工具。 |
| `manus_dexgen_bridge/` | RetargetX TCP 协议、latest-frame client、探针、MANUS evaluator 与自动方向选择。 |

## 正式模型与基线

当前正式稳定模型为 active-recovery v3，不能被实验 checkpoint 覆盖：

```text
Diffusion: dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt
IDM:       dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt
```

历史 v5、tactile-v1/v2b、DAgger v4 和 contact-risk checkpoint 已从交付目录清理，
实验结果仍保留在文档中。确需复现时从云端归档取回或重新训练。每次评估要固定
mode、seed、Diffusion seed、DDIM steps、guidance alpha、command 时间语义与
exposure。

## 训练流程

1. 用冻结 CW/CCW RL 专家采集 full-DR rollout；不要混入不满足契约的历史数据。
2. 数据加载时按 episode 随机分层划分 train/val，失败 episode 的结束前 2 秒不作为动作监督。
3. 先训练 IDM。用验证集 oracle future motion 检查 target MAE，确认关键点表征能够恢复 PD target。
4. 再训练 Diffusion。正式配置使用 `epsilon prediction`、cosine noise schedule 和 100 diffusion steps。
5. 用 `eval_offline.py` 测试 DDIM 步数、motion MAE/RMSE、幅值比与 Diffusion→IDM target MAE；线上默认 DDIM=25。
6. 在 Isaac 中进行 guided 闭环筛选，再以多 seed、1200 steps 与 noisy-RL / clean-expert 做公平对照。主要指标是 drop/env-minute、signed speed、wrong-way、generated→executed RMSE 与 IDM tracking。

## 触觉训练分支

- tactile-v1：将五指连续触觉标量直接拼入每帧状态。
- tactile-v2b：使用独立五指时序编码器、未来触觉辅助头和 recovery 重采样；仅作消融，当前不替代 v1。

触觉输入顺序固定为 `thumb/index/middle/ring/pinky`。所有 checkpoint 会记录 state dim、触觉开关和 architecture；78/83 维或 v1/v2b checkpoint 不可混用。

## 主动恢复数据与新版训练

旧 `large_modes_v5` 数据负责保留连续旋转分布；新版数据只作为主动恢复增强，不能替代旧数据。采集器轮换注入 action bias、单指短暂张开和物体外力，扰动阶段不作为动作监督，随后只记录干净 RL 专家的恢复动作。成功恢复必须同时满足：物体未掉落、至少三指接触、继续沿目标方向旋转。

CW 必须使用 `student_9402.pth`，CCW 必须使用 `student_9503.pth`。采集器会在启动后检查平均有向转速，方向不符则在保存前终止。正式数据默认每方向 10 个分片、每分片 1000 个 episode：

```bash
cd /home/user/rotate_dexgen
./dexgen/pipelines/collect_recovery_dataset.sh all 0 9
```

本机 8 GB 显存默认使用 50 env × 每 env 20 episode，确保小样本不会只保留最早跌落的环境。显存更充足的服务器可保持总数不变并增加并行数：

```bash
DEXGEN_NUM_ENVS=100 DEXGEN_EPISODES_PER_ENV=10 \
  ./dexgen/pipelines/collect_recovery_dataset.sh all 0 9
```

全部分片完成后先做总审计：

```bash
python dexgen/commands/data/audit_recovery_data.py --data \
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt
```

另开终端实时查看采集状态：

```bash
cd /home/user/rotate_dexgen
watch -n 5 python dexgen/tools/monitor_recovery_progress.py
```

训练必须混合旧稳定数据和新恢复数据，并使用 `--explicit_recovery_only` 排除旧版触觉启发式标签。`--recovery_fraction 0.25` 表示每个方向的训练采样中 25% 为明确成功恢复窗口、75% 为普通旋转窗口：

```bash
python dexgen/commands/train/train_idm.py --data \
  dexgen/data/large_modes_v5/cw_full_*.pt \
  dexgen/data/large_modes_v5/ccw_full_*.pt \
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt \
  --include_tactile --tactile_arch v2b \
  --explicit_recovery_only --recovery_fraction 0.25 \
  --tactile_aux_weight 0.005 --batch_size 512 --epochs 15 \
  --output dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_best.pt \
  --latest_output dexgen/checkpoints/idm_modes_full_dr_active_recovery_v3_latest.pt \
  --device cuda:0

python dexgen/commands/train/train_diffusion.py --data \
  dexgen/data/large_modes_v5/cw_full_*.pt \
  dexgen/data/large_modes_v5/ccw_full_*.pt \
  dexgen/data/recovery_modes_v6/cw_recovery_*.pt \
  dexgen/data/recovery_modes_v6/ccw_recovery_*.pt \
  --include_tactile --tactile_arch v2b \
  --explicit_recovery_only --recovery_fraction 0.25 \
  --tactile_aux_weight 0.005 --prediction_type epsilon --noise_schedule cosine \
  --batch_size 512 --epochs 15 \
  --output dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_best.pt \
  --latest_output dexgen/checkpoints/diffusion_modes_full_dr_active_recovery_v3_v2b_latest.pt \
  --device cuda:0
```

先验收 IDM 的 oracle-motion target 误差，再接受 Diffusion 结果。新版 checkpoint
使用独立文件名，不覆盖当前 active-recovery v3 正式模型。

## MANUS 遥操作流程

```text
MANUS → RetargetX → q_glove[21] → TCP 8020
      → DexGen FK → 标定后的 10 点相对 motion command
      → Guided Diffusion → IDM → 仿真 XHand
```

标定后，手套只提供外部 guidance，不进入 Diffusion/IDM state：

```text
x_goal  = x_robot_ref + gain * (FK(q_glove_now) - FK(q_glove_ref))
command = x_goal - FK(q_robot_now)
```

首版采用单环境、broadcast command、`gain=0.55`、`noise_alpha=0`。数据 stale、断连、reset 或关节契约失败时必须自动 disarm 并保持安全 PD target；不得把断连误解释为零位移。

## 修改与验证要求

修改数据、模型、协议或 evaluator 后至少运行：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
python -m pytest tests/test_dexgen_pipeline.py -q
```

修改 MANUS bridge 后还应运行 30 秒 `command_probe.py`；涉及 Isaac 闭环时，先做短 smoke test，再开始正式多 seed exposure。历史决策、实验数字和未完成事项记录在 [worklog.md](worklog.md)。
