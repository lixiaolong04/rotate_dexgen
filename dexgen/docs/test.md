# DexGen 测试命令记录（2026-07-15）

本文件只记录当前 XHand 圆柱旋转 pipeline 已经使用或准备使用的测试命令。完整原理和数据格式见 `dexgen/README.md`。

## 0. 运行环境

每次打开新终端先执行：

```bash
conda activate env_isaaclab
cd /root/rotate_dexgen
```

当前主线文件必须成组使用：

| 类型 | 当前主线 | 历史第一版 |
|---|---|---|
| rollout | `cylinder_rollouts_v2.pt` | `cylinder_rollouts.pt` |
| Diffusion | `diffusion_v3_best.pt`，78 维 state | `diffusion_best.pt`，98 维 state |
| IDM | `idm_v3_best.pt`，绝对 PD target | `idm_best.pt`，旧增量 action |
| 在线 evaluator | `eval_dexgen.py` | 当前没有保留独立的 v1 evaluator |

禁止把同一列以外的 data、Diffusion、IDM 或 evaluator 混用。

## 1. RL 专家 8500 基线

这是当前 DexGen 数据来源和性能上限参考。`frontpad` 必须与 8500 的训练观测对齐：

```bash
python rl_isaaclab/scripts/eval_batch.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --algorithm ProprioAdapt \
  --ckpt_glob ckpts/frontpad_students_20260701/student_8500.pth \
  --friction_scales 1.0 \
  --rollout_windows 1 \
  --sensor_profile frontpad \
  --so3 \
  --device cuda:0 \
  --headless
```

2026-07-15 已得到的参考结果：drop `15.0%`，rotation `+0.99 rad/s`，revolutions `+2.71`，wrong-way `9.2%`，`dAct=0.555`。

## 2. 8103 对照（可选）

8103 使用不同的触觉 mask，必须指定 `v1_mask`：

```bash
python rl_isaaclab/scripts/eval_batch.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --algorithm ProprioAdapt \
  --ckpt_glob ckpts/unilab_v1_students_final_20260630/student_8103_best.pth \
  --friction_scales 1.0 \
  --rollout_windows 1 \
  --sensor_profile v1_mask \
  --so3 \
  --device cuda:0 \
  --headless
```

2026-07-15 已得到的参考结果：drop `15.0%`，rotation `+0.95 rad/s`，revolutions `+2.50`，wrong-way `9.2%`，`dAct=0.552`。当前继续使用 8500。

## 3. v2 数据采集

当前 100 episode 数据已经采集完成，通常不需要重复运行：

```bash
python dexgen/commands/data/collect_rl_data.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --algorithm ProprioAdapt \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth \
  --episodes 100 \
  --output dexgen/data/cylinder_rollouts_v2.pt \
  --so3 \
  --device cuda:0 \
  --headless
```

已得到：100 episodes，38653 transitions，drop `4.0%`。

## 4. v3 模型训练

IDM：

```bash
python dexgen/commands/train/train_idm.py \
  --data dexgen/data/cylinder_rollouts_v2.pt \
  --output dexgen/checkpoints/idm_v3_best.pt \
  --batch_size 512 \
  --epochs 50
```

已得到 best normalized target val MSE `0.010988`。

Diffusion：

```bash
python dexgen/commands/train/train_diffusion.py \
  --data dexgen/data/cylinder_rollouts_v2.pt \
  --output dexgen/checkpoints/diffusion_v3_best.pt \
  --batch_size 512 \
  --epochs 50
```

已得到 best val `0.021858`。

## 5. v3 离线 Oracle 对比

```bash
python dexgen/commands/eval/eval_offline.py \
  --data dexgen/data/cylinder_rollouts_v2.pt \
  --diffusion dexgen/checkpoints/diffusion_v3_best.pt \
  --idm dexgen/checkpoints/idm_v3_best.pt \
  --split val \
  --ddim_steps 12
```

已得到的关键结果：

- motion MAE `0.5078 mm`，幅值比例 `0.9563`；
- Oracle motion → IDM target MAE `0.00487 rad`；
- Diffusion → IDM target MAE `0.01141 rad`。

## 6. 无引导 v3 DexGen 闭环

量化测试：

```bash
python dexgen/commands/eval/eval_dexgen.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --diffusion dexgen/checkpoints/diffusion_v3_best.pt \
  --idm dexgen/checkpoints/idm_v3_best.pt \
  --steps 400 \
  --ddim_steps 12 \
  --so3 \
  --device cuda:0 \
  --headless
```

给同事可视化时去掉 `--headless`，也可以把 `--num_envs` 改成 10。`--so3` 下环境数量必须是 10 的倍数。

此前 400 步结果：drop `61.29%`，rotation `+0.8197 rad/s`，revolutions `+1.5380`，wrong-way `16.06%`。

## 7. 历史 v1 命令为什么会报错

下面是历史第一版命令，保留用于说明，**当前不能直接运行**：

```bash
python dexgen/commands/eval/eval_dexgen.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 10 \
  --diffusion dexgen/checkpoints/diffusion_best.pt \
  --idm dexgen/checkpoints/idm_best.pt \
  --steps 400 \
  --ddim_steps 8 \
  --so3
```

原因不是显卡或 `num_envs=10`，而是：

```text
diffusion_best.pt = xhand_dexgen_diffusion_v1，98 维 state
idm_best.pt       = xhand_dexgen_idm_v1，98 维 state，旧 action 目标
eval_dexgen.py    = 只支持 v3，78 维 state，绝对 PD target
```

因此通常会看到：

```text
ValueError: Unsupported diffusion checkpoint: xhand_dexgen_diffusion_v1
```

如果只是给同事展示当前 pipeline，请使用第 6 节的 v3 checkpoint。若必须复现历史第一版，需要单独恢复 `eval_dexgen_v1.py`，不能强行让 v1 checkpoint 走 v3 的状态和 target 适配逻辑。

## 8. 零引导正式基线（下一步）

该命令和第 6 节不同：它通过新的 guided evaluator 运行，但令 `guidance_strength=0`，用于验证 guided evaluator 自身没有改变无引导策略分布。

```bash
python dexgen/commands/eval/eval_guided.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth \
  --diffusion dexgen/checkpoints/diffusion_v3_best.pt \
  --idm dexgen/checkpoints/idm_v3_best.pt \
  --controller guided \
  --guidance_strength 0 \
  --noise_alpha 0 \
  --steps 400 \
  --ddim_steps 12 \
  --so3 \
  --device cuda:0 \
  --headless
```

这一项运行完成后，保存完整的 `DEXGEN GUIDED CLOSED-LOOP SUMMARY`。非零 guidance 的单位问题已在第 9 节记录，修正后的首轮扫描使用 `guidance_strength=100000/300000/1000000`。

2026-07-15 已完成的零引导结果：

```text
episodes:          34
drop rate:         64.71%
rotation speed:    +0.8851 rad/s
revolutions/ep:    +1.3253
wrong-way:         12.16%
mean |dAction|:    0.56758
mean |dTarget|:    0.05595 rad @ 10 Hz
command |motion|:  2.9617 mm
generated |motion|:2.4443 mm
command RMSE:      3.2894 mm
Jacobian check:    mean=0.0318, max=0.4371 mm/s
```

与第 6 节无引导 v3 结果相比，drop 为 `64.71% vs 61.29%`，`dAction` 为 `0.56758 vs 0.56943`，`dTarget` 为 `0.05595 vs 0.05494 rad`。在只有 31～34 个结束 episode 的条件下，这属于同一性能范围，说明新 evaluator 在 `alpha=0` 时没有引入明显控制偏差。此时 guidance 被关闭，command RMSE 不应被当作控制误差结论，只用作非零 guidance 的相对起点。

## 9. Guidance 和 noisy-command 对照（零引导完成后再运行）

### 9.1 `guidance_strength=100, noise=0`（旧单位实现，仅作诊断）

```text
episodes:          29
drop rate:         72.41%
rotation speed:    +0.8006 rad/s
revolutions/ep:    +1.1666
wrong-way:         12.85%
mean |dAction|:    0.59556
mean |dTarget|:    0.05377 rad @ 10 Hz
command |motion|:  3.0316 mm
generated |motion|:2.4830 mm
command RMSE:      1.9648 mm
Jacobian check:    mean=0.0330, max=0.4838 mm/s
```

该结果证明 guidance 路径会影响生成 motion，但随后发现旧实现把归一化 diffusion covariance 直接用于米制 motion 更新，漏掉了 `motion_std` 链式因子，导致 guidance 被放大约 `1/motion_std²`。该数值不能进入修正后的 alpha 曲线。

### 9.2 `guidance_strength=30, noise=0`（旧单位实现，仅作诊断）

```text
episodes:          29
drop rate:         75.86%
rotation speed:    +0.7451 rad/s
revolutions/ep:    +1.0616
wrong-way:         15.77%
mean |dAction|:    0.59444
mean |dTarget|:    0.05223 rad @ 10 Hz
command |motion|:  3.0214 mm
generated |motion|:2.4397 mm
command RMSE:      2.1197 mm
Jacobian check:    mean=0.0319, max=0.3400 mm/s
```

`alpha=30` 和 `alpha=100` 很快进入相近的强修正区，促使我们检查 guidance 单位并定位上述问题。2026-07-15 已将更新改为归一化变量上的 `Sigma_z * grad_z Dist`。零引导控制结果不受影响；所有非零 alpha 需要使用修正后的代码重新测试。

### 9.3 修正后的离线 guidance 量级检查

在同一批验证状态、同一随机初始噪声下，以真实第一帧 motion 作为单帧 command，12 步 DDIM 得到：

```text
alpha=0         command RMSE=1.9941 mm
alpha=3000      command RMSE=1.9907 mm
alpha=10000     command RMSE=1.9827 mm
alpha=30000     command RMSE=1.9599 mm
alpha=100000    command RMSE=1.8822 mm
alpha=300000    command RMSE=1.6878 mm
alpha=1000000   command RMSE=1.3597 mm
```

所有样本的最大 normalized motion 均小于 `3.2`，尚未触发 `|z|=5` clamp。论文没有公布 motion 单位和 normalization 细节，所以 Figure 8 的横轴数值不能直接迁移到 XHand；当前米制实现从 `alpha=100000` 开始做在线扫描。

### 9.4 修正后的在线 guidance 扫描（seed=123，400 步）

2026-07-16 使用相同 checkpoint、20 env、DDIM 12、`noise_alpha=0` 完成首轮调参：

| alpha | episodes | drop | rotation (rad/s) | rev/ep | wrong-way | dAction | dTarget (rad) | command RMSE (mm) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 34 | 64.71% | +0.8851 | +1.3253 | 12.16% | 0.56758 | 0.05595 | 3.2894 |
| 100000 | 25 | 48.00% | +0.8531 | +1.7391 | 11.94% | 0.56122 | 0.05646 | 3.0694 |
| 300000 | 22 | 40.91% | +0.8604 | +2.0198 | 12.35% | 0.55289 | 0.05663 | 2.7298 |
| 1000000 | 22 | 36.36% | +0.8430 | +1.8774 | 11.18% | 0.55421 | 0.05645 | 2.1892 |

Jacobian mean 依次为 `0.0318/0.0315/0.0331/0.0346 mm/s`，始终接近 0。本轮中 guidance 增强时 command RMSE 单调下降，drop 也同步下降，且转速、dAction 和 dTarget 未明显恶化。因此暂定 `BEST_ALPHA=1000000` 进入 noisy-command 对照。

这只是单 seed、400 步的 alpha 调试；每组仅有 22～34 个结束 episode，不能用作最终统计结论。正式结论需要 1200 步、多 seed，并补充右删失友好的持握时长指标。

Guided DexGen 示例：

```bash
python dexgen/commands/eval/eval_guided.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth \
  --diffusion dexgen/checkpoints/diffusion_v3_best.pt \
  --idm dexgen/checkpoints/idm_v3_best.pt \
  --controller guided \
  --guidance_strength 1000000 \
  --noise_alpha 0.1 \
  --steps 400 \
  --ddim_steps 12 \
  --so3 \
  --device cuda:0 \
  --headless
```

直接执行带噪 RL 的公平对照：

```bash
python dexgen/commands/eval/eval_guided.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth \
  --controller noisy_rl \
  --noise_alpha 0.1 \
  --steps 400 \
  --so3 \
  --device cuda:0 \
  --headless
```

### 9.5 `noise_alpha=0.1` 的首轮对照（seed=123，400 步）

| controller | alpha | noise | episodes | drop | rotation | rev/ep | wrong-way | dAction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| guided | 1000000 | 0.1 | 23 | 47.83% | +0.8317 rad/s | +1.7297 | 12.35% | 0.55775 |
| noisy_rl | N/A | 0.1 | 20 | 10.00% | +0.9765 rad/s | +2.8130 | 9.10% | 0.55851 |

Guided 的 command RMSE 为 `2.1533 mm`，dTarget 为 `0.05630 rad @ 10 Hz`，Jacobian mean/max 为 `0.0345/0.3880 mm/s`。noisy RL 的 Jacobian mean/max 为 `0.0363/0.3798 mm/s`。

当前结果不支持“DexGen motion prior 比直接 noisy command 更安全”：guidance 虽然显著改善了 DexGen 自身的 alpha=0 基线，但仍明显弱于直接 noisy RL。两组只有 20～23 个结束 episode，需要用持握时长/右删失指标和多 seed 复核；在那之前不应将这一对照写成 Figure 8 式正向结论。

### 9.6 新增右删失/持握时长指标后的复跑

`eval_guided.py` 现在额外输出 drops/timeouts、censored 数、每环境分钟掉落数、结束 episode 时长、time-to-fall、timeout 时长和未完成轨迹年龄。10 步 smoke test 确认 20 条 0.5 s 轨迹全部被正确标为 unfinished/censored。

相同 seed=123、400 步复跑：

| controller | drops/timeouts | drop/env-min | mean ended | mean time-to-fall | mean timeout | rotation |
|---|---:|---:|---:|---:|---:|---:|
| guided, alpha=1e6, noise=0.1 | 15 / 11 | 2.2500 | 11.127 s | 4.657 s | 19.950 s | +0.8133 rad/s |
| noisy_rl, noise=0.1 | 2 / 18 | 0.3000 | 18.102 s | 1.475 s | 19.950 s | +0.9679 rad/s |

两组总暴露时间均为 `20 env * 400 step * 0.05 s = 400 env-s`。Guided 的单位暴露时间掉落频率是 noisy RL 的 7.5 倍，因此否定结论不是 completed-only drop rate 的伪影。`mean time-to-fall` 只对已掉落样本求平均，会受选择偏差影响，不可单独用作策略排名。

Guided 的同 seed 复跑从前次 11 drops 变为 15 drops，而 noisy RL 仍为 2 drops，表明 diffusion/PhysX 路径存在实际运行非确定性。正式实验需多次独立运行/多 seed 置信区间，不应把单个 seed 标签视为完全可复现。

## 10. 分段 motion 误差链诊断

### 10.1 Command 对 RL 真实 motion

```bash
python dexgen/commands/eval/eval_guided.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth \
  --diffusion dexgen/checkpoints/diffusion_v3_best.pt \
  --idm dexgen/checkpoints/idm_v3_best.pt \
  --controller command_diag \
  --noise_alpha 0 \
  --steps 200 \
  --so3 \
  --device cuda:0 \
  --headless
```

seed=123 输出：

```text
command -> RL actual @0.1s: samples=1998 RMSE=2.3660 mm MAE=1.5010 mm |pred|=2.9980 mm |actual|=2.0774 mm
command -> RL actual @0.2s: samples=1976 RMSE=3.9321 mm MAE=2.7931 mm |pred|=2.9979 mm |actual|=3.2122 mm
RL desired target -> actual joints @0.1s: samples=1998 RMSE=0.0406 rad MAE=0.0303 rad
```

### 10.2 Guided 完整误差链

`alpha=1000000`、`noise_alpha=0`、200 步：

```text
command -> closed-loop executed @0.1s:   RMSE=2.2505 mm
generated[0] -> closed-loop @0.1s:       RMSE=0.7005 mm
command -> closed-loop executed @0.2s:   RMSE=3.3852 mm
generated[1] -> closed-loop @0.2s:       RMSE=1.5115 mm
IDM target -> actual joints @0.1s:       RMSE=0.0220 rad
planning-time command -> generated:      aggregate RMSE=2.2156 mm
```

诊断队列在 10 步 command/guided smoke test 和 200 步运行中均通过；0.1/0.2 s 的 valid sample 数会正确排除中途 reset 的 env。结果指向 command 时间/方向语义偏差，而不是 IDM target 跟踪失控。

## 11. Command 时间语义对照

新增 `--command_time_mode broadcast|cumulative|separate`。当前推荐候选命令：

```bash
python dexgen/commands/eval/eval_guided.py \
  --task Isaac-Inhand-Rotate-Xhand-Wave-v0 \
  --num_envs 20 \
  --load_path ckpts/frontpad_students_20260701/student_8500.pth \
  --diffusion dexgen/checkpoints/diffusion_v3_best.pt \
  --idm dexgen/checkpoints/idm_v3_best.pt \
  --controller guided \
  --command_time_mode separate \
  --command_horizon_steps 1 \
  --command_second_horizon_steps 2 \
  --guidance_strength 1000000 \
  --noise_alpha 0.1 \
  --steps 400 \
  --ddim_steps 12 \
  --so3 \
  --device cuda:0 \
  --headless
```

400 步结果：

| mode/noise | drops/timeouts | drop/env-min | rotation | command RMSE |
|---|---:|---:|---:|---:|
| broadcast, 0 | 9 / 13 | 1.35 | +0.8532 | 2.1986 mm |
| separate, 0 | 8 / 15 | 1.20 | +0.8003 | 1.9392 mm |
| broadcast, 0.1 | 16 / 10 | 2.40 | +0.8270 | 2.1533 mm |
| separate, 0.1 | 6 / 14 | 0.90 | +0.8381 | 1.9090 mm |

Separate 将带噪 guided 的掉落频率降低 62.5%，但仍弱于同条件 noisy RL 的 `0.30 drops/env-minute`。该结果目前只是单 seed/单次运行，不能作为最终显著性结论。

## 12. 精确 FK、真实历史 warm-up 与固定 latent（2026-07-16）

guided command 已从 `J(q) delta_q` 替换为 `FK(q_coarse)-FK(q_actual)`。启动时 FK 与 Isaac Sim 刚体关键点的一致性为 mean/max `0.001/0.002 mm`。每次 reset 后先用 RL 原语填满 4 帧真实 10 Hz 历史；Diffusion 使用独立 `--diffusion_seed=2026`，使不同 guidance 配置使用可重复的初始 latent 序列。

相同 `seed=123`、20 env、400 步、SO(3) 条件：

| controller | noise | drops/timeouts | drop/env-min | rotation | command RMSE |
|---|---:|---:|---:|---:|---:|
| alpha=0 | 0 | 22 / 11 | 3.30 | +0.8671 | 3.5062 mm |
| broadcast h2, alpha=1e6 | 0.1 | 12 / 11 | 1.80 | +0.9435 | 2.3550 mm |
| separate h1=1,h2=2, alpha=1e6 | 0.1 | 4 / 16 | 0.60 | +0.8688 | 2.2292 mm |
| noisy RL | 0.1 | 2 / 18 | 0.30 | +0.9679 | N/A |

精确 FK + warm-up 将 separate guided 从旧结果 `0.90` 改善到 `0.60 drops/env-minute`，且仍显著优于 broadcast；但没有反超 noisy RL。IDM 的 generated→executed 误差只有 `0.6198 mm @0.1s`、`1.2793 mm @0.2s`，不支持为了 1.5 倍启发式立即把 K=10 改成 K=11。`alpha=0` 仍较弱，后续结构消融应优先针对 Diffusion，而不是重做关键点或 IDM。
