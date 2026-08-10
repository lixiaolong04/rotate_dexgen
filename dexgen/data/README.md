# DexGen 数据目录

本项目交付时不包含历史训练数据集。数据文件体积较大、可由现有 RL 专家重新生成，
因此不上传 GitHub；本目录只保留说明文件。

正式 active-recovery 数据由两部分组成：

```text
dexgen/data/large_modes_v5/      # CW/CCW 稳定 full-DR 旋转数据
dexgen/data/recovery_modes_v6/   # CW/CCW 主动扰动恢复数据
```

CW 使用 `ckpts/overnight_pose5/student_9402.pth`，CCW 使用
`ckpts/overnight_ccw/student_9503.pth`。恢复数据的推荐采集命令为：

```bash
cd /home/user/rotate_dexgen
conda activate env_isaaclab
./dexgen/pipelines/collect_recovery_dataset.sh all 0 9
```

完整的采集、普通 full-DR 分片生成、数据审计和云端运行说明见
[`../HANDOVER.md`](../HANDOVER.md) 第 4 节。
