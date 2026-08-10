# DexGen 目录结构与迁移约定

本机与云服务器统一使用以下结构：

```text
dexgen/
├── commands/
│   ├── data/       # 数据采集和审计入口
│   ├── eval/       # 仿真与离线评估入口
│   └── train/      # 模型训练入口
├── pipelines/      # 数据、训练、验证流水线 Shell 脚本
├── tools/          # 进度监控和结果汇总工具
├── deploy/         # 实物 XHand 部署（云端可为空）
├── docs/           # 开发、实验和交接文档
├── data/           # 数据集，路径保持稳定
├── checkpoints/    # 模型权重，路径保持稳定
├── logs/           # 训练和评估日志，路径保持稳定
├── archive/        # 不参与运行的历史备份
└── *.py            # 被多个入口复用的核心库模块
```

## 运行入口

运行命令直接使用分类后的真实路径，例如：

```bash
python dexgen/commands/train/train_v8_policy.py --help
python dexgen/commands/eval/eval_offline.py --help
bash dexgen/pipelines/run_teacher_student_v8_pipeline.sh
```

新脚本直接创建在相应分类目录中。不要在 `dexgen/` 根目录复制脚本或创建兼容
链接，以免再次形成两套入口。

## 本机与云端迁移

项目根目录分别为：

- 本机：`/home/user/rotate_dexgen`
- 云端：`/root/rotate_dexgen`

训练后通常只需同步新增或更新的源码、配置和指定 checkpoint。不要无差别覆盖
两端的 `data/`、`checkpoints/`、`logs/`、`deploy/`，因为它们分别包含大型训练
资产、本地硬件部署代码或机器相关日志。

迁移前后应检查：

1. 源码相对路径保持为 `dexgen/...`，不要写死本机或云端项目绝对路径。
2. checkpoint 命令行参数仍使用 `dexgen/checkpoints/...`。
3. README、流水线及作业命令均使用分类后的真实路径。
4. 云端没有训练进程时再整理、替换或归档脚本。
