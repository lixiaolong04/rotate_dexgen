# UniLab V1 final student checkpoints

**★ student_8103 = 全项目真机冠军**(2026-06-30 用户硬件实测): 超越 9402 / t701, **360° 旋转 + 物体倒立(inverted)不掉** = 长期"物体被搓往上爬(creep)"基本解决。

## V1 配置(= 9402/pose5, 唯一改动是 mask 两指触觉)
- **唯一实质改动**: `disable_tactile_ids = [1, 4]` — 清零食指(1)/小指(4)接触触觉, sim2real 对齐真机(这俩指真机读≈0, sim 原读整指=OOD)。
- `enable_contact_force_vec = False` → **141 维 obs, 未开剪切**(剪切/治盲没用上, 光触觉对齐就解决了 creep)。
- 其余全 = 9402: pose5; 显式力矩 kp=10/kd=0.3/friction0.1; decimation=12; 重力 -9.81 满 + 全 SO(3); scale[0.4,0.7,10]; reward(rotate2.5 / linvel-0.3 / pos-0.4 / torque-0.1 / work-0.5); DR(friction/PD ×[0.5,2], COM±0.01, mass[0.01,0.25], 上推扰动 force2 prob0.25/步)。
- 训练: 云端 4×L20, num_envs=20480, 4 seed(8100-8103), teacher(PPO看priv)→500M, student(ProprioAdapt盲蒸)→~480M。
- **部署**: `deploy_cfg` 设 `disable_tactile_ids=[1,4]` + mode4 kp=10/kd=0.3/speed=0; obs 141。

Pulled after the 5-hour V1 student rerun completed on cloud.

Cloud source repo: `/root/era_dexhand_lab`
Cloud branch at training: `research_unilab_rotation`
Cloud commit observed: `6b59efc`
Completion log: `logs/v1_student_rerun.log` ended at `2026-06-30 00:10:43 PDT`.

Mapping:
- `student_8100_best.pth` <- `logs/debug/2026-06-29_02-50-57/stage2_nn/best.pth`
- `student_8101_best.pth` <- `logs/debug/2026-06-29_02-51-03/stage2_nn/best.pth`
- `student_8102_best.pth` <- `logs/debug/2026-06-29_02-51-09/stage2_nn/best.pth`
- `student_8103_best.pth` <- `logs/debug/2026-06-29_02-51-15/stage2_nn/best.pth`

Final distillation log tail:

| seed | steps | reward | best |
|---|---:|---:|---:|
| 8100 | 463M | 16.91 | 17.04 |
| 8101 | 458M | 16.38 | 16.41 |
| 8102 | 482M | 16.92 | 17.04 |
| 8103 | 487M | 18.39 | 18.43 |

Hardware-test order by distillation reward: 8103 first, then 8102/8100, then 8101.
For this V1 mask run, deploy with tactile ids 1,4 disabled.
