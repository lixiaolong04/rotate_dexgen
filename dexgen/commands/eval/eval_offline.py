"""离线对比 Oracle motion、Diffusion motion 与专家绝对 PD 目标。

该程序不启动 Isaac Sim，也不需要真实硬件。它使用验证集中的真实未来
关键点作为 oracle 输入来隔离 IDM，再用 Diffusion 生成的关键点运动评估
完整串联误差。v2 的 IDM 不再回归饱和的单步 RL 增量动作。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.data import (
    DexGenDataset,
    FeatureNormalizer,
    validate_checkpoint_representations,
    validate_checkpoint_state_contracts,
)
from dexgen.models import ConditionalMotionDiffusion, InverseDynamicsModel
from dexgen.v8_runtime import (
    V8_DIFFUSION_FORMAT,
    V8_IDM_FORMAT,
    load_v8_runtime,
    student_latent,
    teacher_latent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, nargs="+",
        default=[Path("dexgen/data/cylinder_rollouts_cw_v3.pt"), Path("dexgen/data/cylinder_rollouts_ccw_v3.pt")],
    )
    parser.add_argument("--diffusion", type=Path, default=Path("dexgen/checkpoints/diffusion_modes_v4_best.pt"))
    parser.add_argument("--idm", type=Path, default=Path("dexgen/checkpoints/idm_modes_v4_best.pt"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--latent_source",
        choices=("student", "teacher"),
        default="student",
        help="v8 only: deployment student or privileged teacher latent.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_models(diffusion_path: Path, idm_path: Path, device: torch.device):
    diffusion_ckpt = torch.load(diffusion_path, map_location=device, weights_only=False)
    idm_ckpt = torch.load(idm_path, map_location=device, weights_only=False)
    if diffusion_ckpt.get("format") not in {
        "xhand_dexgen_diffusion_v3", "xhand_dexgen_diffusion_v4",
        "xhand_dexgen_diffusion_v5", "xhand_dexgen_diffusion_tactile_v1",
        "xhand_dexgen_diffusion_tactile_v2b", "xhand_dexgen_diffusion_tactile_film_v3",
        V8_DIFFUSION_FORMAT,
    }:
        raise ValueError(f"不支持的 Diffusion checkpoint: {diffusion_ckpt.get('format')!r}")
    if idm_ckpt.get("format") not in {
        "xhand_dexgen_idm_v3", "xhand_dexgen_idm_v4",
        "xhand_dexgen_idm_v5", "xhand_dexgen_idm_tactile_v1",
        "xhand_dexgen_idm_tactile_v2b",
        V8_IDM_FORMAT,
    }:
        raise ValueError(f"不支持的 IDM checkpoint: {idm_ckpt.get('format')!r}")
    validate_checkpoint_representations(diffusion_ckpt, idm_ckpt)
    state_info = validate_checkpoint_state_contracts(diffusion_ckpt, idm_ckpt)

    diffusion = ConditionalMotionDiffusion(**diffusion_ckpt["model_config"]).to(device)
    diffusion.load_state_dict(diffusion_ckpt["model"])
    diffusion.eval()
    idm = InverseDynamicsModel(**idm_ckpt["model_config"]).to(device)
    idm.load_state_dict(idm_ckpt["model"])
    idm.eval()

    diffusion_dataset_cfg = diffusion_ckpt["dataset_config"]
    idm_dataset_cfg = idm_ckpt["dataset_config"]
    for key in ("history_steps", "prediction_steps", "future_stride", "history_stride", "sample_stride"):
        if diffusion_dataset_cfg[key] != idm_dataset_cfg[key]:
            raise ValueError(f"Diffusion 与 IDM 的 {key} 不一致")
    if diffusion.state_dim != idm.state_dim or diffusion.keypoint_count != idm.keypoint_count:
        raise ValueError("Diffusion 与 IDM 的状态或关键点维度不一致")

    effective_dataset_cfg = {**diffusion_dataset_cfg, **state_info}
    v8_runtime = load_v8_runtime(diffusion_ckpt, idm_ckpt, device)
    return (
        diffusion,
        idm,
        FeatureNormalizer.from_state_dict(diffusion_ckpt["normalizer"]),
        FeatureNormalizer.from_state_dict(idm_ckpt["normalizer"]),
        effective_dataset_cfg,
        v8_runtime,
    )


def pearson_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().flatten()
    target = target.float().flatten()
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = prediction.square().sum().sqrt() * target.square().sum().sqrt()
    return float((prediction * target).sum() / denominator.clamp_min(1.0e-12))


def adjacent_l1(
    values: torch.Tensor, episode_ids: torch.Tensor, steps: torch.Tensor, step_delta: int
) -> tuple[float, int]:
    """计算同一 episode 连续控制步之间的平均绝对变化。"""
    if values.shape[0] < 2:
        return float("nan"), 0
    adjacent = (episode_ids[1:] == episode_ids[:-1]) & (steps[1:] == steps[:-1] + step_delta)
    count = int(adjacent.sum())
    if count == 0:
        return float("nan"), 0
    delta = (values[1:] - values[:-1]).abs().flatten(1).mean(-1)
    return float(delta[adjacent].mean()), count


def print_target_report(
    name: str,
    prediction: torch.Tensor,
    expert: torch.Tensor,
    episode_ids: torch.Tensor,
    steps: torch.Tensor,
    step_delta: int,
) -> dict[str, float]:
    error = prediction - expert
    d_target, pairs = adjacent_l1(prediction, episode_ids, steps, step_delta=step_delta)
    metrics = {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "corr": pearson_correlation(prediction, expert),
        "dtarget": d_target,
    }
    print(
        f"{name:<24} MAE={metrics['mae']:.5f}  RMSE={metrics['rmse']:.5f}  "
        f"corr={metrics['corr']:+.4f}  mean|dTarget|={metrics['dtarget']:.5f} rad  "
        f"(adjacent pairs={pairs})"
    )
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    diffusion, idm, diffusion_norm, idm_norm, dataset_cfg, v8_runtime = load_models(
        args.diffusion, args.idm, device
    )

    dataset = DexGenDataset(
        rollout_path=args.data,
        split=args.split,
        history_steps=dataset_cfg["history_steps"],
        prediction_steps=dataset_cfg["prediction_steps"],
        future_stride=dataset_cfg["future_stride"],
        history_stride=dataset_cfg["history_stride"],
        sample_stride=dataset_cfg["sample_stride"],
        validation_fraction=dataset_cfg.get("validation_fraction", 0.1),
        split_seed=dataset_cfg.get("split_seed", 42),
        drop_trim_seconds=dataset_cfg["drop_trim_seconds"],
        include_tactile=bool(dataset_cfg["include_tactile"]),
    )
    recovery_mask = None
    if dataset_cfg.get("tactile_arch") in {"v2b", "film_v3"}:
        _, recovery_mask = dataset.recovery_sample_weights(
            boost=1.0,
            contact_threshold=dataset_cfg.get("recovery_contact_threshold", 0.1),
            low_contact_count=dataset_cfg.get("recovery_low_contact_count", 2),
            stable_contact_count=dataset_cfg.get("recovery_stable_contact_count", 4),
            horizon_seconds=dataset_cfg.get("recovery_horizon_seconds", 0.5),
        )
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max_samples 必须为正数")
        sample_count = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, range(sample_count))
        if recovery_mask is not None:
            recovery_mask = recovery_mask[:sample_count]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    expert_targets = []
    oracle_targets = []
    diffusion_targets = []
    true_motions = []
    generated_motions = []
    generated_motion_z = []
    episode_ids = []
    steps = []
    latent_squared_error = latent_count = 0.0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            state = batch["state"].to(device, non_blocking=True)
            true_motion = batch["motion"].to(device, non_blocking=True)
            expert_target = batch["target"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)

            diffusion_state = diffusion_norm.normalize_state(state)
            if v8_runtime is None:
                env_latent = None
            else:
                student_z = student_latent(v8_runtime, diffusion_state)
                if args.latent_source == "teacher" and not bool(
                    batch["priv_info_valid"].all()
                ):
                    raise ValueError(
                        "Teacher-latent evaluation requires priv_info on every sample"
                    )
                teacher_z = teacher_latent(v8_runtime, batch["priv_info"])
                latent_squared_error += float((student_z - teacher_z).square().sum())
                latent_count += student_z.numel()
                env_latent = student_z if args.latent_source == "student" else teacher_z
            generated_z = diffusion.sample(
                diffusion_state,
                ddim_steps=args.ddim_steps,
                mode=mode,
                env_latent=env_latent,
            )
            generated_motion = diffusion_norm.denormalize_motion(generated_z)

            idm_state = idm_norm.normalize_state(state)
            oracle_motion_z = idm_norm.normalize_motion(true_motion)
            generated_motion_idm_z = idm_norm.normalize_motion(generated_motion)
            oracle_target_z = idm.act(
                idm_state, oracle_motion_z, env_latent=env_latent
            )
            diffusion_target_z = idm.act(
                idm_state, generated_motion_idm_z, env_latent=env_latent
            )
            oracle_target = idm_norm.denormalize_target(oracle_target_z)
            diffusion_target = idm_norm.denormalize_target(diffusion_target_z)

            expert_targets.append(expert_target.cpu())
            oracle_targets.append(oracle_target.cpu())
            diffusion_targets.append(diffusion_target.cpu())
            true_motions.append(true_motion.cpu())
            generated_motions.append(generated_motion.cpu())
            generated_motion_z.append(generated_motion_idm_z.cpu())
            episode_ids.append(batch["episode_id"].cpu())
            steps.append(batch["step"].cpu())
            if batch_index % 10 == 0:
                print(f"[offline-eval] processed {batch_index * args.batch_size} samples", flush=True)

    expert = torch.cat(expert_targets)
    oracle = torch.cat(oracle_targets)
    generated_target = torch.cat(diffusion_targets)
    true_motion = torch.cat(true_motions)
    generated_motion = torch.cat(generated_motions)
    generated_z = torch.cat(generated_motion_z)
    episode_id = torch.cat(episode_ids)
    step = torch.cat(steps)

    motion_error_mm = (generated_motion - true_motion).abs() * 1000.0
    true_motion_abs_mm = true_motion.abs().mean() * 1000.0
    generated_motion_abs_mm = generated_motion.abs().mean() * 1000.0
    sample_stride = int(dataset_cfg["sample_stride"])
    true_dmotion, motion_pairs = adjacent_l1(true_motion, episode_id, step, sample_stride)
    generated_dmotion, _ = adjacent_l1(generated_motion, episode_id, step, sample_stride)
    expert_dtarget, target_pairs = adjacent_l1(expert, episode_id, step, sample_stride)

    print("\n" + "=" * 92)
    print("DEXGEN OFFLINE ORACLE COMPARISON")
    print("=" * 92)
    print(f"split={args.split}  samples={len(expert)}  ddim_steps={args.ddim_steps}")
    if v8_runtime is not None:
        print(
            f"v8 latent source={args.latent_source}  "
            f"student->teacher MSE={latent_squared_error / max(latent_count, 1):.6f}"
        )
    print("\n[Motion: Diffusion vs recorded future motion]")
    print(f"motion MAE:                 {float(motion_error_mm.mean()):.4f} mm")
    print(f"motion RMSE:                {float(motion_error_mm.square().mean().sqrt()):.4f} mm")
    print(f"true mean |motion|:         {float(true_motion_abs_mm):.4f} mm")
    print(f"generated mean |motion|:    {float(generated_motion_abs_mm):.4f} mm")
    print(f"motion amplitude ratio:     {float(generated_motion_abs_mm / true_motion_abs_mm.clamp_min(1e-8)):.4f}")
    print(f"true mean |dMotion|:        {true_dmotion * 1000.0:.4f} mm")
    print(f"generated mean |dMotion|:   {generated_dmotion * 1000.0:.4f} mm  (pairs={motion_pairs})")
    print(f"generated motion |z| > 3:   {float((generated_z.abs() > 3.0).float().mean()) * 100:.3f}%")

    print("\n[Absolute PD target: predictions vs recorded RL expert]")
    print(f"{'RL expert reference':<24} MAE=0.00000  RMSE=0.00000  corr=+1.0000  "
          f"mean|dTarget|={expert_dtarget:.5f} rad  (adjacent pairs={target_pairs})")
    oracle_metrics = print_target_report(
        "Oracle motion -> IDM", oracle, expert, episode_id, step, sample_stride
    )
    diffusion_metrics = print_target_report(
        "Diffusion -> IDM", generated_target, expert, episode_id, step, sample_stride
    )

    per_joint_oracle = (oracle - expert).abs().mean(0)
    per_joint_diffusion = (generated_target - expert).abs().mean(0)
    worst_oracle = torch.topk(per_joint_oracle, k=min(5, per_joint_oracle.numel()))
    worst_diffusion = torch.topk(per_joint_diffusion, k=min(5, per_joint_diffusion.numel()))
    print("\n[Worst joint targets: index=MAE(rad)]")
    print("oracle:    " + ", ".join(
        f"{int(index)}={float(value):.4f}" for value, index in zip(worst_oracle.values, worst_oracle.indices)
    ))
    print("diffusion: " + ", ".join(
        f"{int(index)}={float(value):.4f}" for value, index in zip(worst_diffusion.values, worst_diffusion.indices)
    ))

    if recovery_mask is not None:
        print("\n[Tactile recovery subsets]")
        for name, selected in (
            ("recovery", recovery_mask),
            ("regular", ~recovery_mask),
        ):
            count_selected = int(selected.sum())
            if count_selected == 0:
                print(f"{name:<10} samples=0")
                continue
            motion_selected = (generated_motion[selected] - true_motion[selected]) * 1000.0
            oracle_selected = oracle[selected] - expert[selected]
            diffusion_selected = generated_target[selected] - expert[selected]
            print(
                f"{name:<10} samples={count_selected:<7d} "
                f"motion_MAE={float(motion_selected.abs().mean()):.4f} mm "
                f"motion_RMSE={float(motion_selected.square().mean().sqrt()):.4f} mm "
                f"oracle_target_MAE={float(oracle_selected.abs().mean()):.5f} rad "
                f"diff_target_MAE={float(diffusion_selected.abs().mean()):.5f} rad"
            )

    print("\n[Automatic diagnosis]")
    oracle_good = oracle_metrics["mae"] < 0.05 and oracle_metrics["corr"] > 0.95
    diffusion_error_ratio = diffusion_metrics["mae"] / max(oracle_metrics["mae"], 1.0e-8)
    jitter_vs_expert = diffusion_metrics["dtarget"] / max(expert_dtarget, 1.0e-8)
    amplitude_ratio = float(generated_motion_abs_mm / true_motion_abs_mm.clamp_min(1e-8))
    if oracle_good and diffusion_error_ratio > 1.5:
        print(
            f"Oracle IDM 映射良好，但 Diffusion 使目标角 MAE 放大到 oracle 的 "
            f"{diffusion_error_ratio:.2f} 倍；主要误差来自生成 motion。"
        )
        if jitter_vs_expert > 1.5:
            print("Diffusion 串联目标角也显著比专家更抖；需要优先修复在线 motion 时序连续性。")
        else:
            print(
                f"Diffusion 目标变化仅为专家的 {jitter_vs_expert:.2f} 倍，未明显恶化。"
            )
    elif not oracle_good:
        print("Oracle IDM 目标角误差仍偏大；在调整 Diffusion 前先改善 IDM 或关键点表征。")
    else:
        print("Oracle 与 Diffusion 目标角差距有限；需要结合仿真闭环指标继续定位误差放大来源。")
    if amplitude_ratio < 0.8:
        print(f"Diffusion motion 幅值只有真实 motion 的 {amplitude_ratio:.2f} 倍，存在系统性低估。")
    elif amplitude_ratio > 1.2:
        print(f"Diffusion motion 幅值达到真实 motion 的 {amplitude_ratio:.2f} 倍，存在系统性放大。")
    if float((generated_z.abs() > 3.0).float().mean()) > 0.01:
        print("部分生成 motion 超出 IDM 训练分布（|z|>3）；需要限制采样或增强数据覆盖。")
    print("=" * 92)


if __name__ == "__main__":
    main()
