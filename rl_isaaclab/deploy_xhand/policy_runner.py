# rl_isaaclab/deploy_xhand/policy_runner.py
"""复用 rl_isaaclab.algo.padapt.ProprioAdapt 加载 student 并做推理。无 isaac 依赖。"""
import os
import torch
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt

# ppo_cfg.yaml 的等价内容 (避免 import isaac 的 hydra 加载链)
_AGENT_CFG = {
    "seed": 42, "algo": "ProprioAdapt", "device": "cpu", "load_path": None,
    "network": {"mlp": {"units": [512, 256, 128]}, "priv_mlp": {"units": [256, 128, 9]}},   # 2026-06-17 对齐 UniLab: priv embed 8→9 (加物体scale通道); 必须与 student ckpt 一致否则 strict load 崩
    "algorithm": {
        "experiment_name": "deploy", "num_actors": 1,
        "priv_info": True, "priv_info_dim": 9, "priv_info_embed_dim": 9,   # 2026-06-17 对齐 UniLab: 8→9
        "normalize_input": True, "normalize_value": True,
    },
}


class _CfgWrap:
    def __init__(self, train, task=None, test=True):
        self.train = train; self.task = task; self.test = test


class PolicyRunner:
    def __init__(self, env, ckpt_path, device="cpu"):
        assert os.path.exists(ckpt_path), f"ckpt 不存在: {ckpt_path}"
        cfg = dict(_AGENT_CFG); cfg["device"] = device
        full = _CfgWrap(cfg)
        self.agent = ProprioAdapt(env, output_dir="/tmp/xhand_deploy",
                                  full_config=full, create_output_dir=False)
        # CPU-safe 加载: checkpoint 在 CUDA 上保存, 真机部署(容器)用 CPU-only torch;
        # padapt.restore_test 内部 torch.load 不带 map_location, 无 GPU 时会崩
        # (Attempting to deserialize object on a CUDA device)。故这里显式按 device 加载,
        # 不改共享的训练 algo 代码。等价于 restore_test 的三件套加载。
        ckpt = torch.load(ckpt_path, map_location=device)
        self.agent.running_mean_std.load_state_dict(ckpt["running_mean_std"])
        self.agent.model.load_state_dict(ckpt["model"])
        self.agent.sa_mean_std.load_state_dict(ckpt["sa_mean_std"])
        self.agent.set_eval()
        self.device = device

    @torch.no_grad()
    def act(self, obs_dict):
        inp = {
            "obs": self.agent.running_mean_std(obs_dict["obs"].to(self.device)),
            "proprio_hist": self.agent.sa_mean_std(obs_dict["proprio_hist"].to(self.device)),
        }
        mu = self.agent.model.act_inference(inp)
        return torch.clamp(mu, -1.0, 1.0).cpu()
