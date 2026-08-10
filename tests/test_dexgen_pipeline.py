from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dexgen.data import (
    DexGenDataset,
    FeatureNormalizer,
    PROPRIO_STATE_SCHEMA,
    TACTILE_ARCH_DIRECT,
    TACTILE_ARCH_FILM,
    TACTILE_ARCH_V2B,
    TACTILE_NAMES,
    TACTILE_STATE_SCHEMA,
    state_contract,
    tactile_from_observation,
    validate_checkpoint_state_contracts,
)
from dexgen.contact_risk_runtime import (
    load_contact_risk_runtime,
    sample_motion_candidates,
    score_motion_candidates,
)
from dexgen.models import ConditionalMotionDiffusion, InverseDynamicsModel
from dexgen.risk_models import CandidateContactRiskHead, TactileProprioTConv
from dexgen.xhand_keypoints import (
    ACTUATED_JOINT_NAMES,
    exact_keypoint_motion,
    representation_metadata,
    xhand_keypoints_fk,
)
from dexgen.modes import mode_metadata
from dexgen.training_utils import (
    atomic_torch_save,
    default_latest_path,
    validate_resume_contract,
)


def _metadata() -> dict:
    return {
        "control_hz": 20.0,
        "dexgen_control_hz": 10.0,
        **representation_metadata(),
    }


def _episode(length: int, termination: str = "timeout") -> dict:
    keypoints = torch.randn(length, 10, 3) * 0.02
    joint_pos = torch.randn(length, 21) * 0.1
    joint_target = joint_pos + torch.randn(length, 21) * 0.01
    return {
        "joint_pos": joint_pos,
        "joint_vel": torch.randn(length, 21) * 0.1,
        "joint_target": joint_target,
        "next_joint_target": joint_target + torch.randn(length, 21) * 0.01,
        "action": torch.randn(length, 21).clamp(-1, 1),
        "keypoints": keypoints,
        "tactile": torch.rand(length, 5),
        "object_pos": torch.randn(length, 3),
        "object_quat": torch.randn(length, 4),
        "object_linvel": torch.randn(length, 3),
        "object_angvel": torch.randn(length, 3),
        "reward": torch.randn(length),
        "termination": termination,
    }


def test_dataset_and_models_round_trip(tmp_path):
    rollout = tmp_path / "rollout.pt"
    torch.save(
        {
            "format": "xhand_dexgen_rollout_v2",
            "metadata": _metadata(),
            "episodes": [_episode(60), _episode(60)],
        },
        rollout,
    )
    raw_train = DexGenDataset(rollout, split="train", drop_trim_seconds=0.0)
    normalizer = raw_train.fit_normalizer(max_samples=20)
    dataset = DexGenDataset(
        rollout, split="train", drop_trim_seconds=0.0, normalizer=normalizer
    )
    sample = dataset[0]
    assert sample["state"].shape == (4, 78)
    assert sample["motion"].shape == (2, 10, 3)
    assert sample["target"].shape == (21,)
    assert sample["current_target"].shape == (21,)

    diffusion = ConditionalMotionDiffusion(
        state_dim=78, history_steps=4, keypoint_count=10, prediction_steps=2,
        hidden_dim=64, state_hidden_dim=64, state_layers=2, num_blocks=2,
        diffusion_steps=10,
    )
    state = sample["state"].unsqueeze(0).repeat(3, 1, 1)
    motion = sample["motion"].unsqueeze(0).repeat(3, 1, 1, 1)
    loss = diffusion.training_loss(motion, state)
    assert loss.ndim == 0 and torch.isfinite(loss)
    generated = diffusion.sample(state, ddim_steps=4)
    assert generated.shape == (3, 2, 10, 3)
    assert torch.isfinite(generated).all()
    command = torch.zeros(3, 10, 3)
    guided = diffusion.guided_sample(
        state,
        command_motion=command,
        guidance_strength=10.0,
        motion_mean=normalizer.motion_mean,
        motion_std=normalizer.motion_std,
        ddim_steps=4,
    )
    assert guided.shape == (3, 2, 10, 3)
    assert torch.isfinite(guided).all()

    idm = InverseDynamicsModel(
        state_dim=78, history_steps=4, keypoint_count=10,
        prediction_steps=2, action_dim=21, hidden_dim=64, num_blocks=2,
    )
    action = idm.act(state, generated)
    assert action.shape == (3, 21)
    assert torch.isfinite(action).all()


def test_tactile_dataset_and_models_round_trip(tmp_path):
    rollout = tmp_path / "tactile_rollout.pt"
    episode = _episode(60)
    episode["tactile"] = torch.arange(60 * 5, dtype=torch.float32).reshape(60, 5)
    torch.save(
        {
            "format": "xhand_dexgen_rollout_v2",
            "metadata": _metadata(),
            "episodes": [episode, _episode(60)],
        },
        rollout,
    )
    dataset = DexGenDataset(
        rollout, split="all", drop_trim_seconds=0.0, include_tactile=True
    )
    sample = dataset[0]
    assert sample["state"].shape == (4, 83)
    assert torch.equal(sample["state"][:, -5:], episode["tactile"][[0, 2, 4, 6]])

    state = sample["state"].unsqueeze(0).repeat(2, 1, 1)
    motion = sample["motion"].unsqueeze(0).repeat(2, 1, 1, 1)
    diffusion = ConditionalMotionDiffusion(
        state_dim=83, history_steps=4, keypoint_count=10, prediction_steps=2,
        hidden_dim=64, state_hidden_dim=64, state_layers=2, num_blocks=2,
        diffusion_steps=10,
    )
    generated = diffusion.sample(state, ddim_steps=4)
    assert generated.shape == motion.shape and torch.isfinite(generated).all()
    idm = InverseDynamicsModel(
        state_dim=83, history_steps=4, keypoint_count=10,
        prediction_steps=2, action_dim=21, hidden_dim=64, num_blocks=2,
    )
    target = idm.act(state, generated)
    assert target.shape == (2, 21) and torch.isfinite(target).all()


def test_tactile_v2b_targets_recovery_weights_and_models(tmp_path):
    rollout = tmp_path / "tactile_v2b.pt"
    episode = _episode(60)
    episode["tactile"] = torch.zeros(60, 5)
    # First dataset sample is current t=6 with history [0,2,4,6].  Finger 0
    # loses force at t=6 and regains it within the 0.2 s recovery horizon.
    episode["tactile"][4, :4] = 1.0
    episode["tactile"][7:12, :4] = 1.0
    torch.save(
        {
            "format": "xhand_dexgen_rollout_v2",
            "metadata": _metadata(),
            "episodes": [episode],
        },
        rollout,
    )
    raw = DexGenDataset(
        rollout, split="all", drop_trim_seconds=0.0, include_tactile=True
    )
    weights, recoveries = raw.recovery_sample_weights(
        boost=4.0,
        contact_threshold=0.1,
        low_contact_count=2,
        stable_contact_count=4,
        horizon_seconds=0.2,
    )
    assert bool(recoveries[0])
    assert weights[0] > weights[~recoveries].mean()
    normalizer = raw.fit_normalizer(max_samples=20)
    dataset = DexGenDataset(
        rollout,
        split="all",
        drop_trim_seconds=0.0,
        include_tactile=True,
        normalizer=normalizer,
    )
    sample = dataset[0]
    expected_future = normalizer.normalize_tactile(episode["tactile"][[8, 10]])
    assert torch.allclose(sample["future_tactile"], expected_future)

    state = sample["state"].unsqueeze(0).repeat(2, 1, 1)
    motion = sample["motion"].unsqueeze(0).repeat(2, 1, 1, 1)
    future_tactile = sample["future_tactile"].unsqueeze(0).repeat(2, 1, 1)
    threshold = normalizer.normalized_tactile_threshold(0.1).tolist()
    diffusion = ConditionalMotionDiffusion(
        state_dim=83,
        history_steps=4,
        keypoint_count=10,
        prediction_steps=2,
        hidden_dim=64,
        state_hidden_dim=64,
        state_layers=2,
        num_blocks=2,
        diffusion_steps=10,
        tactile_arch=TACTILE_ARCH_V2B,
        tactile_per_finger_dim=8,
        tactile_embedding_dim=16,
        tactile_contact_threshold_normalized=threshold,
    )
    motion_loss, tactile_loss = diffusion.training_loss_components(
        motion, state, future_tactile=future_tactile
    )
    assert motion_loss.shape == (2,) and tactile_loss is not None
    assert tactile_loss.shape == (2,) and torch.isfinite(tactile_loss).all()
    reloaded_diffusion = ConditionalMotionDiffusion(**diffusion.config())
    reloaded_diffusion.load_state_dict(diffusion.state_dict())

    idm = InverseDynamicsModel(
        state_dim=83,
        history_steps=4,
        keypoint_count=10,
        prediction_steps=2,
        action_dim=21,
        hidden_dim=64,
        num_blocks=2,
        tactile_arch=TACTILE_ARCH_V2B,
        tactile_per_finger_dim=8,
        tactile_embedding_dim=16,
        tactile_contact_threshold_normalized=threshold,
    )
    target = sample["target"].unsqueeze(0).repeat(2, 1)
    target_loss, idm_tactile_loss = idm.loss_components(
        state, motion, target, future_tactile
    )
    assert target_loss.shape == (2,) and idm_tactile_loss is not None
    assert torch.isfinite(target_loss).all() and torch.isfinite(idm_tactile_loss).all()
    reloaded_idm = InverseDynamicsModel(**idm.config())
    reloaded_idm.load_state_dict(idm.state_dict())


def test_tactile_film_v3_round_trip_and_learns_independent_touch_path():
    torch.manual_seed(19)
    model = ConditionalMotionDiffusion(
        state_dim=83,
        history_steps=4,
        keypoint_count=10,
        prediction_steps=2,
        hidden_dim=32,
        state_hidden_dim=32,
        state_layers=2,
        num_blocks=2,
        time_dim=16,
        diffusion_steps=10,
        tactile_arch=TACTILE_ARCH_FILM,
        tactile_per_finger_dim=8,
        tactile_embedding_dim=16,
        tactile_condition_dropout=0.1,
    )
    state = torch.randn(3, 4, 83)
    motion = torch.randn(3, 2, 10, 3)
    future_tactile = torch.randn(3, 2, 5)
    timesteps = torch.tensor([1, 4, 8])
    noise = torch.randn_like(motion)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    losses, tactile_losses = model.training_loss_components(
        motion,
        state,
        future_tactile=future_tactile,
        timesteps=timesteps,
        noise=noise,
    )
    assert tactile_losses is not None and torch.isfinite(tactile_losses).all()
    (losses.mean() + 0.005 * tactile_losses.mean()).backward()
    first_block = model.encoder[0]
    assert first_block.tactile_condition.weight.grad is not None
    assert bool((first_block.tactile_condition.weight.grad != 0).any())
    optimizer.step()

    model.eval()
    noisy = torch.randn_like(motion)
    altered_touch = state.clone()
    altered_touch[..., -5:] += 3.0
    original_output = model(noisy, timesteps, state)
    altered_output = model(noisy, timesteps, altered_touch)
    assert not torch.equal(original_output, altered_output)

    reloaded = ConditionalMotionDiffusion(**model.config())
    reloaded.load_state_dict(model.state_dict())
    reloaded.eval()
    assert reloaded.config()["tactile_condition_dropout"] == 0.1
    assert torch.allclose(reloaded(noisy, timesteps, state), original_output)


def test_explicit_recovery_labels_and_perturbation_supervision_mask(tmp_path):
    rollout = tmp_path / "explicit_recovery.pt"
    episode = _episode(60)
    episode["supervision_valid"] = torch.ones(60, dtype=torch.bool)
    episode["supervision_valid"][8:11] = False
    episode["perturbation_phase"] = torch.zeros(60, dtype=torch.long)
    episode["perturbation_phase"][8:11] = 1
    episode["perturbation_phase"][11:31] = 2
    episode["perturbation_id"] = torch.full((60,), -1, dtype=torch.long)
    episode["perturbation_id"][8:31] = 7
    episode["perturbation_type_id"] = torch.zeros(60, dtype=torch.long)
    episode["perturbation_type_id"][8:31] = 2
    episode["perturbed_finger_id"] = torch.full((60,), -1, dtype=torch.long)
    episode["perturbed_finger_id"][8:31] = 1
    episode["recovery_success"] = torch.zeros(60, dtype=torch.bool)
    episode["recovery_success"][8:31] = True
    episode["rotation_axis"] = torch.tensor([0.0, 1.0, 0.0]).repeat(60, 1)
    episode["object_angvel"] = torch.tensor([0.0, 0.4, 0.0]).repeat(60, 1)
    torch.save(
        {
            "format": "xhand_dexgen_rollout_v2",
            "metadata": _metadata(),
            "episodes": [episode],
        },
        rollout,
    )
    dataset = DexGenDataset(
        rollout, split="all", drop_trim_seconds=0.0, include_tactile=True
    )
    sample_steps = [index for _, index in dataset.samples]
    assert 6 not in sample_steps and 8 not in sample_steps and 10 not in sample_steps
    recovery_index = next(i for i, (_, step) in enumerate(dataset.samples) if step == 12)
    sample = dataset._raw_item(recovery_index)
    assert sample["recovery_label"].item() == 1.0
    assert sample["perturbation_type"].item() == 2
    assert sample["perturbed_finger"].item() == 1
    assert torch.allclose(sample["future_signed_rotation"], torch.tensor([0.4, 0.4]))
    weights, labels = dataset.recovery_sample_weights(boost=4.0)
    assert bool(labels[recovery_index])
    assert weights[recovery_index] > weights[~labels].mean()
    fraction_weights, fraction_labels = dataset.recovery_sample_weights(
        target_fraction=0.25
    )
    assert torch.equal(labels, fraction_labels)
    assert torch.isclose(fraction_weights[fraction_labels].sum(), torch.tensor(0.25, dtype=torch.double))
    assert torch.isclose(fraction_weights[~fraction_labels].sum(), torch.tensor(0.75, dtype=torch.double))

    regular_index = next(i for i, (_, step) in enumerate(dataset.samples) if step == 40)
    regular_sample = dataset._raw_item(regular_index)
    assert regular_sample["recovery_label"].item() == 0.0
    assert regular_sample["perturbation_type"].item() == 0

    old_rollout = tmp_path / "steady_old.pt"
    torch.save(
        {
            "format": "xhand_dexgen_rollout_v2",
            "metadata": _metadata(),
            "episodes": [_episode(60)],
        },
        old_rollout,
    )
    mixed = DexGenDataset(
        [old_rollout, rollout], split="all", drop_trim_seconds=0.0,
        include_tactile=True,
    )
    batch = next(iter(DataLoader(mixed, batch_size=len(mixed))))
    assert batch["recovery_label"].shape == (len(mixed),)
    assert bool((batch["recovery_label"] == 0).any())
    assert bool((batch["recovery_label"] == 1).any())


def test_recovery_type_balancing_and_release_metadata(tmp_path):
    rollout = tmp_path / "typed_recovery.pt"
    episodes = []
    for type_id in range(1, 5):
        episode = _episode(60)
        episode["mode_id"] = 0
        episode["supervision_valid"] = torch.ones(60, dtype=torch.bool)
        episode["perturbation_phase"] = torch.zeros(60, dtype=torch.long)
        episode["perturbation_phase"][10:31] = 2
        episode["perturbation_id"] = torch.full((60,), -1, dtype=torch.long)
        episode["perturbation_id"][10:31] = type_id
        episode["perturbation_type_id"] = torch.zeros(60, dtype=torch.long)
        episode["perturbation_type_id"][10:31] = type_id
        episode["perturbation_subtype_id"] = torch.zeros(60, dtype=torch.long)
        episode["perturbation_subtype_id"][10:31] = 2 if type_id == 2 else 0
        episode["perturbed_finger_id"] = torch.full((60,), -1, dtype=torch.long)
        episode["perturbed_finger_mask"] = torch.zeros(60, 5, dtype=torch.bool)
        if type_id == 2:
            episode["perturbed_finger_id"][10:31] = 1
            episode["perturbed_finger_mask"][10:31, 1:3] = True
        episode["recovery_success"] = torch.zeros(60, dtype=torch.bool)
        episode["recovery_success"][10:31] = True
        episodes.append(episode)
    torch.save(
        {
            "format": "xhand_dexgen_rollout_v3",
            "metadata": _metadata(),
            "episodes": episodes,
        },
        rollout,
    )
    dataset = DexGenDataset(
        rollout, split="all", drop_trim_seconds=0.0, include_tactile=True
    )
    weights, labels = dataset.recovery_sample_weights(
        target_fraction=0.35,
        type_weights=(0.1, 0.5, 0.1, 0.3),
        explicit_only=True,
    )
    assert torch.isclose(weights[labels].sum(), torch.tensor(0.35, dtype=torch.double))
    for type_id, expected in enumerate((0.035, 0.175, 0.035, 0.105), start=1):
        selected = torch.tensor(
            [
                bool(labels[index])
                and int(dataset._raw_item(index)["perturbation_type"]) == type_id
                for index in range(len(dataset))
            ]
        )
        assert torch.isclose(
            weights[selected].sum(), torch.tensor(expected, dtype=torch.double)
        )
    release_index = next(
        index
        for index in range(len(dataset))
        if int(dataset._raw_item(index)["perturbation_type"]) == 2
        and bool(dataset._raw_item(index)["recovery_label"])
    )
    release = dataset._raw_item(release_index)
    assert int(release["perturbation_subtype"]) == 2
    assert torch.equal(
        release["perturbed_finger_mask"],
        torch.tensor([False, True, True, False, False]),
    )


def test_tactile_state_contract_and_live_extraction():
    assert state_contract(False)["state_schema"] == PROPRIO_STATE_SCHEMA
    tactile_contract = state_contract(True)
    assert tactile_contract["state_schema"] == TACTILE_STATE_SCHEMA
    assert tactile_contract["state_dim"] == 83
    assert tactile_contract["tactile_names"] == list(TACTILE_NAMES)

    history = torch.arange(2 * 3 * 47, dtype=torch.float32).reshape(2, 3, 47)
    assert torch.equal(
        tactile_from_observation({"proprio_hist": history}), history[:, -1, -5:]
    )

    def checkpoint(include_tactile: bool) -> dict:
        contract = state_contract(include_tactile)
        dimension = int(contract["state_dim"])
        return {
            "model_config": {"state_dim": dimension},
            "dataset_config": contract,
            "normalizer": {
                "state_mean": torch.zeros(dimension),
                "state_std": torch.ones(dimension),
            },
        }

    assert validate_checkpoint_state_contracts(
        checkpoint(True), checkpoint(True)
    )["include_tactile"] is True
    try:
        validate_checkpoint_state_contracts(checkpoint(False), checkpoint(True))
    except ValueError as error:
        assert "different state contracts" in str(error)
    else:
        raise AssertionError("Mixed proprio/tactile checkpoints were accepted")

    direct = checkpoint(True)
    v2b = checkpoint(True)
    film = checkpoint(True)
    v2b["dataset_config"] = state_contract(True, TACTILE_ARCH_V2B)
    v2b["model_config"]["tactile_arch"] = TACTILE_ARCH_V2B
    film["dataset_config"] = state_contract(True, TACTILE_ARCH_FILM)
    film["model_config"]["tactile_arch"] = TACTILE_ARCH_FILM
    # Internal encoders may differ: film_v3 Diffusion is paired with v2b IDM.
    assert validate_checkpoint_state_contracts(film, v2b)["tactile_arch"] == TACTILE_ARCH_FILM
    assert validate_checkpoint_state_contracts(direct, v2b)["include_tactile"] is True
    assert direct["dataset_config"]["tactile_arch"] == TACTILE_ARCH_DIRECT


def test_drop_tail_is_removed(tmp_path):
    rollout = tmp_path / "drop.pt"
    torch.save(
        {
            "format": "xhand_dexgen_rollout_v2",
            "metadata": _metadata(),
            "episodes": [_episode(100, "drop"), _episode(100)],
        }, rollout,
    )
    all_data = DexGenDataset(
        rollout, split="all", drop_trim_seconds=2.0,
        history_steps=4, prediction_steps=2, future_stride=2,
    )
    per_episode = {0: 0, 1: 0}
    for episode_id, _ in all_data.samples:
        per_episode[episode_id] += 1
    assert per_episode[0] == per_episode[1] - 20


def test_dataset_time_alignment(tmp_path):
    episode = _episode(30)
    time = torch.arange(30, dtype=torch.float32)
    episode["keypoints"] = time[:, None, None].expand(-1, 10, 3).clone()
    episode["joint_pos"] = time[:, None].expand(-1, 21).clone()
    episode["joint_target"] = (100.0 + time)[:, None].expand(-1, 21).clone()
    episode["next_joint_target"] = (200.0 + time)[:, None].expand(-1, 21).clone()
    rollout = tmp_path / "aligned.pt"
    torch.save(
        {"format": "xhand_dexgen_rollout_v2", "metadata": _metadata(), "episodes": [episode]},
        rollout,
    )
    dataset = DexGenDataset(rollout, split="all", drop_trim_seconds=0.0)
    sample = dataset[0]  # current t=6; history=[0,2,4,6], future=[8,10]
    assert sample["step"].item() == 6
    assert torch.equal(sample["state"][:, 15], torch.tensor([0.0, 2.0, 4.0, 6.0]))
    assert torch.equal(sample["motion"][:, 0, 0], torch.tensor([2.0, 4.0]))
    assert sample["target"][0].item() == 207.0


def test_representation_metadata_is_enforced(tmp_path):
    metadata = _metadata()
    metadata["keypoint_names"] = list(reversed(metadata["keypoint_names"]))
    rollout = tmp_path / "bad_metadata.pt"
    torch.save(
        {"format": "xhand_dexgen_rollout_v2", "metadata": metadata, "episodes": [_episode(30)]},
        rollout,
    )
    try:
        DexGenDataset(rollout, split="all", drop_trim_seconds=0.0)
    except ValueError as error:
        assert "keypoint_names" in str(error)
    else:
        raise AssertionError("Reordered keypoint metadata was accepted")


def test_exact_fk_and_joint_name_reordering():
    torch.manual_seed(7)
    current = torch.randn(3, 21) * 0.1
    target = current + torch.randn(3, 21) * 0.02
    canonical = xhand_keypoints_fk(current)
    reverse_names = list(reversed(ACTUATED_JOINT_NAMES))
    reordered = xhand_keypoints_fk(current.flip(1), reverse_names)
    assert torch.allclose(canonical, reordered, atol=1.0e-7)
    expected = xhand_keypoints_fk(target) - canonical
    actual = exact_keypoint_motion(current, target, max_motion_m=None)
    assert torch.allclose(actual, expected, atol=1.0e-7)
    assert exact_keypoint_motion(current, current).abs().max() == 0.0
    clipped = exact_keypoint_motion(current, target, max_motion_m=0.001)
    assert bool((clipped.norm(dim=-1) <= 0.001001).all())


def test_guidance_reproducibility_and_broadcast():
    torch.manual_seed(11)
    model = ConditionalMotionDiffusion(
        state_dim=6, history_steps=4, keypoint_count=2, prediction_steps=2,
        hidden_dim=32, state_hidden_dim=32, state_layers=2, num_blocks=2,
        diffusion_steps=10,
    )
    state = torch.randn(2, 4, 6)
    noise = torch.randn(2, 2, 2, 3)
    plain = model.sample(state, ddim_steps=4, initial_noise=noise)
    zero_guidance = model.guided_sample(
        state,
        command_motion=torch.zeros(2, 2, 3),
        guidance_strength=0.0,
        motion_mean=torch.zeros(2, 2, 3),
        motion_std=torch.ones(2, 2, 3),
        ddim_steps=4,
        initial_noise=noise,
    )
    assert torch.equal(plain, zero_guidance)
    broadcast = model.guided_sample(
        state,
        command_motion=torch.full((2, 2, 3), 0.01),
        guidance_strength=1.0,
        motion_mean=torch.zeros(2, 2, 3),
        motion_std=torch.ones(2, 2, 3),
        ddim_steps=4,
        initial_noise=noise,
    )
    expanded = model.guided_sample(
        state,
        command_motion=torch.full((2, 2, 2, 3), 0.01),
        guidance_strength=1.0,
        motion_mean=torch.zeros(2, 2, 3),
        motion_std=torch.ones(2, 2, 3),
        ddim_steps=4,
        initial_noise=noise,
    )
    assert torch.equal(broadcast, expanded)


def test_contact_risk_candidate_zero_preserves_v3_rng_stream():
    torch.manual_seed(19)
    model = ConditionalMotionDiffusion(
        state_dim=6, history_steps=4, keypoint_count=2, prediction_steps=2,
        hidden_dim=32, state_hidden_dim=32, state_layers=2, num_blocks=2,
        diffusion_steps=10, num_modes=2,
    )
    state = torch.randn(3, 4, 6)
    mode = torch.tensor([0, 1, 0])
    baseline_generator = torch.Generator().manual_seed(2026)
    expected = model.sample(
        state, ddim_steps=4, mode=mode, generator=baseline_generator
    )

    rerank_baseline_generator = torch.Generator().manual_seed(2026)
    extra_generator = torch.Generator().manual_seed(90031)
    candidates = sample_motion_candidates(
        model,
        state,
        mode,
        candidate_count=1,
        ddim_steps=4,
        baseline_generator=rerank_baseline_generator,
        extra_generator=extra_generator,
    )
    assert torch.equal(candidates[:, 0], expected)
    assert torch.equal(
        torch.randn(8, generator=baseline_generator),
        torch.randn(8, generator=rerank_baseline_generator),
    )


def test_contact_risk_runtime_load_and_score(tmp_path):
    torch.manual_seed(23)
    normalizer = FeatureNormalizer(
        state_mean=torch.randn(83),
        state_std=torch.rand(83) + 0.1,
        motion_mean=torch.randn(2, 10, 3),
        motion_std=torch.rand(2, 10, 3) + 0.1,
        action_mean=torch.randn(21),
        action_std=torch.rand(21) + 0.1,
    )
    student = TactileProprioTConv(
        state_dim=83, history_steps=30, latent_dim=8, hidden_dim=32
    )
    head = CandidateContactRiskHead(
        state_dim=83,
        policy_history_steps=4,
        keypoint_count=10,
        prediction_steps=2,
        latent_dim=8,
        num_modes=2,
        hidden_dim=64,
    )
    checkpoint = tmp_path / "risk.pt"
    torch.save(
        {
            "format": "xhand_dexgen_contact_risk_student_v9",
            "dataset_config": {
                "recovery_label_mode": "local",
                "adapter_hz": 20.0,
                "generator_hz": 10.0,
                "disable_tactile_ids": [],
            },
            "normalizer": normalizer.state_dict(),
            "student": student.state_dict(),
            "student_config": student.config(),
            "risk_head": head.state_dict(),
            "risk_head_config": head.config(),
        },
        checkpoint,
    )
    runtime = load_contact_risk_runtime(
        checkpoint,
        torch.device("cpu"),
        base_normalizer=normalizer,
        state_dim=83,
        history_steps=4,
        prediction_steps=2,
        keypoint_count=10,
        num_modes=2,
    )
    logits = score_motion_candidates(
        runtime,
        adapter_history=torch.randn(3, 30, 83),
        policy_history=torch.randn(3, 4, 83),
        motion_candidates=torch.randn(3, 5, 2, 10, 3),
        mode=torch.tensor([0, 1, 0]),
    )
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()


def test_epsilon_prediction_training_and_sampling():
    torch.manual_seed(13)
    model = ConditionalMotionDiffusion(
        state_dim=6, history_steps=4, keypoint_count=2, prediction_steps=2,
        hidden_dim=32, state_hidden_dim=32, state_layers=2, num_blocks=2,
        diffusion_steps=10, prediction_type="epsilon",
    )
    state = torch.randn(3, 4, 6)
    motion = torch.randn(3, 2, 2, 3)
    timesteps = torch.tensor([0, 4, 9])
    noise = torch.randn_like(motion)
    losses = model.training_loss(
        motion, state, timesteps=timesteps, noise=noise, reduction="none"
    )
    assert losses.shape == (3,) and torch.isfinite(losses).all()
    generated = model.sample(state, ddim_steps=4, initial_noise=noise)
    assert generated.shape == motion.shape and torch.isfinite(generated).all()
    assert model.config()["prediction_type"] == "epsilon"


def test_cosine_schedule_reaches_near_zero_terminal_snr():
    model = ConditionalMotionDiffusion(
        state_dim=4, history_steps=2, keypoint_count=2,
        hidden_dim=16, state_hidden_dim=16, state_layers=1, num_blocks=1,
        time_dim=8, diffusion_steps=100, num_modes=2,
        prediction_type="epsilon", noise_schedule="cosine",
    )
    assert model.config()["noise_schedule"] == "cosine"
    assert float(model.alphas_cumprod[-1]) < 1.0e-5


def test_stratified_split_is_seeded_and_not_tail_only(tmp_path):
    episodes = []
    for mode_id in (0, 1):
        for _ in range(20):
            episode = _episode(30)
            episode["mode_id"] = mode_id
            episodes.append(episode)
    metadata = {
        **_metadata(), **mode_metadata(), "sensor_profile": "raw",
    }
    path = tmp_path / "split.pt"
    torch.save(
        {"format": "xhand_dexgen_rollout_v3", "metadata": metadata, "episodes": episodes},
        path,
    )
    val_a = DexGenDataset(path, split="val", validation_fraction=0.2, split_seed=123)
    val_b = DexGenDataset(path, split="val", validation_fraction=0.2, split_seed=123)
    val_c = DexGenDataset(path, split="val", validation_fraction=0.2, split_seed=124)
    assert val_a.episode_ids == val_b.episode_ids
    assert val_a.episode_ids != val_c.episode_ids
    assert len(val_a.episode_ids) == 8
    assert {int(val_a.payload["episodes"][i]["mode_id"]) for i in val_a.episode_ids} == {0, 1}
    assert val_a.episode_ids != list(range(16, 20)) + list(range(36, 40))


def test_atomic_checkpoint_helpers(tmp_path):
    best = tmp_path / "model_best.pt"
    latest = default_latest_path(best)
    assert latest.name == "model_latest.pt"
    atomic_torch_save({"value": torch.tensor([1, 2, 3])}, latest)
    assert torch.equal(torch.load(latest, weights_only=False)["value"], torch.tensor([1, 2, 3]))


def test_legacy_resume_accepts_only_proprio_state_contract():
    legacy = {
        "model_config": {"state_dim": 78},
        "dataset_config": {"history_steps": 4},
        "source_data": ["rollout.pt"],
    }
    proprio = {
        "history_steps": 4,
        **state_contract(False),
    }
    validate_resume_contract(legacy, proprio, ["rollout.pt"])
    try:
        validate_resume_contract(
            legacy,
            {"history_steps": 4, **state_contract(True)},
            ["rollout.pt"],
        )
    except ValueError as error:
        assert "state_schema" in str(error)
    else:
        raise AssertionError("Legacy 78-D checkpoint was accepted as tactile")


def test_two_mode_rollout_collection(tmp_path):
    paths = []
    for mode_id, name in enumerate(("cw", "ccw")):
        metadata = {
            **_metadata(),
            **mode_metadata(),
            "mode_id": mode_id,
            "mode_name": name,
            "rotation_axis_sign": 1 if mode_id == 0 else -1,
            "sensor_profile": "raw",
        }
        episodes = []
        for _ in range(3):
            episode = _episode(40)
            episode["mode_id"] = mode_id
            episodes.append(episode)
        path = tmp_path / f"{name}.pt"
        torch.save(
            {"format": "xhand_dexgen_rollout_v3", "metadata": metadata, "episodes": episodes},
            path,
        )
        paths.append(path)

    train = DexGenDataset(paths, split="train", drop_trim_seconds=0.0)
    val = DexGenDataset(paths, split="val", drop_trim_seconds=0.0)
    assert train.mode_ids == (0, 1)
    assert train.num_modes == 2
    assert val.mode_ids == (0, 1)
    assert {int(train[index]["mode"]) for index in range(len(train))} == {0, 1}
    weights = train.balanced_sample_weights()
    per_mode_weight = {mode_id: 0.0 for mode_id in train.mode_ids}
    for weight, (episode_id, _) in zip(weights, train.samples):
        mode_id = int(train.payload["episodes"][episode_id]["mode_id"])
        per_mode_weight[mode_id] += float(weight)
    assert abs(per_mode_weight[0] - per_mode_weight[1]) < 1.0e-9
