"""Audit active-perturbation DexGen rollouts before recovery training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexgen.data import DexGenDataset, load_rollout_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--drop_trim_seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_rollout_collection(args.data)
    episodes = payload["episodes"]
    drops = sum(episode.get("termination") == "drop" for episode in episodes)
    transitions = sum(int(episode["action"].shape[0]) for episode in episodes)
    event_steps: dict[tuple[int, int], dict[str, object]] = {}
    for episode_index, episode in enumerate(episodes):
        if "perturbation_id" not in episode:
            continue
        event_ids = episode["perturbation_id"].long()
        for event_id in event_ids.unique().tolist():
            if event_id < 0:
                continue
            selected = event_ids == event_id
            type_id = int(episode["perturbation_type_id"][selected][0])
            subtype_id = (
                int(episode["perturbation_subtype_id"][selected][0])
                if "perturbation_subtype_id" in episode else 0
            )
            success = bool(episode["recovery_success"][selected].any())
            event_steps[(episode_index, event_id)] = {
                "type": type_id,
                "subtype": subtype_id,
                "success": success,
                "steps": int(selected.sum()),
            }

    by_type: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    by_subtype: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for event in event_steps.values():
        type_id = int(event["type"])
        by_type[type_id][0] += 1
        by_type[type_id][1] += int(bool(event["success"]))
        if type_id == 2:
            subtype_id = int(event["subtype"])
            by_subtype[subtype_id][0] += 1
            by_subtype[subtype_id][1] += int(bool(event["success"]))

    print(
        f"episodes={len(episodes)} drops={drops} ({100.0 * drops / max(len(episodes), 1):.2f}%) "
        f"transitions={transitions} events={len(event_steps)}"
    )
    type_names = payload["metadata"].get("recovery_perturbations", {}).get("types", {})
    for type_id, (count, success) in sorted(by_type.items()):
        name = type_names.get(type_id, type_names.get(str(type_id), str(type_id)))
        print(
            f"type={name} events={count} success={success} "
            f"rate={100.0 * success / max(count, 1):.2f}%"
        )
    subtype_names = payload["metadata"].get("recovery_perturbations", {}).get(
        "finger_release_subtypes", {}
    )
    for subtype_id, (count, success) in sorted(by_subtype.items()):
        name = subtype_names.get(
            subtype_id, subtype_names.get(str(subtype_id), str(subtype_id))
        )
        print(
            f"release_subtype={name} events={count} success={success} "
            f"rate={100.0 * success / max(count, 1):.2f}%"
        )

    try:
        dataset = DexGenDataset(
            args.data,
            split="all",
            include_tactile=True,
            drop_trim_seconds=args.drop_trim_seconds,
        )
    except ValueError as error:
        print(f"training_windows=0 reason={error}")
        return
    _, labels = dataset.recovery_sample_weights(boost=4.0, explicit_only=True)
    rotation = []
    for sample_index in labels.nonzero(as_tuple=False).flatten().tolist():
        item = dataset._raw_item(sample_index)
        if "future_signed_rotation" in item:
            rotation.append(float(item["future_signed_rotation"].mean()))
    rotation_text = (
        f" mean_recovery_rotation={sum(rotation) / len(rotation):+.4f} rad/s"
        if rotation else ""
    )
    print(
        f"training_windows={len(dataset)} explicit_recovery_windows={int(labels.sum())}"
        f"{rotation_text}"
    )


if __name__ == "__main__":
    main()
