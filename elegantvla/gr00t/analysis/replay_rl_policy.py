#!/usr/bin/env python3

"""Replay saved schedule observations through one or more PPO checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
import glob
import itertools
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO


OBSERVATION_KEYS = (
    "eef_rotation_speed",
    "eef_translation_speed",
    "feature_similarity",
    "gripper_speed",
    "step_progress",
)
BACKBONE_NAMES = (
    "REFRESH",
    "REUSE",
    "REUSE_THEN_NO_IMAGE_1",
    "REUSE_THEN_NO_IMAGE_2",
    "REUSE_THEN_NO_IMAGE_3",
)
ACTION_HEAD_NAMES = (
    "REFRESH",
    "REUSE_MIDDLE",
    "REUSE_ALL_BUT_FIRST",
)


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=/path/to/model.zip")
    label, raw_path = value.split("=", maxsplit=1)
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("checkpoint label cannot be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return label, path


def _checkpoint_sort_key(item: tuple[str, Path]) -> tuple[int, str]:
    match = re.search(r"_(\d+)_episodes$", item[1].stem)
    return (int(match.group(1)), item[0]) if match else (10**12, item[0])


def _resolve_checkpoints(
    explicit: list[tuple[str, Path]],
    patterns: Iterable[str],
) -> list[tuple[str, Path]]:
    checkpoints = list(explicit)
    for pattern in patterns:
        for raw_path in glob.glob(str(Path(pattern).expanduser()), recursive=True):
            path = Path(raw_path).resolve()
            if not path.is_file():
                continue
            match = re.search(r"_(\d+)_episodes$", path.stem)
            label = f"ep{match.group(1)}" if match else path.stem
            checkpoints.append((label, path))
    checkpoints.sort(key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError("no PPO checkpoints were supplied or matched")
    labels = [label for label, _ in checkpoints]
    if len(set(labels)) != len(labels):
        raise ValueError("checkpoint labels must be unique")
    return checkpoints


def _resolve_trace_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(str(Path(pattern).expanduser()), recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        paths.update(Path(match).resolve() for match in matches if Path(match).is_file())
    if not paths:
        raise FileNotFoundError("no decision traces matched the supplied patterns")
    return sorted(paths)


def _vecnormalize_path(checkpoint_path: Path) -> Path:
    stem = checkpoint_path.stem if checkpoint_path.suffix == ".zip" else checkpoint_path.name
    candidates = [checkpoint_path.with_name(f"{stem}_vecnormalize.pkl")]
    if stem.startswith("ppo_schedule_"):
        suffix = stem[len("ppo_schedule_") :]
        candidates.append(checkpoint_path.with_name(f"ppo_schedule_vecnormalize_{suffix}.pkl"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"no VecNormalize state found; checked: {rendered}")


def _load_vecnormalize(checkpoint_path: Path) -> tuple[object, Path]:
    normalizer_path = _vecnormalize_path(checkpoint_path)
    with normalizer_path.open("rb") as handle:
        normalizer = pickle.load(handle)
    setattr(normalizer, "training", False)
    setattr(normalizer, "norm_reward", False)
    return normalizer, normalizer_path


def _load_traces(
    trace_paths: list[Path],
    *,
    include_latency_warmup: bool,
) -> dict[str, Any]:
    observations: list[list[float]] = []
    masks: list[list[bool]] = []
    teacher_backbone: list[int] = []
    teacher_action_head: list[int] = []
    task_ids: list[int] = []
    episode_ids: list[int] = []
    seeds: list[int] = []
    task_names: list[str] = []
    task_to_id: dict[str, int] = {}
    episode_keys: dict[int, set[tuple[int, int]]] = defaultdict(set)
    excluded_warmup_rows = 0

    for trace_path in trace_paths:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("latency_warmup", False) and not include_latency_warmup:
                    excluded_warmup_rows += 1
                    continue

                task_name = str(row["env_name"])
                if task_name not in task_to_id:
                    task_to_id[task_name] = len(task_names)
                    task_names.append(task_name)
                task_id = task_to_id[task_name]
                mask = np.asarray(row["backbone_action_mask"], dtype=bool).reshape(-1)
                if mask.size != len(BACKBONE_NAMES) or not mask.any():
                    raise ValueError(
                        f"invalid backbone mask at {trace_path}:{line_number}: {mask.tolist()}"
                    )
                backbone_action = int(row["backbone_action"])
                action_head_action = int(row["action_head_action"])
                if not 0 <= backbone_action < len(BACKBONE_NAMES):
                    raise ValueError(f"invalid backbone action at {trace_path}:{line_number}")
                if not mask[backbone_action]:
                    raise ValueError(f"teacher backbone action is masked at {trace_path}:{line_number}")
                if not 0 <= action_head_action < len(ACTION_HEAD_NAMES):
                    raise ValueError(f"invalid action-head action at {trace_path}:{line_number}")

                observation = [float(row[key]) for key in OBSERVATION_KEYS]
                if not np.isfinite(observation).all():
                    raise ValueError(f"non-finite observation at {trace_path}:{line_number}")
                observations.append(observation)
                masks.append(mask.tolist())
                teacher_backbone.append(backbone_action)
                teacher_action_head.append(action_head_action)
                task_ids.append(task_id)
                episode_id = int(row["episode"])
                seed = int(row["seed"])
                episode_ids.append(episode_id)
                seeds.append(seed)
                episode_keys[task_id].add((episode_id, seed))

    if not observations:
        raise ValueError("all trace rows were filtered out")
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "backbone_masks": np.asarray(masks, dtype=bool),
        "teacher_backbone": np.asarray(teacher_backbone, dtype=np.int64),
        "teacher_action_head": np.asarray(teacher_action_head, dtype=np.int64),
        "task_ids": np.asarray(task_ids, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "task_names": task_names,
        "episodes_per_task": {
            task_names[task_id]: len(keys) for task_id, keys in episode_keys.items()
        },
        "excluded_warmup_rows": excluded_warmup_rows,
    }


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -terms.sum(axis=1)


def _action_distribution(values: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    counts = np.bincount(values, minlength=len(names))
    total = int(counts.sum())
    return {
        name: {
            "count": int(count),
            "ratio": 0.0 if total == 0 else float(count / total),
        }
        for name, count in zip(names, counts, strict=True)
    }


def _confusion_matrix(
    expected: np.ndarray,
    predicted: np.ndarray,
    size: int,
) -> list[list[int]]:
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (expected, predicted), 1)
    return matrix.tolist()


def _mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _metrics_for_rows(
    indices: np.ndarray,
    *,
    backbone_masks: np.ndarray,
    teacher_backbone: np.ndarray,
    teacher_action_head: np.ndarray,
    predicted_backbone: np.ndarray,
    predicted_action_head: np.ndarray,
    backbone_probs: np.ndarray,
    action_head_probs: np.ndarray,
) -> dict[str, Any]:
    row_masks = backbone_masks[indices]
    row_teacher_backbone = teacher_backbone[indices]
    row_teacher_action_head = teacher_action_head[indices]
    row_predicted_backbone = predicted_backbone[indices]
    row_predicted_action_head = predicted_action_head[indices]
    row_backbone_probs = backbone_probs[indices]
    row_action_head_probs = action_head_probs[indices]
    free_local = row_masks.sum(axis=1) > 1
    forced_local = ~free_local
    backbone_match = row_predicted_backbone == row_teacher_backbone
    action_head_match = row_predicted_action_head == row_teacher_action_head
    joint_match = backbone_match & action_head_match

    backbone_entropy = _entropy(row_backbone_probs)
    action_head_entropy = _entropy(row_action_head_probs)
    valid_counts = row_masks.sum(axis=1)
    normalized_backbone_entropy = np.zeros_like(backbone_entropy)
    normalized_backbone_entropy[free_local] = (
        backbone_entropy[free_local] / np.log(valid_counts[free_local])
    )
    teacher_backbone_probability = row_backbone_probs[
        np.arange(indices.size), row_teacher_backbone
    ]
    teacher_action_head_probability = row_action_head_probs[
        np.arange(indices.size), row_teacher_action_head
    ]
    eps = np.finfo(np.float32).tiny
    mismatch_penalty = (~backbone_match).astype(np.float32) + 0.5 * (
        ~action_head_match
    ).astype(np.float32)
    sampled_mismatch_penalty = 0.5 * (1.0 - teacher_action_head_probability)
    sampled_mismatch_penalty[free_local] += (
        1.0 - teacher_backbone_probability[free_local]
    )

    result = {
        "rows": int(indices.size),
        "free_backbone_rows": int(free_local.sum()),
        "forced_backbone_rows": int(forced_local.sum()),
        "backbone_match_all": _mean(backbone_match),
        "backbone_match_free": _mean(backbone_match[free_local]),
        "action_head_match": _mean(action_head_match),
        "joint_match_all": _mean(joint_match),
        "joint_match_free_backbone": _mean(joint_match[free_local]),
        "stage1_deterministic_mismatch_penalty_mean": float(mismatch_penalty.mean()),
        "stage1_expected_sampled_mismatch_penalty_mean": float(
            sampled_mismatch_penalty.mean()
        ),
        "backbone_entropy_free": _mean(backbone_entropy[free_local]),
        "backbone_normalized_entropy_free": _mean(
            normalized_backbone_entropy[free_local]
        ),
        "action_head_entropy": float(action_head_entropy.mean()),
        "action_head_normalized_entropy": float(
            action_head_entropy.mean() / np.log(len(ACTION_HEAD_NAMES))
        ),
        "teacher_backbone_probability_free": _mean(
            teacher_backbone_probability[free_local]
        ),
        "teacher_backbone_probability_all": float(
            teacher_backbone_probability.mean()
        ),
        "teacher_action_head_probability": float(
            teacher_action_head_probability.mean()
        ),
        "sampled_joint_teacher_match_probability": float(
            np.mean(teacher_backbone_probability * teacher_action_head_probability)
        ),
        "teacher_backbone_nll_free": _mean(
            -np.log(np.maximum(teacher_backbone_probability[free_local], eps))
        ),
        "teacher_action_head_nll": float(
            -np.log(np.maximum(teacher_action_head_probability, eps)).mean()
        ),
        "predicted_backbone_all": _action_distribution(
            row_predicted_backbone, BACKBONE_NAMES
        ),
        "teacher_backbone_all": _action_distribution(
            row_teacher_backbone, BACKBONE_NAMES
        ),
        "predicted_backbone_free": _action_distribution(
            row_predicted_backbone[free_local], BACKBONE_NAMES
        ),
        "teacher_backbone_free": _action_distribution(
            row_teacher_backbone[free_local], BACKBONE_NAMES
        ),
        "predicted_action_head": _action_distribution(
            row_predicted_action_head, ACTION_HEAD_NAMES
        ),
        "teacher_action_head": _action_distribution(
            row_teacher_action_head, ACTION_HEAD_NAMES
        ),
        "backbone_confusion_free": _confusion_matrix(
            row_teacher_backbone[free_local],
            row_predicted_backbone[free_local],
            len(BACKBONE_NAMES),
        ),
        "action_head_confusion": _confusion_matrix(
            row_teacher_action_head,
            row_predicted_action_head,
            len(ACTION_HEAD_NAMES),
        ),
    }
    return result


def _evaluate_checkpoint(
    checkpoint_path: Path,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model = MaskablePPO.load(str(checkpoint_path), device="cpu")
    model.policy.set_training_mode(False)
    normalizer, normalizer_path = _load_vecnormalize(checkpoint_path)
    observations = dataset["observations"]
    backbone_masks = dataset["backbone_masks"]
    row_count = observations.shape[0]
    backbone_probs = np.empty((row_count, len(BACKBONE_NAMES)), dtype=np.float32)
    action_head_probs = np.empty((row_count, len(ACTION_HEAD_NAMES)), dtype=np.float32)

    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        raw_observation = {
            key: observations[start:stop, index : index + 1]
            for index, key in enumerate(OBSERVATION_KEYS)
        }
        normalized_observation = normalizer.normalize_obs(raw_observation)
        observation_tensor, _ = model.policy.obs_to_tensor(normalized_observation)
        action_masks = np.concatenate(
            [
                backbone_masks[start:stop],
                np.ones((stop - start, len(ACTION_HEAD_NAMES)), dtype=bool),
            ],
            axis=1,
        )
        with torch.inference_mode():
            distribution = model.policy.get_distribution(
                observation_tensor,
                action_masks=action_masks,
            )
            backbone_probs[start:stop] = (
                distribution.distributions[0].probs.detach().cpu().numpy()
            )
            action_head_probs[start:stop] = (
                distribution.distributions[1].probs.detach().cpu().numpy()
            )

    predicted_backbone = backbone_probs.argmax(axis=1).astype(np.int64)
    predicted_action_head = action_head_probs.argmax(axis=1).astype(np.int64)
    all_indices = np.arange(row_count, dtype=np.int64)
    overall = _metrics_for_rows(
        all_indices,
        backbone_masks=backbone_masks,
        teacher_backbone=dataset["teacher_backbone"],
        teacher_action_head=dataset["teacher_action_head"],
        predicted_backbone=predicted_backbone,
        predicted_action_head=predicted_action_head,
        backbone_probs=backbone_probs,
        action_head_probs=action_head_probs,
    )
    per_task: dict[str, Any] = {}
    for task_id, task_name in enumerate(dataset["task_names"]):
        indices = np.flatnonzero(dataset["task_ids"] == task_id)
        metrics = _metrics_for_rows(
            indices,
            backbone_masks=backbone_masks,
            teacher_backbone=dataset["teacher_backbone"],
            teacher_action_head=dataset["teacher_action_head"],
            predicted_backbone=predicted_backbone,
            predicted_action_head=predicted_action_head,
            backbone_probs=backbone_probs,
            action_head_probs=action_head_probs,
        )
        metrics["episodes"] = int(dataset["episodes_per_task"][task_name])
        per_task[task_name] = metrics

    result = {
        "checkpoint": str(checkpoint_path),
        "vecnormalize": str(normalizer_path),
        "overall": overall,
        "per_task": per_task,
    }
    predictions = {
        "backbone": predicted_backbone,
        "action_head": predicted_action_head,
    }
    return result, predictions


def _pairwise_comparisons(
    predictions: dict[str, dict[str, np.ndarray]],
    backbone_masks: np.ndarray,
) -> dict[str, Any]:
    free = backbone_masks.sum(axis=1) > 1
    comparisons: dict[str, Any] = {}
    for left, right in itertools.combinations(predictions, 2):
        left_predictions = predictions[left]
        right_predictions = predictions[right]
        key = f"{left}__vs__{right}"
        comparisons[key] = {
            "backbone_disagreement_all": float(
                np.mean(left_predictions["backbone"] != right_predictions["backbone"])
            ),
            "backbone_disagreement_free": float(
                np.mean(
                    left_predictions["backbone"][free]
                    != right_predictions["backbone"][free]
                )
            ),
            "action_head_disagreement": float(
                np.mean(
                    left_predictions["action_head"]
                    != right_predictions["action_head"]
                )
            ),
            "joint_disagreement": float(
                np.mean(
                    (left_predictions["backbone"] != right_predictions["backbone"])
                    | (
                        left_predictions["action_head"]
                        != right_predictions["action_head"]
                    )
                )
            ),
        }
    return comparisons


def _ratio_text(distribution: dict[str, Any], names: tuple[str, ...]) -> str:
    return "/".join(f"{100.0 * distribution[name]['ratio']:.1f}" for name in names)


def _print_summary(result: dict[str, Any], *, compact: bool) -> None:
    if compact:
        print(
            "checkpoint rows bb_free_match ah_match joint_all bb_entropy ah_entropy "
            "bb_no_image_3_free ah_refresh deterministic_penalty sampled_penalty"
        )
        for label, checkpoint_result in result["checkpoints"].items():
            metrics = checkpoint_result["overall"]
            print(
                f"{label} {metrics['rows']} "
                f"{metrics['backbone_match_free']:.6f} "
                f"{metrics['action_head_match']:.6f} "
                f"{metrics['joint_match_all']:.6f} "
                f"{metrics['backbone_normalized_entropy_free']:.6f} "
                f"{metrics['action_head_normalized_entropy']:.6f} "
                f"{metrics['predicted_backbone_free']['REUSE_THEN_NO_IMAGE_3']['ratio']:.6f} "
                f"{metrics['predicted_action_head']['REFRESH']['ratio']:.6f} "
                f"{metrics['stage1_deterministic_mismatch_penalty_mean']:.6f} "
                f"{metrics['stage1_expected_sampled_mismatch_penalty_mean']:.6f}"
            )
        return

    for label, checkpoint_result in result["checkpoints"].items():
        metrics = checkpoint_result["overall"]
        print(f"\n[{label}] {checkpoint_result['checkpoint']}")
        print(
            "  match: "
            f"backbone-free={100.0 * metrics['backbone_match_free']:.2f}%  "
            f"action-head={100.0 * metrics['action_head_match']:.2f}%  "
            f"joint-all={100.0 * metrics['joint_match_all']:.2f}%"
        )
        print(
            "  entropy(normalized): "
            f"backbone-free={metrics['backbone_normalized_entropy_free']:.3f}  "
            f"action-head={metrics['action_head_normalized_entropy']:.3f}  "
            "stage1-mismatch-penalty="
            f"{metrics['stage1_deterministic_mismatch_penalty_mean']:.3f} deterministic / "
            f"{metrics['stage1_expected_sampled_mismatch_penalty_mean']:.3f} sampled"
        )
        print(
            "  predicted backbone all/free [%]: "
            f"{_ratio_text(metrics['predicted_backbone_all'], BACKBONE_NAMES)} / "
            f"{_ratio_text(metrics['predicted_backbone_free'], BACKBONE_NAMES)}"
        )
        print(
            "  teacher   backbone all/free [%]: "
            f"{_ratio_text(metrics['teacher_backbone_all'], BACKBONE_NAMES)} / "
            f"{_ratio_text(metrics['teacher_backbone_free'], BACKBONE_NAMES)}"
        )
        print(
            "  predicted action-head [%]: "
            f"{_ratio_text(metrics['predicted_action_head'], ACTION_HEAD_NAMES)}"
        )
        print(
            "  teacher   action-head [%]: "
            f"{_ratio_text(metrics['teacher_action_head'], ACTION_HEAD_NAMES)}"
        )
        print("  per task: rows episodes bb-free-match ah-match joint-all")
        for task_name, task_metrics in checkpoint_result["per_task"].items():
            short_name = task_name.rsplit("/", maxsplit=1)[-1]
            print(
                f"    {short_name:32s} {task_metrics['rows']:6d} "
                f"{task_metrics['episodes']:3d} "
                f"{100.0 * task_metrics['backbone_match_free']:7.2f}% "
                f"{100.0 * task_metrics['action_head_match']:7.2f}% "
                f"{100.0 * task_metrics['joint_match_all']:7.2f}%"
            )

    if result["pairwise"]:
        print("\n[pairwise deterministic-action disagreement]")
        for pair, metrics in result["pairwise"].items():
            print(
                f"  {pair}: backbone-free={100.0 * metrics['backbone_disagreement_free']:.2f}% "
                f"action-head={100.0 * metrics['action_head_disagreement']:.2f}% "
                f"joint={100.0 * metrics['joint_disagreement']:.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        type=_parse_checkpoint,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--include-latency-warmup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    try:
        checkpoints = _resolve_checkpoints(args.checkpoint, args.checkpoint_glob)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    torch.set_num_threads(args.torch_threads)
    trace_paths = _resolve_trace_paths(args.trace_glob)
    dataset = _load_traces(
        trace_paths,
        include_latency_warmup=args.include_latency_warmup,
    )
    checkpoint_results: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for label, checkpoint_path in checkpoints:
        checkpoint_result, checkpoint_predictions = _evaluate_checkpoint(
            checkpoint_path,
            dataset,
            batch_size=args.batch_size,
        )
        checkpoint_results[label] = checkpoint_result
        predictions[label] = checkpoint_predictions

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replay_semantics": (
            "One-step offline replay on observations and backbone masks produced by the "
            "saved rule-policy trajectories. Actions do not alter later replay observations."
        ),
        "include_latency_warmup": bool(args.include_latency_warmup),
        "trace_paths": [str(path) for path in trace_paths],
        "rows": int(dataset["observations"].shape[0]),
        "excluded_warmup_rows": int(dataset["excluded_warmup_rows"]),
        "observation_keys": list(OBSERVATION_KEYS),
        "backbone_action_names": list(BACKBONE_NAMES),
        "action_head_action_names": list(ACTION_HEAD_NAMES),
        "checkpoints": checkpoint_results,
        "pairwise": _pairwise_comparisons(predictions, dataset["backbone_masks"]),
    }
    _print_summary(result, compact=args.compact)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
