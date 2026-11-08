#!/usr/bin/env python3

"""Class-balanced behavior cloning for the temporal-skip PPO actor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sb3_contrib import MaskablePPO

from elegantvla.gr00t.analysis.replay_rl_policy import (
    ACTION_HEAD_NAMES,
    BACKBONE_NAMES,
    OBSERVATION_KEYS,
    _load_traces,
    _load_vecnormalize,
    _metrics_for_rows,
    _resolve_trace_paths,
)


def _reset_linear(module: nn.Module, *, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _reset_policy(model: MaskablePPO) -> None:
    policy = model.policy
    policy.mlp_extractor.policy_net.apply(
        lambda module: _reset_linear(module, gain=math.sqrt(2.0))
    )
    policy.mlp_extractor.value_net.apply(
        lambda module: _reset_linear(module, gain=math.sqrt(2.0))
    )
    policy.action_net.apply(lambda module: _reset_linear(module, gain=0.01))
    policy.value_net.apply(lambda module: _reset_linear(module, gain=1.0))


def _normalize_observations(
    observations: np.ndarray,
    normalizer: object,
) -> dict[str, np.ndarray]:
    raw = {
        key: observations[:, index : index + 1]
        for index, key in enumerate(OBSERVATION_KEYS)
    }
    normalized = normalizer.normalize_obs(raw)
    if not isinstance(normalized, dict):
        raise TypeError("VecNormalize did not return a dict observation")
    return {
        key: np.asarray(normalized[key], dtype=np.float32)
        for key in OBSERVATION_KEYS
    }


def _class_weights(labels: np.ndarray, class_count: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=class_count).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError(f"training split is missing classes: counts={counts.tolist()}")
    weights = labels.size / (class_count * counts.astype(np.float64))
    return counts, weights.astype(np.float32)


def _policy_logits(
    model: MaskablePPO,
    normalized_observations: dict[str, np.ndarray],
    indices: np.ndarray,
) -> torch.Tensor:
    device = model.policy.device
    observation_tensor = {
        key: torch.as_tensor(values[indices], dtype=torch.float32, device=device)
        for key, values in normalized_observations.items()
    }
    features = model.policy.extract_features(observation_tensor)
    if not isinstance(features, torch.Tensor):
        raise TypeError("separate policy/value feature extractors are not supported")
    latent_policy = model.policy.mlp_extractor.forward_actor(features)
    return model.policy.action_net(latent_policy)


def _predict_probabilities(
    model: MaskablePPO,
    normalized_observations: dict[str, np.ndarray],
    backbone_masks: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    backbone_probabilities = np.empty(
        (indices.size, len(BACKBONE_NAMES)), dtype=np.float32
    )
    action_head_probabilities = np.empty(
        (indices.size, len(ACTION_HEAD_NAMES)), dtype=np.float32
    )
    model.policy.set_training_mode(False)
    with torch.inference_mode():
        for output_start in range(0, indices.size, batch_size):
            output_stop = min(output_start + batch_size, indices.size)
            batch_indices = indices[output_start:output_stop]
            logits = _policy_logits(model, normalized_observations, batch_indices)
            backbone_logits = logits[:, : len(BACKBONE_NAMES)]
            masks = torch.as_tensor(
                backbone_masks[batch_indices],
                dtype=torch.bool,
                device=logits.device,
            )
            backbone_logits = backbone_logits.masked_fill(~masks, -1e8)
            backbone_probabilities[output_start:output_stop] = (
                torch.softmax(backbone_logits, dim=1).cpu().numpy()
            )
            action_head_probabilities[output_start:output_stop] = (
                torch.softmax(logits[:, len(BACKBONE_NAMES) :], dim=1).cpu().numpy()
            )
    return backbone_probabilities, action_head_probabilities


def _macro_recall(confusion: list[list[int]]) -> tuple[float, list[float]]:
    matrix = np.asarray(confusion, dtype=np.float64)
    support = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros_like(support),
        where=support > 0,
    )
    return float(recalls[support > 0].mean()), recalls.tolist()


def _evaluate(
    model: MaskablePPO,
    normalized_observations: dict[str, np.ndarray],
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    backbone_probabilities, action_head_probabilities = _predict_probabilities(
        model,
        normalized_observations,
        dataset["backbone_masks"],
        indices,
        batch_size=batch_size,
    )
    predicted_backbone = backbone_probabilities.argmax(axis=1).astype(np.int64)
    predicted_action_head = action_head_probabilities.argmax(axis=1).astype(np.int64)
    subset_size = indices.size
    subset_metrics = _metrics_for_rows(
        np.arange(subset_size, dtype=np.int64),
        backbone_masks=dataset["backbone_masks"][indices],
        teacher_backbone=dataset["teacher_backbone"][indices],
        teacher_action_head=dataset["teacher_action_head"][indices],
        predicted_backbone=predicted_backbone,
        predicted_action_head=predicted_action_head,
        backbone_probs=backbone_probabilities,
        action_head_probs=action_head_probabilities,
    )
    backbone_macro_recall, backbone_class_recall = _macro_recall(
        subset_metrics["backbone_confusion_free"]
    )
    action_head_macro_recall, action_head_class_recall = _macro_recall(
        subset_metrics["action_head_confusion"]
    )
    subset_metrics.update(
        {
            "backbone_macro_recall_free": backbone_macro_recall,
            "backbone_class_recall_free": {
                name: recall
                for name, recall in zip(
                    BACKBONE_NAMES, backbone_class_recall, strict=True
                )
            },
            "action_head_macro_recall": action_head_macro_recall,
            "action_head_class_recall": {
                name: recall
                for name, recall in zip(
                    ACTION_HEAD_NAMES, action_head_class_recall, strict=True
                )
            },
            "selection_score": 0.5
            * (backbone_macro_recall + action_head_macro_recall),
        }
    )
    return subset_metrics


def _train_epoch(
    model: MaskablePPO,
    optimizer: torch.optim.Optimizer,
    normalized_observations: dict[str, np.ndarray],
    dataset: dict[str, Any],
    train_indices: np.ndarray,
    backbone_class_weights: torch.Tensor,
    action_head_class_weights: torch.Tensor,
    rng: np.random.Generator,
    *,
    batch_size: int,
    max_grad_norm: float,
) -> dict[str, float]:
    model.policy.set_training_mode(True)
    shuffled_indices = rng.permutation(train_indices)
    backbone_loss_sum = 0.0
    action_head_loss_sum = 0.0
    batches = 0
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )

    for start in range(0, shuffled_indices.size, batch_size):
        batch_indices = shuffled_indices[start : start + batch_size]
        logits = _policy_logits(model, normalized_observations, batch_indices)
        masks = torch.as_tensor(
            dataset["backbone_masks"][batch_indices],
            dtype=torch.bool,
            device=logits.device,
        )
        free = masks.sum(dim=1) > 1
        backbone_logits = logits[:, : len(BACKBONE_NAMES)].masked_fill(~masks, -1e8)
        backbone_labels = torch.as_tensor(
            dataset["teacher_backbone"][batch_indices],
            dtype=torch.long,
            device=logits.device,
        )
        action_head_labels = torch.as_tensor(
            dataset["teacher_action_head"][batch_indices],
            dtype=torch.long,
            device=logits.device,
        )
        if not bool(free.any()):
            continue
        backbone_loss = F.cross_entropy(
            backbone_logits[free],
            backbone_labels[free],
            weight=backbone_class_weights,
        )
        action_head_loss = F.cross_entropy(
            logits[:, len(BACKBONE_NAMES) :],
            action_head_labels,
            weight=action_head_class_weights,
        )
        loss = backbone_loss + action_head_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor_parameters, max_grad_norm)
        optimizer.step()
        backbone_loss_sum += float(backbone_loss.detach())
        action_head_loss_sum += float(action_head_loss.detach())
        batches += 1

    if batches == 0:
        raise RuntimeError("no training batches contained a free backbone decision")
    return {
        "backbone_loss": backbone_loss_sum / batches,
        "action_head_loss": action_head_loss_sum / batches,
        "total_loss": (backbone_loss_sum + action_head_loss_sum) / batches,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-seed-modulus", type=int, default=5)
    parser.add_argument("--validation-seed-remainder", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--include-latency-warmup", action="store_true")
    parser.add_argument(
        "--reinitialize-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.validation_seed_modulus <= 1:
        parser.error("--validation-seed-modulus must be greater than 1")
    if not 0 <= args.validation_seed_remainder < args.validation_seed_modulus:
        parser.error("validation seed remainder must be in [0, modulus)")
    for name in ("epochs", "patience", "batch_size", "eval_batch_size", "torch_threads"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")

    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    if not base_checkpoint.is_file():
        parser.error(f"base checkpoint does not exist: {base_checkpoint}")
    output_dir = args.output_dir.expanduser().resolve()
    output_checkpoint = output_dir / "ppo_schedule_bc.zip"
    if output_checkpoint.exists():
        parser.error(f"refusing to overwrite existing checkpoint: {output_checkpoint}")

    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    trace_paths = _resolve_trace_paths(args.trace_glob)
    dataset = _load_traces(
        trace_paths,
        include_latency_warmup=args.include_latency_warmup,
    )
    validation_mask = (
        dataset["seeds"] % args.validation_seed_modulus
        == args.validation_seed_remainder
    )
    validation_indices = np.flatnonzero(validation_mask)
    train_indices = np.flatnonzero(~validation_mask)
    if train_indices.size == 0 or validation_indices.size == 0:
        parser.error("train/validation seed split produced an empty partition")

    model = MaskablePPO.load(str(base_checkpoint), device="cpu")
    normalizer, normalizer_path = _load_vecnormalize(base_checkpoint)
    normalized_observations = _normalize_observations(
        dataset["observations"], normalizer
    )
    expected_keys = tuple(model.policy.observation_space.spaces.keys())
    if expected_keys != OBSERVATION_KEYS:
        raise ValueError(
            f"unexpected PPO observation order: {expected_keys}; expected {OBSERVATION_KEYS}"
        )
    if args.reinitialize_policy:
        _reset_policy(model)

    train_free = dataset["backbone_masks"][train_indices].sum(axis=1) > 1
    backbone_counts, backbone_weights = _class_weights(
        dataset["teacher_backbone"][train_indices][train_free],
        len(BACKBONE_NAMES),
    )
    action_head_counts, action_head_weights = _class_weights(
        dataset["teacher_action_head"][train_indices],
        len(ACTION_HEAD_NAMES),
    )
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    for parameter in actor_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        actor_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )
    backbone_weight_tensor = torch.as_tensor(
        backbone_weights, dtype=torch.float32, device=model.policy.device
    )
    action_head_weight_tensor = torch.as_tensor(
        action_head_weights, dtype=torch.float32, device=model.policy.device
    )

    print(
        f"loaded rows={dataset['observations'].shape[0]} "
        f"train={train_indices.size} validation={validation_indices.size}"
    )
    print(
        f"backbone train counts={backbone_counts.tolist()} "
        f"weights={np.round(backbone_weights, 4).tolist()}"
    )
    print(
        f"action-head train counts={action_head_counts.tolist()} "
        f"weights={np.round(action_head_weights, 4).tolist()}"
    )

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        losses = _train_epoch(
            model,
            optimizer,
            normalized_observations,
            dataset,
            train_indices,
            backbone_weight_tensor,
            action_head_weight_tensor,
            rng,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
        )
        validation_metrics = _evaluate(
            model,
            normalized_observations,
            dataset,
            validation_indices,
            batch_size=args.eval_batch_size,
        )
        score = float(validation_metrics["selection_score"])
        history.append(
            {
                "epoch": epoch,
                **losses,
                "validation_selection_score": score,
                "validation_backbone_macro_recall_free": validation_metrics[
                    "backbone_macro_recall_free"
                ],
                "validation_action_head_macro_recall": validation_metrics[
                    "action_head_macro_recall"
                ],
                "validation_backbone_match_free": validation_metrics[
                    "backbone_match_free"
                ],
                "validation_action_head_match": validation_metrics[
                    "action_head_match"
                ],
                "validation_joint_match_free_backbone": validation_metrics[
                    "joint_match_free_backbone"
                ],
            }
        )
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.policy.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"epoch={epoch:03d} loss={losses['total_loss']:.5f} "
                f"val_macro_bb={validation_metrics['backbone_macro_recall_free']:.4f} "
                f"val_macro_ah={validation_metrics['action_head_macro_recall']:.4f} "
                f"val_joint_free={validation_metrics['joint_match_free_backbone']:.4f} "
                f"best={best_score:.4f}@{best_epoch}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a best policy state")
    model.policy.load_state_dict(best_state)
    final_train_metrics = _evaluate(
        model,
        normalized_observations,
        dataset,
        train_indices,
        batch_size=args.eval_batch_size,
    )
    final_validation_metrics = _evaluate(
        model,
        normalized_observations,
        dataset,
        validation_indices,
        batch_size=args.eval_batch_size,
    )
    all_indices = np.arange(dataset["observations"].shape[0], dtype=np.int64)
    final_all_metrics = _evaluate(
        model,
        normalized_observations,
        dataset,
        all_indices,
        batch_size=args.eval_batch_size,
    )

    for parameter in model.policy.parameters():
        parameter.requires_grad_(True)
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=float(model.lr_schedule(1.0)),
        **model.policy.optimizer_kwargs,
    )
    model.num_timesteps = 0
    model._n_updates = 0
    model._current_progress_remaining = 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_checkpoint))
    output_normalizer = output_dir / "ppo_schedule_bc_vecnormalize.pkl"
    shutil.copy2(normalizer_path, output_normalizer)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "inverse-frequency class-balanced cross-entropy behavior cloning",
        "base_checkpoint": str(base_checkpoint),
        "source_vecnormalize": str(normalizer_path),
        "saved_checkpoint": str(output_checkpoint),
        "saved_vecnormalize": str(output_normalizer),
        "trace_paths": [str(path) for path in trace_paths],
        "config": vars(args),
        "split": {
            "train_rows": int(train_indices.size),
            "validation_rows": int(validation_indices.size),
            "validation_rule": (
                f"seed % {args.validation_seed_modulus} == "
                f"{args.validation_seed_remainder}"
            ),
        },
        "class_balance": {
            "backbone_names": list(BACKBONE_NAMES),
            "backbone_counts_free": backbone_counts,
            "backbone_weights": backbone_weights,
            "action_head_names": list(ACTION_HEAD_NAMES),
            "action_head_counts": action_head_counts,
            "action_head_weights": action_head_weights,
        },
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "history": history,
        "metrics": {
            "train": final_train_metrics,
            "validation": final_validation_metrics,
            "all": final_all_metrics,
        },
    }
    metadata_path = output_dir / "bc_training.json"
    metadata_path.write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "best policy: "
        f"epoch={best_epoch} "
        f"val_macro_bb={final_validation_metrics['backbone_macro_recall_free']:.4f} "
        f"val_macro_ah={final_validation_metrics['action_head_macro_recall']:.4f} "
        f"val_joint_free={final_validation_metrics['joint_match_free_backbone']:.4f}"
    )
    print(f"saved {output_checkpoint}")
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()
