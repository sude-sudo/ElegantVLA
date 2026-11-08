"""Action error metrics shared by the open-loop evaluator and its tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _action_array(value: Any, *, prefix_steps: int | None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if prefix_steps is not None and array.ndim >= 2:
        array = array[..., : max(int(prefix_steps), 1), :]
    return array


def array_error(reference: Any, candidate: Any) -> dict[str, float | int]:
    reference_array = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Action shapes do not match: "
            f"reference={reference_array.shape}, candidate={candidate_array.shape}"
        )
    if reference_array.size == 0:
        raise ValueError("Cannot compute action error for an empty array.")

    delta = candidate_array - reference_array
    mse = float(np.mean(np.square(delta)))
    reference_rms = float(np.sqrt(np.mean(np.square(reference_array))))
    denominator = float(np.linalg.norm(reference_array) * np.linalg.norm(candidate_array))
    cosine_similarity = (
        1.0
        if denominator == 0.0 and np.array_equal(reference_array, candidate_array)
        else (0.0 if denominator == 0.0 else float(np.dot(reference_array, candidate_array) / denominator))
    )
    return {
        "count": int(delta.size),
        "mae": float(np.mean(np.abs(delta))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "max_abs": float(np.max(np.abs(delta))),
        "reference_rms": reference_rms,
        "nrmse_by_reference_rms": (
            0.0 if reference_rms == 0.0 and mse == 0.0 else float(np.sqrt(mse) / max(reference_rms, 1e-12))
        ),
        "cosine_similarity": cosine_similarity,
    }


def action_dict_error(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    prefix_steps: int | None = None,
) -> dict[str, Any]:
    reference_keys = sorted(reference)
    candidate_keys = sorted(candidate)
    if reference_keys != candidate_keys:
        raise ValueError(
            f"Action keys do not match: reference={reference_keys}, candidate={candidate_keys}"
        )

    reference_arrays: list[np.ndarray] = []
    candidate_arrays: list[np.ndarray] = []
    per_key: dict[str, Any] = {}
    per_dimension: dict[str, Any] = {}
    for key in reference_keys:
        ref = _action_array(reference[key], prefix_steps=prefix_steps)
        pred = _action_array(candidate[key], prefix_steps=prefix_steps)
        if ref.shape != pred.shape:
            raise ValueError(
                f"Action shape mismatch for {key!r}: reference={ref.shape}, candidate={pred.shape}"
            )
        reference_arrays.append(ref.reshape(-1))
        candidate_arrays.append(pred.reshape(-1))
        per_key[key] = array_error(ref, pred)
        if ref.ndim == 0:
            per_dimension[key] = array_error(ref, pred)
        else:
            for dimension in range(ref.shape[-1]):
                per_dimension[f"{key}[{dimension}]"] = array_error(
                    ref[..., dimension],
                    pred[..., dimension],
                )

    return {
        "overall": array_error(
            np.concatenate(reference_arrays),
            np.concatenate(candidate_arrays),
        ),
        "per_key": per_key,
        "per_dimension": per_dimension,
    }


def mean_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
) -> dict[str, float | int]:
    if not rows:
        return {"steps": 0}
    metrics = [row[metric_key]["overall"] for row in rows]
    names = (
        "mae",
        "mse",
        "rmse",
        "max_abs",
        "reference_rms",
        "nrmse_by_reference_rms",
        "cosine_similarity",
    )
    return {
        "steps": int(len(metrics)),
        **{
            name: float(np.mean([float(metric[name]) for metric in metrics]))
            for name in names
        },
    }
