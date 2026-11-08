"""Cache features, layer masks and experiment record utilities."""

from collections.abc import Sequence
import hashlib
import json
import math
import pathlib
from typing import Any

import numpy as np
import torch

from elegantvla.openpi.legacy.config import PrefixVLMCacheConfig


PREFIX_VLM_ACTION_DEVIATION_GATE_FEATURES = frozenset(
    {
        "prefix_vlm_cache_first_layer_cka",
        "prefix_vlm_cache_first_layer_l2",
        "prefix_vlm_cache_image_signature_l2",
        "prefix_vlm_cache_state_delta",
        "prefix_vlm_cache_state_translation_delta",
        "prefix_vlm_cache_state_translation_window_delta",
        "prefix_vlm_cache_state_rotation_delta",
        "prefix_vlm_cache_state_gripper_delta",
        "prefix_vlm_cache_cached_state_delta",
        "prefix_vlm_cache_stable_state_deltas",
        "prefix_vlm_cache_next_stable_state_deltas",
        "prefix_vlm_cache_episode_query",
        "prefix_vlm_cache_continuous_hits",
        "prefix_vlm_cache_motion_signal",
        "prefix_vlm_cache_bank_selected_cka",
        "first_layer_cka",
        "first_layer_l2",
        "image_signature_l2",
        "state_delta",
        "state_translation_delta",
        "state_rotation_delta",
        "state_gripper_delta",
        "cached_state_delta",
        "episode_query",
        "continuous_hits",
        "motion_signal",
        "bank_selected_cka",
    }
)


PREFIX_VLM_ACTION_DEVIATION_GATE_CKA_FEATURES = frozenset(
    {
        "prefix_vlm_cache_first_layer_cka",
        "prefix_vlm_cache_bank_selected_cka",
        "first_layer_cka",
        "bank_selected_cka",
    }
)


def _prefix_prompt_key(inputs: dict) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for key in ("tokenized_prompt", "tokenized_prompt_mask"):
        value = inputs.get(key)
        if value is not None:
            arr = np.asarray(value)
            hasher.update(str(arr.shape).encode())
            hasher.update(arr.tobytes())
    for key, value in sorted(inputs.get("image_mask", {}).items()):
        arr = np.asarray(value)
        hasher.update(key.encode())
        hasher.update(str(arr.shape).encode())
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _prefix_raw_prompt(raw_inputs: dict | None) -> str:
    if raw_inputs is None:
        return ""
    for key in ("prompt", "task_description", "language_instruction"):
        if key not in raw_inputs:
            continue
        value = raw_inputs[key]
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else " ".join(str(item) for item in value.ravel())
        return str(value).strip().lower()
    return ""


def _prefix_prompt_matches(prompt: str, spec: str | None) -> bool:
    prompt = str(prompt or "").lower()
    if not prompt or not spec:
        return False
    for item in str(spec).replace("||", ";").replace(",", ";").split(";"):
        needle = item.strip().lower()
        if needle and needle in prompt:
            return True
    return False


def _load_prefix_vlm_adaptive_scheduler(path: str | pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        return None
    scheduler_path = pathlib.Path(path).expanduser()
    with scheduler_path.open() as f:
        scheduler = json.load(f)
    if scheduler.get("type") != "threshold_prefix_cache_scheduler":
        raise ValueError(f"Unsupported prefix VLM scheduler type in {scheduler_path}: {scheduler.get('type')!r}")
    schedules = scheduler.get("schedules")
    if not isinstance(schedules, list) or not schedules:
        raise ValueError(f"Prefix VLM scheduler {scheduler_path} does not contain any schedules")
    return scheduler


def _load_prefix_vlm_action_deviation_gate(path: str | pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        return None
    gate_path = pathlib.Path(path).expanduser()
    with gate_path.open() as f:
        payload = json.load(f)
    if payload.get("type") == "action_deviation_prefix_cache_gate":
        gate = payload
    elif isinstance(payload.get("recommended_policy_gate"), dict):
        gate = payload["recommended_policy_gate"]
    elif isinstance(payload.get("policy_gate"), dict):
        gate = payload["policy_gate"]
    elif isinstance(payload.get("top_gates"), list) and payload["top_gates"]:
        gate = {
            "type": "action_deviation_prefix_cache_gate",
            "name": "top_gate_1",
            "conditions": payload["top_gates"][0].get("conditions", []),
            "source_metrics": {
                key: payload["top_gates"][0].get(key)
                for key in (
                    "safe_accept",
                    "unsafe_accept",
                    "safe_total",
                    "unsafe_total",
                    "safe_accept_rate",
                    "unsafe_accept_rate",
                    "precision",
                )
            },
        }
    else:
        raise ValueError(f"Unsupported action-deviation gate JSON in {gate_path}")
    if "max_unsafe_accept_rate" not in gate and payload.get("max_unsafe_accept_rate") is not None:
        gate = dict(gate)
        gate["max_unsafe_accept_rate"] = payload.get("max_unsafe_accept_rate")

    conditions = gate.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"Action-deviation gate {gate_path} does not contain any conditions")
    for condition in conditions:
        if not isinstance(condition, dict) or "feature" not in condition or "threshold" not in condition:
            raise ValueError(f"Malformed action-deviation gate condition in {gate_path}: {condition!r}")
        feature = str(condition.get("feature", ""))
        if feature not in PREFIX_VLM_ACTION_DEVIATION_GATE_FEATURES:
            raise ValueError(f"Unsupported action-deviation gate feature in {gate_path}: {feature!r}")
        op = str(condition.get("op", "<="))
        if op not in {"<=", ">=", "<", ">", "=="}:
            raise ValueError(f"Unsupported action-deviation gate op in {gate_path}: {op!r}")
    if all(
        str(condition.get("feature", "")) in PREFIX_VLM_ACTION_DEVIATION_GATE_CKA_FEATURES
        for condition in conditions
    ):
        raise ValueError(f"Refusing CKA-only action-deviation gate in {gate_path}")
    if bool(gate.get("cka_only_gate", False)):
        raise ValueError(f"Refusing gate marked CKA-only in {gate_path}")
    if gate.get("deployable_with_current_policy") is False:
        raise ValueError(f"Action-deviation gate {gate_path} is marked non-deployable")
    unknown_policy_features = gate.get("unknown_policy_features") or []
    if unknown_policy_features:
        raise ValueError(f"Action-deviation gate {gate_path} has unknown policy features: {unknown_policy_features}")
    _validate_prefix_vlm_action_deviation_gate_metrics(gate, gate_path)
    _validate_prefix_vlm_action_deviation_gate_label_integrity(gate, gate_path)
    gate = dict(gate)
    gate["type"] = "action_deviation_prefix_cache_gate"
    return gate


def _prefix_gate_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _validate_prefix_vlm_action_deviation_gate_metrics(gate: dict[str, Any], gate_path: pathlib.Path) -> None:
    metrics = gate.get("source_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Action-deviation gate {gate_path} is missing source metrics")
    required_positive = ("safe_total", "unsafe_total", "safe_accept")
    for key in required_positive:
        value = _prefix_gate_float(metrics.get(key))
        if value is None or value <= 0:
            raise ValueError(f"Action-deviation gate {gate_path} has invalid source metric {key}: {metrics.get(key)!r}")
    unsafe_accept_rate = _prefix_gate_float(metrics.get("unsafe_accept_rate"))
    if unsafe_accept_rate is None:
        raise ValueError(f"Action-deviation gate {gate_path} is missing source unsafe_accept_rate")
    max_unsafe_accept_rate = _prefix_gate_float(gate.get("max_unsafe_accept_rate"))
    if max_unsafe_accept_rate is None:
        max_unsafe_accept_rate = 0.02
    if unsafe_accept_rate > max_unsafe_accept_rate:
        raise ValueError(
            f"Action-deviation gate {gate_path} source unsafe accept rate "
            f"{unsafe_accept_rate:.6f} exceeds {max_unsafe_accept_rate:.6f}"
        )


def _prefix_gate_true_count(counts: dict[str, Any]) -> int:
    total = 0
    for value in ("True", "true", "1"):
        raw_count = counts.get(value, 0)
        try:
            total += int(raw_count)
        except (TypeError, ValueError):
            return 0
    return total


def _validate_prefix_vlm_action_deviation_gate_label_integrity(
    gate: dict[str, Any],
    gate_path: pathlib.Path,
) -> None:
    provenance = gate.get("training_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Action-deviation gate {gate_path} is missing training provenance")
    for key in (
        "prefix_vlm_cache_action_deviation_probe_enabled",
        "prefix_vlm_cache_action_deviation_shared_noise",
        "prefix_vlm_cache_action_deviation_candidate_suffix_cache_isolated",
    ):
        counts = provenance.get(key)
        if not isinstance(counts, dict):
            raise ValueError(f"Action-deviation gate {gate_path} is missing provenance counts for {key}")
        true_count = _prefix_gate_true_count(counts)
        non_true_count = 0
        for value, count in counts.items():
            if value in {"True", "true", "1"}:
                continue
            try:
                non_true_count += int(count)
            except (TypeError, ValueError):
                non_true_count += 1
        if true_count <= 0 or non_true_count:
            raise ValueError(f"Action-deviation gate {gate_path} failed label-integrity check for {key}: {counts}")


def _threshold_condition_accepts(value: float, op: str, threshold: float) -> bool:
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == ">":
        return value > threshold
    if op == "==":
        return value == threshold
    return False


def _prefix_image_signature(inputs: dict) -> np.ndarray:
    parts = []
    for key, image in sorted(inputs.get("image", {}).items()):
        arr = np.asarray(image, dtype=np.float32)
        parts.append(arr.mean(axis=tuple(range(arr.ndim - 1))))
        parts.append(arr.std(axis=tuple(range(arr.ndim - 1))))
        parts.append(np.asarray([arr.mean(), arr.std()], dtype=np.float32))
        mask = np.asarray(inputs.get("image_mask", {}).get(key, True), dtype=np.float32).reshape(-1)
        parts.append(mask)
    if not parts:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate([np.ravel(p).astype(np.float32) for p in parts])


def _prefix_state_signature(raw_inputs: dict | None, inputs: dict | None) -> np.ndarray | None:
    if raw_inputs is not None and "observation/state" in raw_inputs:
        return np.ravel(np.asarray(raw_inputs["observation/state"], dtype=np.float32))
    if inputs is not None and "state" in inputs:
        return np.ravel(np.asarray(inputs["state"], dtype=np.float32))
    return None


def _prefix_raw_images(raw_inputs: dict | None) -> dict[str, np.ndarray] | None:
    if raw_inputs is None:
        return None
    if "observation/image" not in raw_inputs or "observation/wrist_image" not in raw_inputs:
        return None
    return {
        "base": np.asarray(raw_inputs["observation/image"], dtype=np.float32),
        "wrist": np.asarray(raw_inputs["observation/wrist_image"], dtype=np.float32),
    }


def _prefix_compute_layer_mask(
    depth: int,
    explicit_mask: str | None,
    fn_blocks: int,
    bn_blocks: int,
    strategy: str = "first_last",
    compute_layer_count: int = 0,
) -> list[bool]:
    if explicit_mask is not None and explicit_mask.strip():
        mask_text = explicit_mask.strip().lower().replace("-", "_")
        if mask_text in {"none", "empty", "all_reuse", "reuse_all"}:
            return [False] * depth
        mask = [False] * depth
        for item in explicit_mask.replace(" ", "").split(","):
            if not item:
                continue
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise ValueError(f"Invalid descending prefix cache layer range: {item!r}")
                for layer_idx in range(start, end + 1):
                    if layer_idx < 0 or layer_idx >= depth:
                        raise ValueError(f"Prefix cache layer {layer_idx} out of range for depth {depth}")
                    mask[layer_idx] = True
            else:
                layer_idx = int(item)
                if layer_idx < 0 or layer_idx >= depth:
                    raise ValueError(f"Prefix cache layer {layer_idx} out of range for depth {depth}")
                mask[layer_idx] = True
        return mask

    compute_layer_count = int(compute_layer_count)
    if compute_layer_count > 0:
        strategy = str(strategy or "first_last").strip().lower().replace("-", "_")
        if strategy in {"uniform", "uniform_sparse", "anchor_sparse", "even"}:
            return _prefix_uniform_sparse_layer_mask(depth, compute_layer_count)
        if strategy not in {"first_last", "firstlast", "edge", "edges"}:
            raise ValueError(f"Unsupported prefix cache layer strategy: {strategy!r}")
        fn_blocks = int(math.ceil(compute_layer_count / 2))
        bn_blocks = compute_layer_count // 2

    head = min(max(fn_blocks, 0), depth)
    tail = min(max(bn_blocks, 0), depth)
    return [layer_idx < head or layer_idx >= depth - tail for layer_idx in range(depth)]


def _prefix_uniform_sparse_layer_mask(depth: int, compute_layer_count: int) -> list[bool]:
    if depth <= 0:
        return []
    count = min(max(int(compute_layer_count), 0), depth)
    if count <= 0:
        return [False] * depth
    if count == 1:
        indices = [0]
    else:
        indices = sorted({int(round(x)) for x in np.linspace(0, depth - 1, count)})
        # Rounding can collapse adjacent anchors for small depths. Fill the
        # largest remaining gaps so the requested compute budget is respected.
        while len(indices) < count:
            gaps = []
            prev = -1
            for current in indices + [depth]:
                if current - prev > 1:
                    gaps.append((current - prev, prev, current))
                prev = current
            if not gaps:
                break
            _gap, left, right = max(gaps)
            indices.append((left + right) // 2)
            indices = sorted(set(indices))
    mask = [False] * depth
    for layer_idx in indices[:count]:
        mask[layer_idx] = True
    return mask


def _prefix_mask_to_spec(mask: Sequence[bool]) -> str:
    parts = []
    start = None
    for layer_idx, enabled in enumerate(mask):
        if enabled and start is None:
            start = layer_idx
        if start is not None and (not enabled or layer_idx == len(mask) - 1):
            end = layer_idx if enabled and layer_idx == len(mask) - 1 else layer_idx - 1
            parts.append(str(start) if start == end else f"{start}-{end}")
            start = None
    return ",".join(parts)


def _torch_gated_residual(residual: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return residual + delta
    return residual + delta * gate


def _state_translation_delta(current: np.ndarray, previous: np.ndarray) -> float:
    if current.shape != previous.shape:
        return np.inf
    diff = np.ravel(np.asarray(current, dtype=np.float32) - np.asarray(previous, dtype=np.float32))
    translation = diff[:3] if diff.shape[0] >= 3 else diff
    return float(np.linalg.norm(translation))


def _state_delta_components(current: np.ndarray, previous: np.ndarray | None) -> dict[str, float]:
    if previous is None or current.shape != previous.shape:
        return {
            "state_delta": np.inf,
            "state_translation_delta": np.inf,
            "state_rotation_delta": np.inf,
            "state_gripper_delta": np.inf,
        }
    diff = np.ravel(np.asarray(current, dtype=np.float32) - np.asarray(previous, dtype=np.float32))
    rotation = diff[3:6] if diff.shape[0] >= 6 else np.zeros((0,), dtype=np.float32)
    gripper = diff[6:] if diff.shape[0] > 6 else np.zeros((0,), dtype=np.float32)
    return {
        "state_delta": float(np.linalg.norm(diff)),
        "state_translation_delta": _state_translation_delta(current, previous),
        "state_rotation_delta": float(np.linalg.norm(rotation)),
        "state_gripper_delta": float(np.linalg.norm(gripper)),
    }


def _prefix_state_info_to_cache_info(state_info: dict[str, Any]) -> dict[str, Any]:
    if not state_info:
        return {}
    return {
        "prefix_vlm_cache_state_guard_enabled": bool(state_info.get("enabled", False)),
        "prefix_vlm_cache_state_guard_reason": str(state_info.get("reason", "")),
        "prefix_vlm_cache_state_guard_accept": bool(state_info.get("accept", False)),
        "prefix_vlm_cache_state_delta": float(state_info.get("state_delta", np.inf)),
        "prefix_vlm_cache_state_translation_delta": float(state_info.get("state_translation_delta", np.inf)),
        "prefix_vlm_cache_state_translation_window_delta": float(
            state_info.get("state_translation_window_delta", np.inf)
        ),
        "prefix_vlm_cache_state_rotation_delta": float(state_info.get("state_rotation_delta", np.inf)),
        "prefix_vlm_cache_state_gripper_delta": float(state_info.get("state_gripper_delta", np.inf)),
        "prefix_vlm_cache_cached_state_delta": float(state_info.get("cached_state_delta", np.inf)),
        "prefix_vlm_cache_stable_state_deltas": int(state_info.get("stable_count", 0)),
        "prefix_vlm_cache_next_stable_state_deltas": int(state_info.get("next_stable_count", 0)),
    }


def _prefix_observation_info_to_cache_info(observation_info: dict[str, Any]) -> dict[str, Any]:
    if not observation_info:
        return {}
    return {
        "prefix_vlm_cache_observation_gate_enabled": bool(observation_info.get("enabled", False)),
        "prefix_vlm_cache_observation_gate_accept": bool(observation_info.get("accept", False)),
        "prefix_vlm_cache_observation_gate_reason": str(observation_info.get("reason", "")),
        "prefix_vlm_cache_base_image_mean_abs": float(observation_info.get("base_image_mean_abs", np.inf)),
        "prefix_vlm_cache_wrist_image_rel_l2": float(observation_info.get("wrist_image_rel_l2", np.inf)),
    }


def _prefix_action_deviation_info(
    full_actions: np.ndarray | None,
    cached_actions: np.ndarray | None,
    config: PrefixVLMCacheConfig,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "prefix_vlm_cache_action_deviation_evaluated": False,
        "prefix_vlm_cache_action_deviation_reason": "missing",
    }
    if full_actions is None or cached_actions is None:
        return info

    full = np.asarray(full_actions, dtype=np.float32)
    cached = np.asarray(cached_actions, dtype=np.float32)
    if full.shape != cached.shape:
        info.update(
            {
                "prefix_vlm_cache_action_deviation_reason": "shape_mismatch",
                "prefix_vlm_cache_action_deviation_full_shape": str(tuple(full.shape)),
                "prefix_vlm_cache_action_deviation_cached_shape": str(tuple(cached.shape)),
            }
        )
        return info

    diff = cached - full
    abs_diff = np.abs(diff)
    flat = diff.reshape(-1)
    full_steps = full.reshape(1, -1) if full.ndim == 1 else full.reshape(-1, full.shape[-1])
    cached_steps = cached.reshape(1, -1) if cached.ndim == 1 else cached.reshape(-1, cached.shape[-1])
    diff_steps = cached_steps - full_steps
    action_dim = int(diff_steps.shape[-1]) if diff_steps.ndim == 2 else int(diff.shape[-1])

    first_step = diff_steps[0] if diff_steps.size else np.zeros((0,), dtype=np.float32)
    translation = diff_steps[:, :3] if action_dim >= 3 else np.zeros((diff_steps.shape[0], 0), dtype=np.float32)
    rotation = diff_steps[:, 3:6] if action_dim >= 6 else np.zeros((diff_steps.shape[0], 0), dtype=np.float32)
    gripper_abs = np.zeros((0,), dtype=np.float32)
    gripper_flip = False
    if action_dim >= 7:
        full_gripper = full_steps[:, -1]
        cached_gripper = cached_steps[:, -1]
        gripper_abs = np.abs(cached_gripper - full_gripper)
        eps = float(config.action_deviation_gripper_flip_eps)
        active = (np.abs(full_gripper) > eps) | (np.abs(cached_gripper) > eps)
        gripper_flip = bool(np.any(active & (np.sign(full_gripper) != np.sign(cached_gripper))))

    l2 = float(np.linalg.norm(flat))
    rmse = float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0
    linf = float(np.max(abs_diff)) if abs_diff.size else 0.0
    mean_abs = float(np.mean(abs_diff)) if abs_diff.size else 0.0
    translation_l2 = float(np.linalg.norm(translation)) if translation.size else 0.0
    rotation_l2 = float(np.linalg.norm(rotation)) if rotation.size else 0.0
    eef_position_l2 = 0.0
    eef_position_step_linf = 0.0
    if translation.size:
        cumulative_translation = np.cumsum(translation, axis=0)
        eef_position_l2 = float(np.linalg.norm(cumulative_translation[-1]))
        eef_position_step_linf = float(np.max(np.linalg.norm(cumulative_translation, axis=1)))
    eef_rotation_l2 = 0.0
    eef_rotation_step_linf = 0.0
    if rotation.size:
        cumulative_rotation = np.cumsum(rotation, axis=0)
        eef_rotation_l2 = float(np.linalg.norm(cumulative_rotation[-1]))
        eef_rotation_step_linf = float(np.max(np.linalg.norm(cumulative_rotation, axis=1)))
    eef_pose_l2 = float(np.linalg.norm([eef_position_l2, eef_rotation_l2]))
    gripper_abs_max = float(np.max(gripper_abs)) if gripper_abs.size else 0.0

    checks = []
    if config.action_deviation_l2_threshold >= 0:
        checks.append(l2 <= float(config.action_deviation_l2_threshold))
    if config.action_deviation_translation_l2_threshold >= 0:
        checks.append(translation_l2 <= float(config.action_deviation_translation_l2_threshold))
    if config.action_deviation_rotation_l2_threshold >= 0:
        checks.append(rotation_l2 <= float(config.action_deviation_rotation_l2_threshold))
    if config.action_deviation_eef_position_l2_threshold >= 0:
        checks.append(eef_position_l2 <= float(config.action_deviation_eef_position_l2_threshold))
    if config.action_deviation_eef_rotation_l2_threshold >= 0:
        checks.append(eef_rotation_l2 <= float(config.action_deviation_eef_rotation_l2_threshold))
    if config.action_deviation_eef_pose_l2_threshold >= 0:
        checks.append(eef_pose_l2 <= float(config.action_deviation_eef_pose_l2_threshold))
    if config.action_deviation_gripper_abs_threshold >= 0:
        checks.append(gripper_abs_max <= float(config.action_deviation_gripper_abs_threshold))
    if action_dim >= 7 and not bool(config.action_deviation_allow_gripper_flip):
        checks.append(not gripper_flip)

    info.update(
        {
            "prefix_vlm_cache_action_deviation_evaluated": True,
            "prefix_vlm_cache_action_deviation_reason": "ok",
            "prefix_vlm_cache_action_deviation_horizon": int(full_steps.shape[0]),
            "prefix_vlm_cache_action_deviation_dim": action_dim,
            "prefix_vlm_cache_action_deviation_l2": l2,
            "prefix_vlm_cache_action_deviation_rmse": rmse,
            "prefix_vlm_cache_action_deviation_linf": linf,
            "prefix_vlm_cache_action_deviation_mean_abs": mean_abs,
            "prefix_vlm_cache_action_deviation_first_step_l2": float(np.linalg.norm(first_step)),
            "prefix_vlm_cache_action_deviation_first_step_linf": float(np.max(np.abs(first_step)))
            if first_step.size
            else 0.0,
            "prefix_vlm_cache_action_deviation_translation_l2": translation_l2,
            "prefix_vlm_cache_action_deviation_translation_linf": float(np.max(np.abs(translation)))
            if translation.size
            else 0.0,
            "prefix_vlm_cache_action_deviation_rotation_l2": rotation_l2,
            "prefix_vlm_cache_action_deviation_rotation_linf": float(np.max(np.abs(rotation)))
            if rotation.size
            else 0.0,
            "prefix_vlm_cache_action_deviation_eef_position_l2": eef_position_l2,
            "prefix_vlm_cache_action_deviation_eef_position_step_linf": eef_position_step_linf,
            "prefix_vlm_cache_action_deviation_eef_rotation_l2": eef_rotation_l2,
            "prefix_vlm_cache_action_deviation_eef_rotation_step_linf": eef_rotation_step_linf,
            "prefix_vlm_cache_action_deviation_eef_pose_l2": eef_pose_l2,
            "prefix_vlm_cache_action_deviation_gripper_abs": gripper_abs_max,
            "prefix_vlm_cache_action_deviation_gripper_flip": gripper_flip,
            "prefix_vlm_cache_action_deviation_thresholds_enabled": bool(checks),
            "prefix_vlm_cache_action_deviation_safe_label": bool(all(checks)) if checks else None,
        }
    )
    return info


def _relative_l2(current: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None or current.shape != previous.shape:
        return np.inf
    denom = max(float(np.linalg.norm(previous)), 1e-6)
    return float(np.linalg.norm(current - previous) / denom)


def _torch_relative_l2(current: torch.Tensor, previous: torch.Tensor | None) -> float:
    if previous is None or current.shape != previous.shape:
        return np.inf
    current_f32 = current.detach().to(torch.float32)
    previous_f32 = previous.detach().to(device=current_f32.device, dtype=torch.float32)
    denom = torch.linalg.vector_norm(previous_f32).clamp_min(1e-6)
    return float((torch.linalg.vector_norm(current_f32 - previous_f32) / denom).detach().cpu())


def _torch_linear_cka(current: torch.Tensor, previous: torch.Tensor | None, token_mask: torch.Tensor | None) -> float:
    if previous is None or current.shape != previous.shape:
        return -np.inf
    current_f32 = current.detach().to(torch.float32)
    previous_f32 = previous.detach().to(device=current_f32.device, dtype=torch.float32)
    if token_mask is not None:
        mask = token_mask.detach().to(device=current_f32.device).bool()
        if mask.shape != current_f32.shape[:-1]:
            return -np.inf
        current_f32 = current_f32[mask]
        previous_f32 = previous_f32[mask]
    else:
        current_f32 = current_f32.reshape(-1, current_f32.shape[-1])
        previous_f32 = previous_f32.reshape(-1, previous_f32.shape[-1])
    if current_f32.shape[0] < 2:
        return -np.inf

    current_f32 = current_f32 - current_f32.mean(dim=0, keepdim=True)
    previous_f32 = previous_f32 - previous_f32.mean(dim=0, keepdim=True)
    cross = current_f32 @ previous_f32.T
    current_gram = current_f32 @ current_f32.T
    previous_gram = previous_f32 @ previous_f32.T
    cross_norm_sq = torch.square(cross).sum()
    denom = torch.sqrt(torch.square(current_gram).sum()).clamp_min(1e-12)
    denom = denom * torch.sqrt(torch.square(previous_gram).sum()).clamp_min(1e-12)
    return float((cross_norm_sq / denom).clamp(0.0, 1.0).detach().cpu())


def _kv_cache_has_layer(current_kv: Any | None, previous_kv: Any | None, layer_idx: int) -> bool:
    if current_kv is None or previous_kv is None:
        return False
    if not hasattr(current_kv, "key_cache") or not hasattr(previous_kv, "key_cache"):
        return False
    if not hasattr(current_kv, "value_cache") or not hasattr(previous_kv, "value_cache"):
        return False
    return (
        layer_idx < len(current_kv.key_cache)
        and layer_idx < len(previous_kv.key_cache)
        and layer_idx < len(current_kv.value_cache)
        and layer_idx < len(previous_kv.value_cache)
    )
