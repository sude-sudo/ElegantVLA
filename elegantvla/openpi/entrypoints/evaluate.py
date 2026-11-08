from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import time
from typing import Any

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import torch
import tqdm

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
_TORCH_LOAD = torch.load


def _torch_load_libero_compat(*args, **kwargs):
    # LIBERO init-state files are trusted local benchmark artifacts created
    # before PyTorch 2.6 changed torch.load's default to weights_only=True.
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_libero_compat
ACTION_DEVIATION_TIMING_KEYS = (
    "prefix_vlm_cache_action_deviation_label_enabled",
    "prefix_vlm_cache_action_deviation_candidate_mode",
    "prefix_vlm_cache_action_deviation_candidate",
    "prefix_vlm_cache_action_deviation_candidate_reason",
    "prefix_vlm_cache_action_deviation_candidate_compute_layer_mask",
    "prefix_vlm_cache_action_deviation_probe_enabled",
    "prefix_vlm_cache_action_deviation_shared_noise",
    "prefix_vlm_cache_action_deviation_candidate_suffix_cache_isolated",
    "prefix_vlm_cache_action_deviation_policy_shadow_hit",
    "prefix_vlm_cache_action_deviation_pre_guard_hit",
    "prefix_vlm_cache_action_deviation_pre_guard_reason",
    "prefix_vlm_cache_action_deviation_gate_enabled",
    "prefix_vlm_cache_action_deviation_gate_replace_legacy_guards",
    "prefix_vlm_cache_action_deviation_gate_accept",
    "prefix_vlm_cache_action_deviation_gate_reason",
    "prefix_vlm_cache_action_deviation_evaluated",
    "prefix_vlm_cache_action_deviation_reason",
    "prefix_vlm_cache_action_deviation_horizon",
    "prefix_vlm_cache_action_deviation_dim",
    "prefix_vlm_cache_action_deviation_l2",
    "prefix_vlm_cache_action_deviation_rmse",
    "prefix_vlm_cache_action_deviation_linf",
    "prefix_vlm_cache_action_deviation_mean_abs",
    "prefix_vlm_cache_action_deviation_first_step_l2",
    "prefix_vlm_cache_action_deviation_first_step_linf",
    "prefix_vlm_cache_action_deviation_translation_l2",
    "prefix_vlm_cache_action_deviation_translation_linf",
    "prefix_vlm_cache_action_deviation_rotation_l2",
    "prefix_vlm_cache_action_deviation_rotation_linf",
    "prefix_vlm_cache_action_deviation_eef_position_l2",
    "prefix_vlm_cache_action_deviation_eef_position_step_linf",
    "prefix_vlm_cache_action_deviation_eef_rotation_l2",
    "prefix_vlm_cache_action_deviation_eef_rotation_step_linf",
    "prefix_vlm_cache_action_deviation_eef_pose_l2",
    "prefix_vlm_cache_action_deviation_gripper_abs",
    "prefix_vlm_cache_action_deviation_gripper_flip",
    "prefix_vlm_cache_action_deviation_thresholds_enabled",
    "prefix_vlm_cache_action_deviation_safe_label",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small LIBERO eval harness for OpenPI cache threshold sweeps.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--num-trials-per-task", type=int, default=2)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output", default="output_libero_cache_threshold_eval.json")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-out-path", default="data/libero/videos_cache_thresholds")
    return parser.parse_args()


def _task_ids(spec: str, n_tasks: int) -> list[int]:
    spec = str(spec).strip().lower()
    if spec in {"all", "*"}:
        return list(range(n_tasks))
    ids: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(chunk))
    return [idx for idx in ids if 0 <= idx < n_tasks]


def _default_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _write_output(path: pathlib.Path, args: argparse.Namespace, episodes: list[dict[str, Any]]) -> None:
    successes = sum(1 for episode in episodes if episode["success"])
    query_records = [query for episode in episodes for query in episode["queries"]]
    policy_ms = [float(q["policy_infer_ms"]) for q in query_records if q.get("policy_infer_ms") is not None]
    server_ms = [float(q["server_infer_ms"]) for q in query_records if q.get("server_infer_ms") is not None]
    roundtrip_ms = [float(q["roundtrip_ms"]) for q in query_records if q.get("roundtrip_ms") is not None]
    cache_hits = sum(1 for q in query_records if q.get("prefix_vlm_cache_hit"))
    no_image_hits = sum(1 for q in query_records if q.get("prefix_vlm_cache_skip_v4_no_image"))
    bank_selected = sum(1 for q in query_records if q.get("prefix_vlm_cache_bank_selected"))
    reused_blocks = sum(float(q.get("prefix_vlm_cache_reused_blocks") or 0.0) for q in query_records)
    computed_blocks = sum(float(q.get("prefix_vlm_cache_computed_blocks") or 0.0) for q in query_records)
    sublayer_reused = sum(float(q.get("prefix_vlm_cache_sublayer_reused") or 0.0) for q in query_records)
    sublayer_total = sum(float(q.get("prefix_vlm_cache_sublayer_total") or 0.0) for q in query_records)
    cross_layer_residual_reused = sum(
        float(q.get("prefix_vlm_cache_cross_layer_residual_reused") or 0.0) for q in query_records
    )
    cross_layer_residual_total = sum(
        float(q.get("prefix_vlm_cache_cross_layer_residual_total") or 0.0) for q in query_records
    )
    intra_action_cached_steps = sum(float(q.get("intra_action_cached_steps") or 0.0) for q in query_records)
    intra_action_refresh_steps = sum(float(q.get("intra_action_refresh_steps") or 0.0) for q in query_records)
    intra_action_total_steps = sum(float(q.get("intra_action_total_steps") or 0.0) for q in query_records)
    intra_action_computed_blocks = sum(float(q.get("intra_action_computed_blocks") or 0.0) for q in query_records)
    intra_action_skipped_blocks = sum(float(q.get("intra_action_skipped_blocks") or 0.0) for q in query_records)
    prefix_hit_reuse_fraction = [
        float(q["prefix_vlm_cache_reused_block_fraction"])
        for q in query_records
        if q.get("prefix_vlm_cache_hit") and q.get("prefix_vlm_cache_reused_block_fraction") is not None
    ]
    action_deviation_records = [
        q for q in query_records if q.get("prefix_vlm_cache_action_deviation_evaluated")
    ]
    action_deviation_safe_labels = [
        bool(q["prefix_vlm_cache_action_deviation_safe_label"])
        for q in action_deviation_records
        if q.get("prefix_vlm_cache_action_deviation_safe_label") is not None
    ]
    action_deviation_flips = [
        bool(q["prefix_vlm_cache_action_deviation_gripper_flip"])
        for q in action_deviation_records
        if q.get("prefix_vlm_cache_action_deviation_gripper_flip") is not None
    ]
    action_deviation_candidate_count = sum(
        1 for q in query_records if q.get("prefix_vlm_cache_action_deviation_candidate")
    )

    def _action_deviation_mean(key: str) -> float | None:
        return _mean([float(q[key]) for q in action_deviation_records if q.get(key) is not None])

    scheduler_actions: dict[str, int] = {}
    motion_tiers: dict[str, int] = {}
    no_image_decision_reasons: dict[str, int] = {}
    action_deviation_gate_reasons: dict[str, int] = {}
    for q in query_records:
        action = q.get("threshold_scheduler_ah_action")
        if action is not None:
            scheduler_actions[str(action)] = scheduler_actions.get(str(action), 0) + 1
        tier = q.get("prefix_vlm_cache_motion_tier")
        if tier is not None:
            motion_tiers[str(tier)] = motion_tiers.get(str(tier), 0) + 1
        no_image_reason = q.get("prefix_vlm_cache_skip_v4_no_image_decision_reason")
        if no_image_reason is not None:
            no_image_decision_reasons[str(no_image_reason)] = (
                no_image_decision_reasons.get(str(no_image_reason), 0) + 1
            )
        gate_reason = q.get("prefix_vlm_cache_action_deviation_gate_reason")
        if gate_reason is not None:
            action_deviation_gate_reasons[str(gate_reason)] = action_deviation_gate_reasons.get(str(gate_reason), 0) + 1
    payload = {
        "args": vars(args),
        "summary": {
            "episodes": len(episodes),
            "successes": successes,
            "success_rate": float(successes / len(episodes)) if episodes else None,
            "policy_queries": len(query_records),
            "policy_infer_mean_ms": _mean(policy_ms),
            "server_infer_mean_ms": _mean(server_ms),
            "roundtrip_mean_ms": _mean(roundtrip_ms),
            "cache_hit_rate": float(cache_hits / len(query_records)) if query_records else None,
            "cache_bank_selected_rate": float(bank_selected / len(query_records)) if query_records else None,
            "skip_v4_no_image_rate": float(no_image_hits / len(query_records)) if query_records else None,
            "cache_layer_fraction": float(reused_blocks / (reused_blocks + computed_blocks))
            if reused_blocks + computed_blocks
            else None,
            "sublayer_cache_fraction": float(sublayer_reused / sublayer_total) if sublayer_total else None,
            "cross_layer_residual_reuse_fraction": float(cross_layer_residual_reused / cross_layer_residual_total)
            if cross_layer_residual_total
            else None,
            "effective_cache_layer_fraction": float((reused_blocks + 0.5 * sublayer_reused) / (reused_blocks + computed_blocks))
            if reused_blocks + computed_blocks
            else None,
            "intra_action_cached_step_fraction": float(intra_action_cached_steps / intra_action_total_steps)
            if intra_action_total_steps
            else None,
            "intra_action_refresh_step_fraction": float(intra_action_refresh_steps / intra_action_total_steps)
            if intra_action_total_steps
            else None,
            "intra_action_cache_block_fraction": float(
                intra_action_skipped_blocks / (intra_action_computed_blocks + intra_action_skipped_blocks)
            )
            if intra_action_computed_blocks + intra_action_skipped_blocks
            else None,
            "intra_action_computed_block_fraction": float(
                intra_action_computed_blocks / (intra_action_computed_blocks + intra_action_skipped_blocks)
            )
            if intra_action_computed_blocks + intra_action_skipped_blocks
            else None,
            "intra_action_computed_blocks": int(intra_action_computed_blocks),
            "intra_action_skipped_blocks": int(intra_action_skipped_blocks),
            "prefix_hit_reused_block_fraction_mean": _mean(prefix_hit_reuse_fraction),
            "action_deviation_candidate_count": int(action_deviation_candidate_count),
            "action_deviation_candidate_rate": float(action_deviation_candidate_count / len(query_records))
            if query_records
            else None,
            "action_deviation_evaluated_count": int(len(action_deviation_records)),
            "action_deviation_evaluated_rate": float(len(action_deviation_records) / len(query_records))
            if query_records
            else None,
            "action_deviation_safe_label_rate": float(
                sum(action_deviation_safe_labels) / len(action_deviation_safe_labels)
            )
            if action_deviation_safe_labels
            else None,
            "action_deviation_gripper_flip_rate": float(sum(action_deviation_flips) / len(action_deviation_flips))
            if action_deviation_flips
            else None,
            "action_deviation_l2_mean": _action_deviation_mean("prefix_vlm_cache_action_deviation_l2"),
            "action_deviation_rmse_mean": _action_deviation_mean("prefix_vlm_cache_action_deviation_rmse"),
            "action_deviation_linf_mean": _action_deviation_mean("prefix_vlm_cache_action_deviation_linf"),
            "action_deviation_translation_l2_mean": _action_deviation_mean(
                "prefix_vlm_cache_action_deviation_translation_l2"
            ),
            "action_deviation_rotation_l2_mean": _action_deviation_mean(
                "prefix_vlm_cache_action_deviation_rotation_l2"
            ),
            "action_deviation_eef_position_l2_mean": _action_deviation_mean(
                "prefix_vlm_cache_action_deviation_eef_position_l2"
            ),
            "action_deviation_eef_rotation_l2_mean": _action_deviation_mean(
                "prefix_vlm_cache_action_deviation_eef_rotation_l2"
            ),
            "action_deviation_eef_pose_l2_mean": _action_deviation_mean(
                "prefix_vlm_cache_action_deviation_eef_pose_l2"
            ),
            "action_deviation_gripper_abs_mean": _action_deviation_mean(
                "prefix_vlm_cache_action_deviation_gripper_abs"
            ),
            "action_deviation_gate_enabled_count": sum(
                1 for q in query_records if q.get("prefix_vlm_cache_action_deviation_gate_enabled")
            ),
            "action_deviation_gate_accept_rate": float(
                sum(1 for q in query_records if q.get("prefix_vlm_cache_action_deviation_gate_accept"))
                / sum(1 for q in query_records if q.get("prefix_vlm_cache_action_deviation_gate_enabled"))
            )
            if any(q.get("prefix_vlm_cache_action_deviation_gate_enabled") for q in query_records)
            else None,
            "action_deviation_gate_reasons": action_deviation_gate_reasons,
            "threshold_scheduler_ah_actions": scheduler_actions,
            "motion_tiers": motion_tiers,
            "skip_v4_no_image_decision_reasons": no_image_decision_reasons,
        },
        "episodes": episodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    selected_task_ids = _task_ids(args.task_ids, task_suite.n_tasks)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else _default_max_steps(args.task_suite_name)
    output_path = pathlib.Path(args.output)
    if args.save_video:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
        import imageio
    else:
        imageio = None

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    episodes: list[dict[str, Any]] = []

    for task_id in tqdm.tqdm(selected_task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            if args.trial_start < 0:
                raise ValueError(f"trial-start must be non-negative, got {args.trial_start}")
            trial_indices = range(args.trial_start, args.trial_start + args.num_trials_per_task)
            for episode_idx in tqdm.tqdm(trial_indices, desc=f"task {task_id}"):
                client.reset()
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
                done = False
                t = 0
                replay_images = []
                queries: list[dict[str, Any]] = []
                episode_start = time.perf_counter()

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    if args.save_video:
                        replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        query_start = time.perf_counter()
                        response = client.infer(element)
                        roundtrip_ms = (time.perf_counter() - query_start) * 1000.0
                        timing = dict(response.get("policy_timing", {}))
                        server_timing = dict(response.get("server_timing", {}))
                        queries.append(
                            {
                                "t": int(t),
                                "roundtrip_ms": float(roundtrip_ms),
                                "policy_infer_ms": timing.get("infer_ms"),
                                "server_infer_ms": server_timing.get("infer_ms"),
                                "prefix_vlm_cache_hit": timing.get("prefix_vlm_cache_hit"),
                                "prefix_vlm_cache_reason": timing.get("prefix_vlm_cache_reason"),
                                "prefix_vlm_cache_computed_blocks": timing.get("prefix_vlm_cache_computed_blocks"),
                                "prefix_vlm_cache_skipped_blocks": timing.get("prefix_vlm_cache_skipped_blocks"),
                                "prefix_vlm_cache_compute_layer_mask": timing.get(
                                    "prefix_vlm_cache_compute_layer_mask"
                                ),
                                "prefix_vlm_cache_computed_block_fraction": timing.get(
                                    "prefix_vlm_cache_computed_block_fraction"
                                ),
                                "prefix_vlm_cache_reused_blocks": timing.get("prefix_vlm_cache_reused_blocks"),
                                "prefix_vlm_cache_reused_block_fraction": timing.get(
                                    "prefix_vlm_cache_reused_block_fraction"
                                ),
                                "prefix_vlm_cache_first_layer_cka": timing.get("prefix_vlm_cache_first_layer_cka"),
                                "prefix_vlm_cache_state_guard_enabled": timing.get(
                                    "prefix_vlm_cache_state_guard_enabled"
                                ),
                                "prefix_vlm_cache_state_guard_reason": timing.get(
                                    "prefix_vlm_cache_state_guard_reason"
                                ),
                                "prefix_vlm_cache_state_guard_accept": timing.get(
                                    "prefix_vlm_cache_state_guard_accept"
                                ),
                                "prefix_vlm_cache_query": timing.get("prefix_vlm_cache_query"),
                                "prefix_vlm_cache_episode_query": timing.get("prefix_vlm_cache_episode_query"),
                                "prefix_vlm_cache_state_delta": timing.get("prefix_vlm_cache_state_delta"),
                                "prefix_vlm_cache_state_translation_delta": timing.get(
                                    "prefix_vlm_cache_state_translation_delta"
                                ),
                                "prefix_vlm_cache_state_translation_window_delta": timing.get(
                                    "prefix_vlm_cache_state_translation_window_delta"
                                ),
                                "prefix_vlm_cache_state_rotation_delta": timing.get(
                                    "prefix_vlm_cache_state_rotation_delta"
                                ),
                                "prefix_vlm_cache_state_gripper_delta": timing.get(
                                    "prefix_vlm_cache_state_gripper_delta"
                                ),
                                "prefix_vlm_cache_cached_state_delta": timing.get(
                                    "prefix_vlm_cache_cached_state_delta"
                                ),
                                "prefix_vlm_cache_stable_state_deltas": timing.get(
                                    "prefix_vlm_cache_stable_state_deltas"
                                ),
                                "prefix_vlm_cache_next_stable_state_deltas": timing.get(
                                    "prefix_vlm_cache_next_stable_state_deltas"
                                ),
                                "prefix_vlm_cache_skip_v4_no_image": timing.get("prefix_vlm_cache_skip_v4_no_image"),
                                "prefix_vlm_cache_skip_v4_no_image_decision_reason": timing.get(
                                    "prefix_vlm_cache_skip_v4_no_image_decision_reason"
                                ),
                                "prefix_vlm_cache_skip_v4_no_image_remaining": timing.get(
                                    "prefix_vlm_cache_skip_v4_no_image_remaining"
                                ),
                                "prefix_vlm_cache_observe_only": timing.get("prefix_vlm_cache_observe_only"),
                                "prefix_vlm_cache_shadow_hit": timing.get("prefix_vlm_cache_shadow_hit"),
                                "prefix_vlm_cache_shadow_reason": timing.get("prefix_vlm_cache_shadow_reason"),
                                "prefix_vlm_cache_shadow_computed_blocks": timing.get(
                                    "prefix_vlm_cache_shadow_computed_blocks"
                                ),
                                "prefix_vlm_cache_shadow_skipped_blocks": timing.get(
                                    "prefix_vlm_cache_shadow_skipped_blocks"
                                ),
                                "prefix_vlm_cache_shadow_reused_blocks": timing.get(
                                    "prefix_vlm_cache_shadow_reused_blocks"
                                ),
                                "prefix_vlm_cache_shadow_reused_block_fraction": timing.get(
                                    "prefix_vlm_cache_shadow_reused_block_fraction"
                                ),
                                "prefix_vlm_cache_shadow_compute_layer_mask": timing.get(
                                    "prefix_vlm_cache_shadow_compute_layer_mask"
                                ),
                                "prefix_vlm_cache_pairwise_enabled": timing.get(
                                    "prefix_vlm_cache_pairwise_enabled"
                                ),
                                "prefix_vlm_cache_pairwise_query_index": timing.get(
                                    "prefix_vlm_cache_pairwise_query_index"
                                ),
                                "prefix_vlm_cache_pairwise_cka_row": timing.get(
                                    "prefix_vlm_cache_pairwise_cka_row"
                                ),
                                "prefix_vlm_cache_pairwise_truncated": timing.get(
                                    "prefix_vlm_cache_pairwise_truncated"
                                ),
                                "prefix_vlm_cache_layerwise_enabled": timing.get(
                                    "prefix_vlm_cache_layerwise_enabled"
                                ),
                                "prefix_vlm_cache_layerwise_residual_cka": timing.get(
                                    "prefix_vlm_cache_layerwise_residual_cka"
                                ),
                                "prefix_vlm_cache_layerwise_residual_l2": timing.get(
                                    "prefix_vlm_cache_layerwise_residual_l2"
                                ),
                                "prefix_vlm_cache_layerwise_key_l2": timing.get(
                                    "prefix_vlm_cache_layerwise_key_l2"
                                ),
                                "prefix_vlm_cache_layerwise_value_l2": timing.get(
                                    "prefix_vlm_cache_layerwise_value_l2"
                                ),
                                "prefix_vlm_cache_layerwise_reuse_mask": timing.get(
                                    "prefix_vlm_cache_layerwise_reuse_mask"
                                ),
                                "prefix_vlm_cache_layerwise_reuse_blocks": timing.get(
                                    "prefix_vlm_cache_layerwise_reuse_blocks"
                                ),
                                "prefix_vlm_cache_layer_pairwise_enabled": timing.get(
                                    "prefix_vlm_cache_layer_pairwise_enabled"
                                ),
                                "prefix_vlm_cache_layer_pairwise_query_index": timing.get(
                                    "prefix_vlm_cache_layer_pairwise_query_index"
                                ),
                                "prefix_vlm_cache_layer_pairwise_cka_rows": timing.get(
                                    "prefix_vlm_cache_layer_pairwise_cka_rows"
                                ),
                                "prefix_vlm_cache_layer_pairwise_truncated": timing.get(
                                    "prefix_vlm_cache_layer_pairwise_truncated"
                                ),
                                "prefix_vlm_cache_adjacent_layer_enabled": timing.get(
                                    "prefix_vlm_cache_adjacent_layer_enabled"
                                ),
                                "prefix_vlm_cache_adjacent_layer_residual_cka": timing.get(
                                    "prefix_vlm_cache_adjacent_layer_residual_cka"
                                ),
                                "prefix_vlm_cache_adjacent_layer_residual_l2": timing.get(
                                    "prefix_vlm_cache_adjacent_layer_residual_l2"
                                ),
                                "prefix_vlm_cache_lagged_adjacent_layer_enabled": timing.get(
                                    "prefix_vlm_cache_lagged_adjacent_layer_enabled"
                                ),
                                "prefix_vlm_cache_lagged_adjacent_layer_residual_cka": timing.get(
                                    "prefix_vlm_cache_lagged_adjacent_layer_residual_cka"
                                ),
                                "prefix_vlm_cache_lagged_adjacent_layer_residual_l2": timing.get(
                                    "prefix_vlm_cache_lagged_adjacent_layer_residual_l2"
                                ),
                                "prefix_vlm_cache_sublayer_enabled": timing.get(
                                    "prefix_vlm_cache_sublayer_enabled"
                                ),
                                "prefix_vlm_cache_sublayer_attn_delta_cka": timing.get(
                                    "prefix_vlm_cache_sublayer_attn_delta_cka"
                                ),
                                "prefix_vlm_cache_sublayer_attn_delta_l2": timing.get(
                                    "prefix_vlm_cache_sublayer_attn_delta_l2"
                                ),
                                "prefix_vlm_cache_sublayer_mlp_delta_cka": timing.get(
                                    "prefix_vlm_cache_sublayer_mlp_delta_cka"
                                ),
                                "prefix_vlm_cache_sublayer_mlp_delta_l2": timing.get(
                                    "prefix_vlm_cache_sublayer_mlp_delta_l2"
                                ),
                                "prefix_vlm_cache_sublayer_cache_enabled": timing.get(
                                    "prefix_vlm_cache_sublayer_cache_enabled"
                                ),
                                "prefix_vlm_cache_sublayer_reused": timing.get(
                                    "prefix_vlm_cache_sublayer_reused"
                                ),
                                "prefix_vlm_cache_sublayer_total": timing.get(
                                    "prefix_vlm_cache_sublayer_total"
                                ),
                                "prefix_vlm_cache_sublayer_reused_fraction": timing.get(
                                    "prefix_vlm_cache_sublayer_reused_fraction"
                                ),
                                "prefix_vlm_cache_sublayer_attn_mask": timing.get(
                                    "prefix_vlm_cache_sublayer_attn_mask"
                                ),
                                "prefix_vlm_cache_sublayer_mlp_mask": timing.get(
                                    "prefix_vlm_cache_sublayer_mlp_mask"
                                ),
                                "prefix_vlm_cache_sublayer_cka_gate_accept": timing.get(
                                    "prefix_vlm_cache_sublayer_cka_gate_accept"
                                ),
                                "prefix_vlm_cache_sublayer_cka_gate_threshold": timing.get(
                                    "prefix_vlm_cache_sublayer_cka_gate_threshold"
                                ),
                                "prefix_vlm_cache_cross_layer_residual_cache_enabled": timing.get(
                                    "prefix_vlm_cache_cross_layer_residual_cache_enabled"
                                ),
                                "prefix_vlm_cache_cross_layer_residual_reused": timing.get(
                                    "prefix_vlm_cache_cross_layer_residual_reused"
                                ),
                                "prefix_vlm_cache_cross_layer_residual_total": timing.get(
                                    "prefix_vlm_cache_cross_layer_residual_total"
                                ),
                                "prefix_vlm_cache_cross_layer_residual_reused_fraction": timing.get(
                                    "prefix_vlm_cache_cross_layer_residual_reused_fraction"
                                ),
                                "prefix_vlm_cache_cross_layer_residual_mask": timing.get(
                                    "prefix_vlm_cache_cross_layer_residual_mask"
                                ),
                                "prefix_vlm_cache_bank_enabled": timing.get("prefix_vlm_cache_bank_enabled"),
                                "prefix_vlm_cache_bank_size": timing.get("prefix_vlm_cache_bank_size"),
                                "prefix_vlm_cache_bank_selected": timing.get("prefix_vlm_cache_bank_selected"),
                                "prefix_vlm_cache_bank_selected_cka": timing.get(
                                    "prefix_vlm_cache_bank_selected_cka"
                                ),
                                "prefix_vlm_cache_motion_tier_enabled": timing.get(
                                    "prefix_vlm_cache_motion_tier_enabled"
                                ),
                                "prefix_vlm_cache_motion_tier": timing.get("prefix_vlm_cache_motion_tier"),
                                "prefix_vlm_cache_motion_signal": timing.get("prefix_vlm_cache_motion_signal"),
                                "prefix_vlm_cache_motion_tier_cka_threshold": timing.get(
                                    "prefix_vlm_cache_motion_tier_cka_threshold"
                                ),
                                "prefix_vlm_cache_motion_tier_cached_state_delta_threshold": timing.get(
                                    "prefix_vlm_cache_motion_tier_cached_state_delta_threshold"
                                ),
                                "prefix_vlm_cache_motion_tier_prompt_guard": timing.get(
                                    "prefix_vlm_cache_motion_tier_prompt_guard"
                                ),
                                "prefix_vlm_cache_motion_tier_no_image_budget": timing.get(
                                    "prefix_vlm_cache_motion_tier_no_image_budget"
                                ),
                                "prefix_vlm_cache_motion_tier_no_image_min_speed": timing.get(
                                    "prefix_vlm_cache_motion_tier_no_image_min_speed"
                                ),
                                "prefix_vlm_cache_motion_tier_no_image_cached_state_delta_threshold": timing.get(
                                    "prefix_vlm_cache_motion_tier_no_image_cached_state_delta_threshold"
                                ),
                                "threshold_scheduler_ah_action": timing.get(
                                    "prefix_vlm_cache_threshold_scheduler_ah_action"
                                ),
                                "threshold_scheduler_ah_motion_signal": timing.get(
                                    "prefix_vlm_cache_threshold_scheduler_ah_motion_signal"
                                ),
                                "threshold_scheduler_ah_fn_blocks": timing.get(
                                    "prefix_vlm_cache_threshold_scheduler_ah_fn_blocks"
                                ),
                                "threshold_scheduler_ah_bn_blocks": timing.get(
                                    "prefix_vlm_cache_threshold_scheduler_ah_bn_blocks"
                                ),
                                "threshold_scheduler_ah_schedule": timing.get(
                                    "prefix_vlm_cache_threshold_scheduler_ah_schedule"
                                ),
                                "threshold_scheduler_ah_refresh_interval": timing.get(
                                    "prefix_vlm_cache_threshold_scheduler_ah_refresh_interval"
                                ),
                                "intra_action_cached_steps": timing.get("prefix_vlm_cache_intra_action_cached_steps"),
                                "intra_action_refresh_steps": timing.get("prefix_vlm_cache_intra_action_refresh_steps"),
                                "intra_action_total_steps": timing.get("prefix_vlm_cache_intra_action_total_steps"),
                                "intra_action_fn_blocks": timing.get("prefix_vlm_cache_intra_action_fn_blocks"),
                                "intra_action_bn_blocks": timing.get("prefix_vlm_cache_intra_action_bn_blocks"),
                                "intra_action_schedule": timing.get("prefix_vlm_cache_intra_action_schedule"),
                                "intra_action_refresh_interval": timing.get(
                                    "prefix_vlm_cache_intra_action_refresh_interval"
                                ),
                                "intra_action_computed_blocks": timing.get(
                                    "prefix_vlm_cache_intra_action_computed_blocks"
                                ),
                                "intra_action_skipped_blocks": timing.get(
                                    "prefix_vlm_cache_intra_action_skipped_blocks"
                                ),
                                "intra_action_computed_block_fraction": timing.get(
                                    "prefix_vlm_cache_intra_action_computed_block_fraction"
                                ),
                            }
                        )
                        for key in ACTION_DEVIATION_TIMING_KEYS:
                            queries[-1][key] = timing.get(key)
                        action_chunk = response["actions"]
                        if len(action_chunk) < args.replan_steps:
                            raise RuntimeError(
                                f"Need at least {args.replan_steps} actions, got {len(action_chunk)}"
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _reward, done, _info = env.step(action.tolist())
                    if done:
                        break
                    t += 1

                episode = {
                    "task_id": int(task_id),
                    "task_description": str(task_description),
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "steps": int(t),
                    "duration_s": float(time.perf_counter() - episode_start),
                    "policy_queries": int(len(queries)),
                    "queries": queries,
                }
                episodes.append(episode)
                if args.save_video and imageio is not None:
                    suffix = "success" if done else "failure"
                    task_segment = task_description.replace(" ", "_")
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path)
                        / f"rollout_task{task_id}_{episode_idx}_{task_segment}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                _write_output(output_path, args, episodes)
                logging.info("episode task=%s idx=%s success=%s queries=%s", task_id, episode_idx, done, len(queries))
        finally:
            env.close()

    _write_output(output_path, args, episodes)
    logging.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
