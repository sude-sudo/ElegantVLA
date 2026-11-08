import argparse
import dataclasses
import json
import os
import time

import numpy as np

from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config
from openpi.training import config as _config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke/latency probe for OpenPI ElegantVLA image-free reuse.")
    parser.add_argument("--config", default="pi05_libero")
    checkpoint_default = os.environ.get("OPENPI_CHECKPOINT")
    parser.add_argument(
        "--checkpoint",
        default=checkpoint_default,
        required=checkpoint_default is None,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-layer-mask", default="0,1,16,17")
    parser.add_argument("--compute-layer-strategy", default="first_last")
    parser.add_argument("--compute-layer-count", type=int, default=0)
    parser.add_argument("--no-image-budget", type=int, default=1)
    parser.add_argument("--max-continuous", type=int, default=4)
    parser.add_argument("--first-layer-cka-threshold", type=float, default=0.0)
    parser.add_argument("--threshold-scheduler", action="store_true")
    parser.add_argument("--scheduler-max-no-image-budget", type=int, default=1)
    parser.add_argument("--free1-cka-threshold", type=float, default=0.999)
    parser.add_argument("--free2-cka-threshold", type=float, default=0.9995)
    parser.add_argument("--free3-cka-threshold", type=float, default=0.9998)
    parser.add_argument("--action-head-scheduler", action="store_true")
    parser.add_argument("--action-head-selection-mode", default="motion_speed_inner")
    parser.add_argument("--ah-speed-threshold", type=float, default=0.005)
    parser.add_argument("--ah-secondary-threshold", type=float, default=0.05)
    parser.add_argument("--intra-action-cache", action="store_true")
    parser.add_argument("--intra-action-compile", action="store_true")
    parser.add_argument("--intra-action-compile-mode", default="max-autotune")
    parser.add_argument("--intra-action-refresh-interval", type=int, default=3)
    parser.add_argument("--intra-action-fn-blocks", type=int, default=2)
    parser.add_argument("--intra-action-bn-blocks", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_config = _config.get_config(args.config)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
    )
    cache = _policy.PrefixVLMCacheConfig(
        enabled=True,
        compute_layer_mask=args.compute_layer_mask or None,
        compute_layer_strategy=args.compute_layer_strategy,
        compute_layer_count=args.compute_layer_count,
        max_warmup_queries=1,
        max_continuous_cached_queries=args.max_continuous,
        image_signature_l2_threshold=1.0e9,
        first_layer_metric="cka",
        first_layer_cka_threshold=args.first_layer_cka_threshold,
        skip_v4_no_image_budget_after_hit=args.no_image_budget,
        threshold_scheduler_enabled=bool(args.threshold_scheduler),
        threshold_scheduler_max_no_image_budget=args.scheduler_max_no_image_budget,
        threshold_scheduler_free1_cka_threshold=args.free1_cka_threshold,
        threshold_scheduler_free2_cka_threshold=args.free2_cka_threshold,
        threshold_scheduler_free3_cka_threshold=args.free3_cka_threshold,
        threshold_scheduler_action_head_enabled=bool(args.action_head_scheduler),
        threshold_scheduler_action_head_selection_mode=args.action_head_selection_mode,
        threshold_scheduler_ah_speed_threshold=args.ah_speed_threshold,
        threshold_scheduler_ah_secondary_threshold=args.ah_secondary_threshold,
        intra_action_cache_enabled=bool(args.intra_action_cache or args.intra_action_compile),
        intra_action_cache_fn_compute_blocks=args.intra_action_fn_blocks,
        intra_action_cache_bn_compute_blocks=args.intra_action_bn_blocks,
        intra_action_cache_refresh_interval=args.intra_action_refresh_interval,
        intra_action_cache_compile=bool(args.intra_action_compile),
        intra_action_cache_compile_mode=args.intra_action_compile_mode,
    )
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_steps},
        prefix_vlm_cache=cache,
    )
    raw_obs = libero_policy.make_libero_example()
    rng = np.random.default_rng(args.seed)
    action_horizon = int(policy._model.config.action_horizon)  # noqa: SLF001
    action_dim = int(policy._model.config.action_dim)  # noqa: SLF001
    noise = rng.standard_normal((action_horizon, action_dim), dtype=np.float32)

    records = []
    total_queries = args.warmup + args.queries
    for query_idx in range(total_queries):
        start = time.perf_counter()
        out = policy.infer(raw_obs, noise=noise)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timing = dict(out.get("policy_timing", {}))
        records.append(
            {
                "query": query_idx + 1,
                "phase": "warmup" if query_idx < args.warmup else "measure",
                "elapsed_ms": elapsed_ms,
                "infer_ms": timing.get("infer_ms"),
                "hit": timing.get("prefix_vlm_cache_hit"),
                "reason": timing.get("prefix_vlm_cache_reason"),
                "computed_blocks": timing.get("prefix_vlm_cache_computed_blocks"),
                "skipped_blocks": timing.get("prefix_vlm_cache_skipped_blocks"),
                "compute_layer_mask": timing.get("prefix_vlm_cache_compute_layer_mask"),
                "computed_block_fraction": timing.get("prefix_vlm_cache_computed_block_fraction"),
                "reused_blocks": timing.get("prefix_vlm_cache_reused_blocks"),
                "reused_block_fraction": timing.get("prefix_vlm_cache_reused_block_fraction"),
                "skip_v4_no_image": timing.get("prefix_vlm_cache_skip_v4_no_image"),
                "skip_v4_no_image_remaining": timing.get("prefix_vlm_cache_skip_v4_no_image_remaining"),
                "continuous_hits": timing.get("prefix_vlm_cache_continuous_hits"),
                "first_layer_cka": timing.get("prefix_vlm_cache_first_layer_cka"),
                "intra_action_cached_steps": timing.get("prefix_vlm_cache_intra_action_cached_steps"),
                "intra_action_refresh_steps": timing.get("prefix_vlm_cache_intra_action_refresh_steps"),
                "intra_action_computed_blocks": timing.get("prefix_vlm_cache_intra_action_computed_blocks"),
                "scheduler_ah_action": timing.get("prefix_vlm_cache_threshold_scheduler_ah_action"),
                "scheduler_ah_motion_signal": timing.get("prefix_vlm_cache_threshold_scheduler_ah_motion_signal"),
                "scheduler_ah_fn_blocks": timing.get("prefix_vlm_cache_threshold_scheduler_ah_fn_blocks"),
                "scheduler_ah_bn_blocks": timing.get("prefix_vlm_cache_threshold_scheduler_ah_bn_blocks"),
                "state_translation_delta": timing.get("prefix_vlm_cache_state_translation_delta"),
                "state_rotation_delta": timing.get("prefix_vlm_cache_state_rotation_delta"),
                "state_gripper_delta": timing.get("prefix_vlm_cache_state_gripper_delta"),
            }
        )


    measured = [record for record in records if record["phase"] == "measure"]
    measured_elapsed = np.array([record["elapsed_ms"] for record in measured], dtype=np.float64)
    probed_hit_elapsed = np.array(
        [record["elapsed_ms"] for record in measured if record["hit"] and not record["skip_v4_no_image"]],
        dtype=np.float64,
    )
    no_image_elapsed = np.array(
        [record["elapsed_ms"] for record in measured if record["skip_v4_no_image"]],
        dtype=np.float64,
    )

    def _summary(values):
        if values.size == 0:
            return {"count": 0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
        return {
            "count": int(values.size),
            "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }

    print(
        json.dumps(
            {
                "config": args.config,
                "device": args.device,
                "num_steps": args.num_steps,
                "warmup": args.warmup,
                "queries": args.queries,
                "compute_layer_mask": args.compute_layer_mask,
                "compute_layer_strategy": args.compute_layer_strategy,
                "compute_layer_count": args.compute_layer_count,
                "no_image_budget": args.no_image_budget,
                "max_continuous": args.max_continuous,
                "threshold_scheduler": bool(args.threshold_scheduler),
                "scheduler_max_no_image_budget": args.scheduler_max_no_image_budget,
                "free1_cka_threshold": args.free1_cka_threshold,
                "free2_cka_threshold": args.free2_cka_threshold,
                "free3_cka_threshold": args.free3_cka_threshold,
                "action_head_scheduler": bool(args.action_head_scheduler),
                "ah_speed_threshold": args.ah_speed_threshold,
                "ah_secondary_threshold": args.ah_secondary_threshold,
                "intra_action_cache": bool(args.intra_action_cache or args.intra_action_compile),
                "intra_action_compile": bool(args.intra_action_compile),
                "summary": {
                    "measured": _summary(measured_elapsed),
                    "probed_hit": _summary(probed_hit_elapsed),
                    "skip_v4_no_image": _summary(no_image_elapsed),
                    "cache_hits": sum(1 for record in measured if record["hit"]),
                    "skip_v4_no_image_hits": sum(1 for record in measured if record["skip_v4_no_image"]),
                },
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
