#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import tyro

from elegantvla.gr00t.entrypoints._runtime import build_temporal_skip_env, seed_episode_rng
from elegantvla.gr00t.entrypoints.evaluate import RunConfig


@dataclass
class BenchmarkConfig:
    model_path: str
    env_name: str = "simpler_env_google/google_robot_pick_coke_can"
    device: str = "cuda"
    seed: int = 0
    warmup: int = 3
    iterations: int = 20
    output_path: str | None = None


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _benchmark_probe(
    adapter: Any,
    collated_inputs: dict[str, Any],
    *,
    cache_enabled: bool,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], torch.Tensor]:
    adapter.configure_static_inference_cache(cache_enabled)
    for _ in range(max(int(warmup), 0)):
        adapter.get_first_block_gate_probe(collated_inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    durations_ms: list[float] = []
    output = None
    for _ in range(max(int(iterations), 1)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        output, _ = adapter.get_first_block_gate_probe(collated_inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations_ms.append(float((time.perf_counter() - started_at) * 1000.0))

    assert output is not None
    return {
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "p10_ms": _percentile(durations_ms, 10.0),
        "p90_ms": _percentile(durations_ms, 90.0),
        "iterations": int(len(durations_ms)),
    }, output["hidden_states"].detach().clone()


def _benchmark_prepare_inputs(
    adapter: Any,
    policy_observation: dict[str, Any],
    *,
    cache_enabled: bool,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    adapter.configure_static_inference_cache(cache_enabled)
    for _ in range(max(int(warmup), 0)):
        adapter.prepare_inputs(policy_observation)

    durations_ms: list[float] = []
    prepared = None
    for _ in range(max(int(iterations), 1)):
        started_at = time.perf_counter()
        prepared = adapter.prepare_inputs(policy_observation)
        durations_ms.append(float((time.perf_counter() - started_at) * 1000.0))

    assert prepared is not None
    return {
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "p10_ms": _percentile(durations_ms, 10.0),
        "p90_ms": _percentile(durations_ms, 90.0),
        "iterations": int(len(durations_ms)),
    }, prepared["collated_inputs"]


@torch.inference_mode()
def main(cfg: BenchmarkConfig) -> None:
    seed_episode_rng(cfg.seed)
    run_cfg = RunConfig(
        model_path=cfg.model_path,
        env_name=cfg.env_name,
        episodes=1,
        seed=cfg.seed,
        device=cfg.device,
        compile_action_head=False,
        static_inference_cache=True,
    )
    env = build_temporal_skip_env(run_cfg, device_attr="device")
    try:
        env.reset(seed=cfg.seed)
        policy_observation = env._build_policy_observation(env._sim_obs)
        adapter = env.wrapper.adapter
        prepare_disabled, disabled_inputs = _benchmark_prepare_inputs(
            adapter,
            policy_observation,
            cache_enabled=False,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
        )
        prepare_enabled, collated_inputs = _benchmark_prepare_inputs(
            adapter,
            policy_observation,
            cache_enabled=True,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
        )

        disabled, disabled_output = _benchmark_probe(
            adapter,
            collated_inputs,
            cache_enabled=False,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
        )
        enabled, enabled_output = _benchmark_probe(
            adapter,
            collated_inputs,
            cache_enabled=True,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
        )
        difference = (disabled_output.float() - enabled_output.float()).abs()
        result = {
            "config": asdict(cfg),
            "cache_disabled": disabled,
            "cache_enabled": enabled,
            "prepare_cache_disabled": prepare_disabled,
            "prepare_cache_enabled": prepare_enabled,
            "prepare_mean_speedup": float(
                prepare_disabled["mean_ms"] / prepare_enabled["mean_ms"]
            ),
            "prepare_mean_delta_ms": float(
                prepare_enabled["mean_ms"] - prepare_disabled["mean_ms"]
            ),
            "prepare_input_ids_exact_equal": bool(
                torch.equal(disabled_inputs["input_ids"], collated_inputs["input_ids"])
            ),
            "prepare_attention_mask_exact_equal": bool(
                torch.equal(
                    disabled_inputs["attention_mask"],
                    collated_inputs["attention_mask"],
                )
            ),
            "mean_speedup": float(disabled["mean_ms"] / enabled["mean_ms"]),
            "mean_delta_ms": float(enabled["mean_ms"] - disabled["mean_ms"]),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "exact_equal": bool(torch.equal(disabled_output, enabled_output)),
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if cfg.output_path is not None:
            output_path = Path(cfg.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
