"""Single-stream OpenPI baseline or ElegantVLA latency smoke test."""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch
from openpi.policies import libero_policy, policy_config
from openpi.training import config as _config

from elegantvla.openpi.core.config import TemporalSkipConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default=os.environ.get("OPENPI_CHECKPOINT"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-layers", type=int, default=1)
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--max-reuse-steps", type=int, default=5)
    parser.add_argument("--gate-threshold", type=float, default=0.95)
    parser.add_argument("--free1-threshold", type=float, default=0.98)
    parser.add_argument("--free2-threshold", type=float, default=0.99)
    parser.add_argument("--free3-threshold", type=float, default=0.995)
    args = parser.parse_args()
    if not args.checkpoint or args.queries < 1 or args.warmup < 0:
        parser.error("A checkpoint, positive query count and nonnegative warmup are required.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = TemporalSkipConfig(
        enabled=not args.baseline,
        first_layers=args.first_layers,
        last_layers=args.last_layers,
        max_reuse_steps=args.max_reuse_steps,
        gate_threshold=args.gate_threshold,
        free1_similarity_threshold=args.free1_threshold,
        free2_similarity_threshold=args.free2_threshold,
        free3_similarity_threshold=args.free3_threshold,
    )
    train = _config.get_config(args.config)
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(
        train,
        args.checkpoint,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_steps},
        temporal_skip=config,
    )
    observation = libero_policy.make_libero_example()
    model_config = policy._model.config
    noise = np.random.default_rng(args.seed).standard_normal(
        (model_config.action_horizon, model_config.action_dim), dtype=np.float32
    )

    def synchronize():
        if policy._is_pytorch_model and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)

    records = []
    for index in range(args.warmup + args.queries):
        if index == args.warmup:
            policy.reset()
        synchronize()
        start = time.perf_counter()
        output = policy.infer(observation, noise=noise)
        synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        if index >= args.warmup:
            records.append({"query": index - args.warmup + 1, "elapsed_ms": elapsed, **output.get("policy_timing", {})})
    paths = {}
    for path in sorted({record.get("path", "baseline") for record in records}):
        times = [record["elapsed_ms"] for record in records if record.get("path", "baseline") == path]
        paths[path] = {"count": len(times), "p50_ms": float(np.median(times)), "mean_ms": float(np.mean(times))}
    print(
        json.dumps(
            {
                "config": dataclasses.asdict(config),
                "device": args.device,
                "num_steps": args.num_steps,
                "paths": paths,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
