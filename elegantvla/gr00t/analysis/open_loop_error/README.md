# Fixed-trajectory backbone cache error

This evaluator treats current-frame full computation as the ground truth and
compares two one-frame cache approximations on the same SimplerEnv observations:

1. `full_cache`: directly reuse the complete previous-frame backbone cache.
2. `first_last_correction`: recompute the current first layer, add the previous
   middle-layer delta cache, and recompute the current last layer.

The action head is fully recomputed for all three paths. Each path receives the
same per-step random seed, so action error isolates the backbone approximation.
The previous-frame caches always come from the ground-truth path, avoiding
compounded cache drift. Step zero only primes the caches and is excluded from the
aggregate comparison.

By default, the observation set is a controlled sweep derived from a real SimplerEnv
render. Each step translates the rendered frame by one pixel while state and language
remain fixed. This is a deterministic camera-motion proxy for validating the cache-error
measurement, not a substitute for final evidence on real Bridge trajectories. It records
image MAE between adjacent frames and rejects trajectories with too few changed frames.
`scene_sweep`, `reset_sequence`, `physics_only`, and `policy_action` remain available as
diagnostic drivers.

Outputs include representation MAE/MSE, full action-chunk error, executed-action
prefix error, per-action-dimension errors, relative error reduction, the fixed
trajectory, and `cache_comparison.png`.

```bash
GPU=0 MODEL_PATH=/path/to/checkpoint MAX_TRAJECTORY_STEPS=40 \
  bash elegantvla/gr00t/scripts/run_open_loop.sh
```

Saved trajectories can be replayed through the rule thresholds without running the
simulator again:

```bash
python -m elegantvla.gr00t.analysis.open_loop_error.compare_thresholded_cache --help
```
