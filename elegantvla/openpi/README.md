# OpenPI Backend

The backend follows the same architecture as GR00T:

```text
Policy.infer (input/output transforms)
  -> TemporalSkipPolicyWrapper.build_schedule_state
  -> RuleSchedulePolicy / RLSchedulePolicy.select_action
  -> TemporalSkipPolicyWrapper.execute_schedule_action
  -> OpenPIPolicyTemporalSkipAdapter
  -> OpenPIActionHead
```

| Core file | Responsibility |
|---|---|
| `types.py` | Shared five VLM actions, three action-head actions, observation/state types |
| `rule_policy.py` | Shared teacher thresholds |
| `rl_policy.py` | Shared trained-SB3-policy inference interface |
| `wrapper.py` | Probe once, resolve the schedule, execute and commit cache state |
| `adapter.py` | OpenPI prefix layers, middle residual and layer-wise prefix K/V |
| `action_head.py` | Action-expert refresh and within-denoising residual reuse |
| `config.py` | Small `TemporalSkipConfig`; compatibility export for old settings |
| `policy.py` | OpenPI transforms, model initialization and public `infer/reset` API |

The shared implementations live in `elegantvla/scheduling`; backend imports are
re-exports, not separate copies. The GR00T code does not import OpenPI, and vice
versa.

## Execution Paths

| Mode | VLM work | Cache |
|---|---|---|
| Refresh | All layers; continue from the already computed probe | Replace the full-refresh anchor |
| Boundary | First layer + cached middle delta + last layer | Keep the anchor; replace the current prefix K/V |
| Image-Free | No vision encoder, probe or VLM layers | Reuse the latest prefix K/V; use the current action input |

The default boundaries are first-one/last-one. `first_layers` and `last_layers`
can widen them. The five VLM choices are refresh, boundary, and boundary followed
by one/two/three image-free queries. Image-free budgets and maximum consecutive
reuse follow the shared controller. Episode reset, instruction/token-layout
changes and invalid middle-cache shapes force refresh.

Image-Free skips the model's visual computation, not the entire input pipeline:
`Policy.infer` still applies input transforms and moves tensors to the model
device before selecting the path. Prefix caches use `DynamicCache` containers;
there is no shared, fully preallocated GPU buffer pool.

Action-head choices are independent: recompute every denoising step, recompute
the first/last denoising steps, or recompute only the first. Reused steps add the
first step's expert-stack residual to the current suffix embeddings, then apply
the current adaptive output normalization and projection. The residual is never
carried across separate policy queries. This is a PyTorch implementation; GR00T's
model-specific fused kernels/CUDA graphs are not claimed to be ported.

## Run

Prepare the upstream **PyTorch** OpenPI environment, including its documented
Transformers replacements. Then, from the ElegantVLA root:

```bash
python -m pip install --no-deps -e .
bash scripts/install_openpi_integration.sh
bash elegantvla/openpi/scripts/run.sh serve --help

# Set OPENPI_CHECKPOINT to a local PyTorch checkpoint.
bash elegantvla/openpi/scripts/run.sh benchmark --baseline
bash elegantvla/openpi/scripts/run.sh benchmark

bash elegantvla/openpi/scripts/run.sh serve --temporal-skip.enabled \
  policy:checkpoint --policy.config pi05_libero --policy.dir "$OPENPI_CHECKPOINT"
```

The benchmark repeats one synthetic LIBERO observation; it is a latency smoke
test, not a success-rate evaluation. `PYTHON_BIN` selects the environment.
The serving path remains baseline unless `temporal_skip.enabled` is enabled.
The new scheduler requires a PyTorch checkpoint; JAX baseline and old experimental
cache paths remain available separately.

The threshold defaults match GR00T's defaults, not a newly tuned OpenPI
evaluation result. Motion defaults to raw LIBERO `observation/state`:
translation `[0:3]`, axis-angle `[3:6]`, gripper index `6`. For another robot,
configure these indices or supply `scheduler_motion` using the shared state keys.
Set `motion_state_key=None` to disable automatic vector interpretation.
When a raw prompt is present, it is the instruction identity; changes to
state tokens embedded in Pi0.5's prefix remain subject to temporal reuse.

## Compatibility And Tests

Old `PrefixVLMCacheConfig`, `skip_v4` trace names, experiment guards and layer-mask
sweeps live in `legacy/`. Explicitly enabling the old config routes to that policy.
Do not enable both configs. The old benchmark is available as
`python -m elegantvla.openpi.legacy.benchmark`.

```bash
PYTHONPATH=.:third_party/openpi/src:third_party/openpi/packages/openpi-client/src \
  OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  python -m unittest discover -s elegantvla/openpi/tests -v
```

The CPU tests use small randomly initialized layers from OpenPI's pinned Gemma
implementation, with no checkpoint downloads or site-packages changes. They
check native full-path equivalence, actual layer calls, K/V freshness, anchor
lifecycle, image-free budgets, failure handling and denoising modes. Full-size
checkpoint, compiled-kernel latency and closed-loop evaluation remain separate
validation steps. The RL inference API is shared, but OpenPI RL training scripts
are not included. A shared API does not establish transferred-policy performance.
